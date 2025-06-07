#!/usr/bin/env python3
# parse_pdf.py - PDF Parser mit Unstructured für LlamaIndex

# 1. Imports
import os
import logging
import sys
import chromadb
import requests
import json
import base64
from typing import List, Dict, Any, Optional
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core import Settings
from llama_index.readers.file import UnstructuredReader
from pathlib import Path
import time
import psutil
import fitz  # PyMuPDF für PDF-Analyse
import signal  # Für Timeout-Handling
import traceback  # Für detaillierte Fehlerausgabe
import torch  # PyTorch für GPU-Prüfung

# Logging einrichten
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# GPU-Status prüfen und ausgeben
def check_gpu_status():
    """Prüft den GPU-Status und gibt Informationen aus."""
    print("\n===== GPU-STATUS =====")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA verfügbar: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Anzahl GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            # GPU-Speichernutzung prüfen, falls verfügbar
            try:
                import nvidia_smi
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                print(f"  Speicher: {info.used//1024//1024} MB / {info.total//1024//1024} MB")
                nvidia_smi.nvmlShutdown()
            except:
                print("  Speicherinfo nicht verfügbar")
    print("======================\n")

# Konfiguration für lokale Unstructured-Nutzung
# Setze auf True, wenn du eine externe Unstructured API verwenden möchtest
USE_EXTERNAL_API = False

# Nur relevant, wenn USE_EXTERNAL_API = True
UNSTRUCTURED_API_URL = os.environ.get(
    "UNSTRUCTURED_API_URL", 
    "http://localhost:8000/general/v0/general"  # Lokale API falls gewünscht
)

# 2. Konfiguration
PDF_FOLDER = "./PDFs"  # Ordner für PDFs
PERSIST_DIR = "./chroma_db_store"
COLLECTION_NAME = "test_collection"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Hierarchical Parser Konfiguration
CHUNK_SIZES_CONFIG = {
    "chunk_size": 512,  # Für Sätze
    "chunk_overlap": 75,
    "chunk_size_small": 256,  # Für kleine Abschnitte
    "chunk_size_large": 1024,  # Für große Abschnitte
}

def create_hierarchical_parser():
    """
    Erstellt einen hierarchischen Parser mit verschiedenen Ebenen.
    """
    # Chunk-Größen in absteigender Reihenfolge
    sizes_list = [
        CHUNK_SIZES_CONFIG["chunk_size_large"],
        CHUNK_SIZES_CONFIG["chunk_size"],
        CHUNK_SIZES_CONFIG["chunk_size_small"]
    ]
    
    return HierarchicalNodeParser.from_defaults(
        chunk_sizes=sizes_list,
        chunk_overlap=CHUNK_SIZES_CONFIG["chunk_overlap"],
        include_metadata=True,
        include_prev_next_rel=True
    )

def process_pdf_with_direct_api(pdf_path: str, strategy: str = "auto") -> List[Document]:
    """
    Verarbeitet ein PDF direkt über die Unstructured API mit HTTP-Requests.
    Die 'auto' Strategie wählt automatisch die beste Methode für jede Seite:
    - Normale Textseiten: Schnelle Textextraktion
    - Seiten mit Tabellen/komplexen Layouts: Hi-res OCR
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        strategy: Strategie für die Extraktion ('auto', 'hi_res', 'fast', etc.)
        
    Returns:
        Liste von LlamaIndex Document-Objekten
    """
    print(f"[DIREKTE API] Verarbeite {pdf_path} mit Strategie '{strategy}'")
    
    if not os.path.exists(pdf_path):
        print(f"❌ Datei nicht gefunden: {pdf_path}")
        return []
    
    # PDF-Datei als Binärdaten laden
    with open(pdf_path, "rb") as file:
        file_content = file.read()
    
    # Dateiname extrahieren
    filename = os.path.basename(pdf_path)
    
    # Multipart-Formular für den POST-Request vorbereiten
    files = {
        "files": (filename, file_content, "application/pdf")
    }
    
    # Parameter für den API-Aufruf
    params = {
        "strategy": strategy,
        "combine_text_under_n_chars": 0,  # Keine Textkombination
        "include_page_breaks": True,      # Seitenumbrüche einfügen
        "max_characters": 500000,        # Maximale Zeichen pro Chunk
        "output_format": "application/json"  # JSON-Ausgabe
    }
    
    print(f"[DIREKTE API] Sende Anfrage an Unstructured API mit Strategie '{strategy}'...")
    start_time = time.time()
    
    try:
        # POST-Anfrage an die Unstructured API senden
        response = requests.post(
            UNSTRUCTURED_API_URL,
            files=files,
            data=params,
            timeout=3600  # 10 Minuten Timeout
        )
        
        # Überprüfen des Status-Codes
        if response.status_code != 200:
            print(f"❌ API-Fehler: Status {response.status_code}")
            print(f"Antwort: {response.text}")
            return []
        
        # JSON-Antwort parsen
        elements = response.json()
        processing_time = time.time() - start_time
        print(f"✓ API-Antwort erhalten in {processing_time:.2f}s mit {len(elements)} Elementen")
        
        # Elemente in Document-Objekte umwandeln
        documents = []
        page_numbers = set()
        element_types = {}
        
        for element in elements:
            # Extrahiere wichtige Metadaten
            metadata = {
                "file_path": pdf_path,
                "file_directory": os.path.dirname(pdf_path),
                "filename": filename
            }
            
            # Extrahiere Seitennummer, falls vorhanden
            if "metadata" in element and "page_number" in element["metadata"]:
                page_num = element["metadata"]["page_number"]
                metadata["page_number"] = page_num
                page_numbers.add(page_num)
            
            # Zähle Element-Typen für Statistik
            element_type = element.get("type", "unbekannt")
            element_types[element_type] = element_types.get(element_type, 0) + 1
            
            # Erstelle LlamaIndex Document aus dem Element
            document = Document(
                text=element.get("text", ""),
                metadata=metadata
            )
            documents.append(document)
        
        # Statistik ausgeben
        print(f"Extrahierte Elemente nach Typ:")
        for elem_type, count in element_types.items():
            print(f"  - {elem_type}: {count}")
        
        print(f"Gefundene Seitennummern: {sorted(list(page_numbers))}")
        
        return documents
        
    except requests.RequestException as e:
        print(f"❌ Netzwerkfehler bei API-Anfrage: {str(e)}")
        return []
    except Exception as e:
        print(f"❌ Fehler bei der Verarbeitung der API-Antwort: {str(e)}")
        print(traceback.format_exc())
        return []

def load_and_process_pdfs():
    """
    Lädt PDFs mit UnstructuredReader und bereitet sie für die Indexierung vor.
    Versucht explizit die 'hi_res' Strategie für bessere Seitentrennung.
    Falls UnstructuredReader keine strategy unterstützt, wird die direkte API verwendet.
    """
    def log_resources(message):
        """Loggt Ressourcennutzung mit einer Nachricht"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent(interval=0.1)
        print(f"[RESSOURCEN] {message}: Speicher: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")

    print(f"\nLade PDFs aus dem Ordner: {PDF_FOLDER}")
    log_resources("Start PDF-Verarbeitung")

    # Prüfe, ob PDF-Ordner existiert und Dateien enthält
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"PDF-Ordner '{PDF_FOLDER}' wurde erstellt.")
        print("Bitte lege PDF-Dateien in diesen Ordner und starte das Skript erneut.")
        return []

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        print("⚠ Keine PDF-Dateien im Ordner gefunden!")
        print(f"Bitte lege PDFs in den Ordner: {os.path.abspath(PDF_FOLDER)}")
        return []

    # Logge PDF-Details vorab
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        try:
            doc = fitz.open(pdf_path)
            print(f"PDF-Info: '{pdf_file}' hat {len(doc)} Seiten, Größe: {os.path.getsize(pdf_path)/1024/1024:.2f}MB")
            doc.close()
        except Exception as e:
            print(f"Konnte PDF-Info nicht ermitteln: {str(e)}")

    # Initialisiere UnstructuredReader für lokale Verwendung
    reader = UnstructuredReader()
    documents = []
    
    # Konfiguration: Verwende lokale Installation außer USE_EXTERNAL_API ist True
    use_direct_api = USE_EXTERNAL_API  # Verwende nur externe API wenn explizit gewünscht
    api_available = False
    
    if use_direct_api:
        # Teste API-Verfügbarkeit nur wenn externe API gewünscht
        print(f"Teste Verbindung zur externen Unstructured-API ({UNSTRUCTURED_API_URL})...")
        try:
            api_url_base = UNSTRUCTURED_API_URL.split("/general/v0")[0]
            health_check_url = f"{api_url_base}/healthcheck"
            response = requests.get(health_check_url, timeout=5)
            if response.status_code == 200:
                print("✓ Externe Unstructured-API ist erreichbar!")
                api_available = True
            else:
                print(f"⚠ Externe API antwortet mit Status {response.status_code}.")
                print("→ Verwende lokale Installation stattdessen.")
                use_direct_api = False
        except Exception as e:
            print(f"❌ Externe API nicht erreichbar: {str(e)}")
            print("→ Verwende lokale Installation stattdessen.")
            use_direct_api = False
    else:
        print("✓ Verwende lokale Unstructured-Installation (empfohlen)")

    # Teste lokale UnstructuredReader Funktionalität
    print("\nTeste lokale UnstructuredReader...")
    local_reader_works = False
    try:
        if pdf_files:
            test_file = os.path.join(PDF_FOLDER, pdf_files[0])
            # Teste mit auto-Strategie über unstructured_kwargs
            test_docs = reader.load_data(
                file=test_file, 
                unstructured_kwargs={
                    "strategy": "auto",
                    "include_page_breaks": True,
                    "combine_text_under_n_chars": 0
                }
            )
            if test_docs:
                print("✓ Lokale UnstructuredReader funktioniert mit 'auto' Strategie!")
                local_reader_works = True
            else:
                print("⚠ Lokale UnstructuredReader lieferte keine Dokumente")
    except Exception as te:
        print(f"⚠ Fehler beim Testen mit 'auto' Strategie: {str(te)}")
        print("→ Verwende Standard-UnstructuredReader ohne Strategie-Parameter.")
        try:
            test_docs = reader.load_data(file=test_file)
            if test_docs:
                local_reader_works = True
                print("✓ Standard UnstructuredReader funktioniert!")
        except Exception as e2:
            print(f"❌ Auch Standard-Reader funktioniert nicht: {str(e2)}")
    except Exception as e:
        print(f"❌ Fehler beim Testen der lokalen Installation: {str(e)}")

    try:
        # Lade jede PDF einzeln
        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            file_load_success = False
            try:
                print(f"\n[PROZESS] Starte Verarbeitung von '{pdf_file}'")
                start_time = time.time()
                log_resources(f"Vor Laden von '{pdf_file}'")

                file_documents = []
                
                # Priorisiere lokale Installation
                if local_reader_works and not use_direct_api:
                    # Verwende lokale UnstructuredReader
                    print(f"Verwende lokale UnstructuredReader mit 'auto' Strategie")
                    try:
                        file_documents = reader.load_data(
                            file=pdf_path,
                            unstructured_kwargs={
                                "strategy": "auto",  # Automatische Strategiewahl
                                "include_page_breaks": True,  # Seitenumbrüche beibehalten
                                "combine_text_under_n_chars": 0,  # Keine automatische Textkombination
                                "max_characters": 100000  # Größere Chunks erlauben
                            }
                        )
                    except Exception:
                        # Fallback ohne strategy Parameter
                        print("→ Fallback: Verwende Reader ohne strategy Parameter")
                        file_documents = reader.load_data(file=pdf_path)
                        
                elif use_direct_api and api_available:
                    # Externe API nur als Alternative
                    print(f"Verwende externe API mit 'auto' Strategie")
                    file_documents = process_pdf_with_direct_api(pdf_path, strategy="auto")
                    
                else:
                    # Letzter Fallback: Lokaler Reader ohne strategy
                    print(f"Fallback: Verwende lokalen Reader ohne Strategie")
                    file_documents = reader.load_data(file=pdf_path)

                # Verarbeite die geladenen Dokumente
                if file_documents:
                    documents.extend(file_documents)
                    load_time = time.time() - start_time
                    print(f"✓ Datei '{pdf_file}' erfolgreich geladen ({load_time:.2f}s)")
                    print(f"  - Elemente extrahiert: {len(file_documents)}")  # <= Wichtig: Sollte jetzt > 1 sein!

                    # Zeige Beispiele der extrahierten Elemente und ihre Seitenzahlen
                    page_numbers_found = set()
                    for i, elem in enumerate(file_documents[:5]):  # Zeige erste 5 Elemente
                        page_num = elem.metadata.get('page_number', 'N/A')
                        print(f"  - Element {i+1}: {len(elem.text)} Zeichen, Typ: {type(elem)}, Seite: {page_num}")
                        if page_num != 'N/A':
                            page_numbers_found.add(page_num)
                    if len(file_documents) > 5:
                        print("    ...")
                    print(f"    -> Erste gefundene Seitennummern in Elementen: {sorted(list(page_numbers_found))}")

                log_resources(f"Nach Laden von '{pdf_file}'")
                file_load_success = True

            except TimeoutError as te:
                print(f"❌ {str(te)}")
                log_resources("Nach Timeout")
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Timeout deaktivieren im Fehlerfall
            except Exception as e:
                print(f"❌ Fehler beim Laden von '{pdf_file}': {type(e).__name__} - {str(e)}")
                log_resources("Nach Fehler")
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Timeout deaktivieren im Fehlerfall

        print(f"\n✓ Insgesamt {len(documents)} Dokument-Element(e) erfolgreich geladen")
        log_resources("Nach Laden aller Dokumente")

        if not documents:
            print("Keine Dokument-Elemente nach dem Laden vorhanden.")
            return []

        # Bereinige Metadaten, um ihre Größe zu reduzieren
        print("Bereinige Metadaten...")
        start_time = time.time()
        cleaned_documents = []
        for doc in documents:
            cleaned_metadata = {}
            if doc.metadata:
                # Übernehme nur explizit gewünschte Felder
                for key in ['file_path', 'file_directory', 'filename', 'page_number']:
                    if key in doc.metadata:
                        cleaned_metadata[key] = doc.metadata[key]

            # Erstelle neues Dokument mit bereinigten Metadaten
            cleaned_doc = Document(text=doc.text, metadata=cleaned_metadata)
            cleaned_documents.append(cleaned_doc)

        metadata_time = time.time() - start_time
        print(f"✓ Metadaten bereinigt in {metadata_time:.2f}s")
        avg_meta_size = sum(len(str(d.metadata)) for d in cleaned_documents) // len(cleaned_documents) if cleaned_documents else 0
        print(f"  Durchschnittliche Metadatengröße: {avg_meta_size} Zeichen")

        documents = cleaned_documents
        log_resources("Nach Metadatenbereinigung")

        # Debug-Information zu den geladenen Dokumenten
        print("\nAnalyse der extrahierten Dokumente (nach Bereinigung):")
        docs_by_file = {}
        for doc in documents:
            file_path = doc.metadata.get('file_path', 'Unbekannt')
            if file_path not in docs_by_file:
                docs_by_file[file_path] = []
            docs_by_file[file_path].append(doc)

        total_docs_analyzed = 0
        for file_path, docs_in_file in docs_by_file.items():
            file_name = Path(file_path).name
            # ACHTUNG: 'page_number' kann nach Unstructured/hi_res auch None sein oder fehlen!
            page_numbers = set(d.metadata.get('page_number') for d in docs_in_file if d.metadata and 'page_number' in d.metadata and d.metadata['page_number'] is not None)
            print(f"Datei: {file_name}")
            print(f"  - Extrahierte Dokument-Elemente: {len(docs_in_file)}")  # Das sollte jetzt hoffentlich > 1 sein
            print(f"  - Erkannte eindeutige Seitennummern: {sorted(list(page_numbers)) if page_numbers else 'Keine oder nur N/A'}")
            print(f"  - Gesamttextmenge: {sum(len(d.text) for d in docs_in_file):,} Zeichen")
            total_docs_analyzed += len(docs_in_file)

        print(f"\nGesamtzahl der Dokument-Elemente zur Indexierung: {total_docs_analyzed}")

        log_resources("Vor Rückgabe der Dokumente")
        return documents

    except Exception as e:
        print(f"❌ Globaler Fehler im PDF Lade/Verarbeitungsprozess: {str(e)}")
        print(traceback.format_exc())
        log_resources("Nach globalem Fehler")
        return []

def main():
    print("\n=== Start der PDF-Indexierung mit Unstructured ===")
    
    # GPU-Status prüfen
    check_gpu_status()
    
    # 1. Initialisiere ChromaDB Client und Collection
    print("\n1. Initialisiere ChromaDB...")
    try:
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
        # Lösche Collection falls sie existiert
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            print("  -> Existierende Collection gelöscht")
        except:
            pass
        chroma_collection = chroma_client.create_collection(name=COLLECTION_NAME)
        print("✓ Neue ChromaDB Collection erstellt")
    except Exception as e:
        print(f"❌ Fehler bei ChromaDB Initialisierung: {str(e)}")
        return

    # 2. Erstelle ChromaVectorStore
    print("\n2. Erstelle VectorStore...")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Konfiguriere LlamaIndex Settings
    print("\n3. Konfiguriere Embedding Model...")
    
    # GPU-Nutzung für HuggingFace Embeddings konfigurieren
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding-Berechnung wird auf {device.upper()} ausgeführt")
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device=device  # "cuda" oder "cpu" je nach Verfügbarkeit
    )
    Settings.node_parser = create_hierarchical_parser()
    print(f"✓ Embedding Model '{EMBED_MODEL_NAME}' konfiguriert auf {device.upper()}")
    print("✓ Hierarchischer Parser konfiguriert")

    # 4. PDFs laden mit Unstructured
    print("\n4. Lade und verarbeite PDFs mit Unstructured...")
    documents = load_and_process_pdfs()

    if not documents:
        print("Keine Dokumente zum Indexieren gefunden. Beende Programm.")
        return

    # 5. Index erstellen (mit expliziter Node-Generierung zur besseren Kontrolle)
    print("\n5. Parse Dokumente in Nodes...")
    try:
        # Speicherverbrauch vor dem Parsen
        gpu_mem_before = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_before = torch.cuda.memory_allocated() / 1024**2  # in MB
            print(f"GPU-Speicher vor Node-Parsing: {gpu_mem_before:.2f} MB")
            
        start_time = time.time()
        nodes = Settings.node_parser.get_nodes_from_documents(documents, show_progress=True)
        parsing_time = time.time() - start_time
        print(f"-> Parser hat {len(nodes)} Nodes in {parsing_time:.2f}s generiert.")

        print("\n--- Erste 3 Nodes nach dem Parsen (Überprüfung der Metadaten) ---")
        for i, node in enumerate(nodes[:3]):  # Zeige die ersten 3 Nodes
            print(f"\nNode {i+1} (ID: {node.id_}):")
            text_preview = node.text[:150].replace('\n', ' ')
            print(f"  Text (Vorschau): {text_preview}...")  # Zeilenumbrüche für Lesbarkeit ersetzen
            print(f"  Metadaten: {node.metadata}")
            if 'file_path' not in node.metadata or not node.metadata['file_path']:
                print(f"  WARNUNG: 'file_path' fehlt oder ist leer in Metadaten für Node {node.id_}!")
            if 'page_number' not in node.metadata:  # page_number kann auch mal None sein, das ist OK
                print(f"  INFO: 'page_number' fehlt in Metadaten für Node {node.id_} (kann bei Unstructured vorkommen).")
        print("--- Ende Node-Vorschau ---")
        
        # Speicherverbrauch nach dem Parsen
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**2  # in MB
            print(f"GPU-Speicher nach Node-Parsing: {gpu_mem_after:.2f} MB")
            print(f"GPU-Speicher-Differenz: {gpu_mem_after - gpu_mem_before:.2f} MB")

        if not nodes:
            print("❌ Parser hat keine Nodes generiert, obwohl Dokument-Elemente vorhanden waren. Indexierung abgebrochen.")
            return

        print("\n6. Erstelle Index aus Nodes...")
        # Speicher vor der Indizierung
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_before_index = torch.cuda.memory_allocated() / 1024**2  # in MB
            print(f"GPU-Speicher vor Indexierung: {gpu_mem_before_index:.2f} MB")
            
        # Verwende den StorageContext und VectorStore wie zuvor initialisiert
        start_time = time.time()
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        indexing_time = time.time() - start_time
        
        # Speicher nach der Indizierung
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_after_index = torch.cuda.memory_allocated() / 1024**2  # in MB
            print(f"GPU-Speicher nach Indexierung: {gpu_mem_after_index:.2f} MB")
            print(f"GPU-Speicher-Differenz für Indexierung: {gpu_mem_after_index - gpu_mem_before_index:.2f} MB")
            print(f"Indexierung abgeschlossen in {indexing_time:.2f}s mit GPU-Beschleunigung")

        print("\n=== Indexierung erfolgreich abgeschlossen ===")
        print(f"- Verarbeitete Dokument-Elemente (aus Reader): {len(documents)}")
        print(f"- Verarbeitete Nodes durch Parser: {len(nodes)}")

        # Prüfe den Docstore *direkt* nach der Indexierung
        final_nodes_in_docstore = index.docstore.docs
        print(f"- Nodes im Index Docstore: {len(final_nodes_in_docstore)}")

        # Optional: Prüfe ChromaDB direkt
        try:
            count_in_chroma = chroma_collection.count()
            print(f"- Elemente direkt in ChromaDB Collection '{COLLECTION_NAME}': {count_in_chroma}")
            if count_in_chroma != len(nodes):
                print(f"WARNUNG: Anzahl Nodes ({len(nodes)}) stimmt nicht mit ChromaDB Count ({count_in_chroma}) überein!")
        except Exception as chroma_err:
            print(f"Fehler beim Abfragen von ChromaDB Count: {chroma_err}")

        # Debug-Information über die ersten Nodes im Docstore
        print("\n--- Node-Struktur im Docstore (erste 3 Nodes) ---")
        for i, (node_id, node) in enumerate(list(final_nodes_in_docstore.items())[:3]):
            print(f"\nNode {i+1}:")
            print(f"- ID: {node_id}")
            print(f"- Textlänge: {len(node.text)} Zeichen")
            print(f"- Text-Vorschau: {node.text[:100]}...")
            print(f"- Metadaten: {node.metadata}")
        print("--- Ende Node-Struktur ---")
        
        # Abschließender GPU-Status
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print("\nAbschließender GPU-Status:")
            print(f"- GPU-Speichernutzung: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"- GPU-Speicher reserviert: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    except Exception as e:
        print(f"\n❌ Fehler beim Parsen oder Erstellen des Index:")
        print(f"Fehler: {type(e).__name__} - {str(e)}")
        print("\nStack Trace:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
 