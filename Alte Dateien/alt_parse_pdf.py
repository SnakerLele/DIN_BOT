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
import argparse # Für Kommandozeilenargumente
from typing import List, Dict, Any, Optional, Tuple
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    # Füge encoding-Parameter hinzu, um Probleme mit Unicode-Zeichen zu vermeiden
    force=True  # Überschreibe bestehende Konfiguration
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

# 2. Konfiguration
PDF_FOLDER = "./Test_PDF"  # Ordner für PDFs
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

# Unstructured API Konfiguration
# Prüfe, ob wir in einem Docker-Container sind oder lokal laufen
def is_running_in_docker():
    """Prüft, ob der Code in einem Docker-Container läuft"""
    try:
        with open('/proc/self/cgroup', 'r') as f:
            return any('docker' in line for line in f)
    except:
        return False

# Wähle die richtige API-URL basierend auf der Umgebung
if is_running_in_docker() or os.environ.get("USE_DOCKER_NETWORK", "").lower() in ("1", "true", "yes"):
    # In Docker: Verwende Container-Namen im Docker-Netzwerk
    DEFAULT_API_URL = "http://unstructured-api:8000/general/v0/general"
else:
    # Lokale Ausführung: Verwende localhost
    DEFAULT_API_URL = "http://localhost:8000/general/v0/general"

# Ermögliche Überschreibung durch Umgebungsvariable
UNSTRUCTURED_API_URL = os.environ.get("UNSTRUCTURED_API_URL", DEFAULT_API_URL)
print(f"Konfigurierte Unstructured API URL: {UNSTRUCTURED_API_URL}")

# Wenn ein Container-Name verwendet wird, aber er nicht aufgelöst werden kann,
# versuche automatisch auf localhost umzuschalten
def get_effective_api_url():
    """
    Gibt die effektive API-URL zurück und versucht, auf localhost umzuschalten,
    wenn der Container-Name nicht aufgelöst werden kann.
    """
    global UNSTRUCTURED_API_URL
    
    # Wenn die URL bereits localhost verwendet, keine Änderung nötig
    if "localhost" in UNSTRUCTURED_API_URL or "127.0.0.1" in UNSTRUCTURED_API_URL:
        return UNSTRUCTURED_API_URL
    
    # Versuche, die Domain aus der URL zu extrahieren
    import re
    domain_match = re.search(r'http://([^:/]+)', UNSTRUCTURED_API_URL)
    if domain_match:
        domain = domain_match.group(1)
        
        # Prüfe, ob die Domain aufgelöst werden kann
        try:
            import socket
            socket.gethostbyname(domain)
            # Domain kann aufgelöst werden, behalte die URL bei
            return UNSTRUCTURED_API_URL
        except socket.gaierror:
            # Domain kann nicht aufgelöst werden, wechsle zu localhost
            localhost_url = UNSTRUCTURED_API_URL.replace(f"http://{domain}", "http://localhost")
            print(f"[INFO] Domain '{domain}' kann nicht aufgelöst werden. Verwende stattdessen: {localhost_url}")
            UNSTRUCTURED_API_URL = localhost_url
            return UNSTRUCTURED_API_URL
    
    # Fallback: Original-URL zurückgeben
    return UNSTRUCTURED_API_URL

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

def process_pdf_with_direct_api(pdf_path: str, strategy: str = "hi_res") -> List[Document]:
    """
    Verarbeitet ein PDF direkt über die Unstructured API mit HTTP-Requests.
    Dies umgeht die Einschränkungen des UnstructuredReader und erlaubt die direkte Nutzung
    der 'hi_res' Strategie.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        strategy: Strategie für die Extraktion ('hi_res', 'fast', etc.)
        
    Returns:
        Liste von LlamaIndex Document-Objekten
    """
    print(f"[DIREKTE API] Verarbeite {pdf_path} mit Strategie '{strategy}'")
    
    if not os.path.exists(pdf_path):
        print(f"[FEHLER] Datei nicht gefunden: {pdf_path}")
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
    
    # Hole die effektive API-URL mit Namensauflösungs-Prüfung
    effective_api_url = get_effective_api_url()
    print(f"[DIREKTE API] Sende Anfrage an Unstructured API ({effective_api_url}) mit Strategie '{strategy}'...")
    start_time = time.time()
    
    try:
        # POST-Anfrage an die Unstructured API senden
        response = requests.post(
            effective_api_url,
            files=files,
            data=params,
            timeout=3600  # 60 Minuten Timeout
        )
        
        # Überprüfen des Status-Codes
        if response.status_code != 200:
            print(f"[FEHLER] API-Fehler: Status {response.status_code}")
            print(f"Antwort: {response.text}")
            return []
        
        # JSON-Antwort parsen
        elements = response.json()
        processing_time = time.time() - start_time
        print(f"[OK] API-Antwort erhalten in {processing_time:.2f}s mit {len(elements)} Elementen")
        
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
        print(f"[FEHLER] Netzwerkfehler bei API-Anfrage: {str(e)}")
        return []
    except Exception as e:
        print(f"[FEHLER] Fehler bei der Verarbeitung der API-Antwort: {str(e)}")
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
        print("[WARNUNG] Keine PDF-Dateien im Ordner gefunden!")
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

    # Initialisiere UnstructuredReader für optionale Verwendung
    reader = UnstructuredReader()
    documents = []
    use_direct_api = False  # Flag für API-Direktnutzung
    api_available = True    # Flag, ob API verfügbar ist

    # Teste, ob UnstructuredReader die strategy unterstützt
    print("\nTeste UnstructuredReader auf Unterstützung der 'hi_res' Strategie...")
    try:
        # Versuche mit einem einfachen Testaufruf festzustellen, ob die Strategie unterstützt wird
        if pdf_files:
            test_file = os.path.join(PDF_FOLDER, pdf_files[0])
            reader.load_data(file=test_file, strategy="hi_res")
            print("[OK] UnstructuredReader unterstützt die 'hi_res' Strategie!")
            use_direct_api = False
    except TypeError as te:
        if "unexpected keyword argument 'strategy'" in str(te):
            print("[INFO] UnstructuredReader unterstützt die 'hi_res' Strategie NICHT.")
            print("-> Versuche stattdessen direkte API-Aufrufe für bessere Ergebnisse.")
            use_direct_api = True
            
            # Verwende die neue Funktion, um die effektive API-URL zu ermitteln
            effective_api_url = get_effective_api_url()
            
            # Teste API-Verfügbarkeit mit der effektiven URL
            print(f"Teste Verbindung zum Unstructured-API-Server ({effective_api_url})...")
            try:
                api_url_base = effective_api_url.split("/general/v0")[0]
                health_check_url = f"{api_url_base}/healthcheck"
                response = requests.get(health_check_url, timeout=5)
                if response.status_code == 200:
                    print("[OK] Unstructured-API-Server ist erreichbar und funktioniert!")
                    api_available = True
                else:
                    print(f"[WARNUNG] Unstructured-API-Server antwortet mit Status {response.status_code}.")
                    api_available = False
            except Exception as e:
                print(f"[FEHLER] Unstructured-API-Server ist nicht erreichbar: {str(e)}")
                print("-> Falle zurück auf Standard-UnstructuredReader ohne hi_res-Strategie.")
                use_direct_api = False
                api_available = False
        else:
            # Anderer TypeError, kein Problem mit strategy
            print(f"[WARNUNG] Test mit UnstructuredReader fehlgeschlagen: {str(te)}")
            print("-> Verwende regulären Ansatz, aber möglicherweise ohne hi_res-Strategie.")
    except Exception as e:
        # Anderer Fehler, setze einfach fort und verwende Standardansatz
        print(f"[WARNUNG] Fehler beim Testen von UnstructuredReader: {str(e)}")
        print("-> Verwende regulären Ansatz, aber möglicherweise ohne hi_res-Strategie.")

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
                
                if use_direct_api and api_available:
                    # Direkte API-Nutzung mit hi_res Strategie
                    print(f"Verwende direkte API mit 'hi_res' Strategie")
                    file_documents = process_pdf_with_direct_api(pdf_path, strategy="hi_res")
                    
                    # Wenn die API keine Dokumente zurückgibt, versuche es mit UnstructuredReader als Fallback
                    if not file_documents:
                        print("[WARNUNG] Direkte API lieferte keine Dokumente. Versuche Fallback mit UnstructuredReader...")
                        # Setze Timeout-Handler (nur unter Linux/Mac)
                        if hasattr(signal, 'SIGALRM'):
                            def timeout_handler(signum, frame):
                                raise TimeoutError(f"Zeitüberschreitung (>10min) beim Laden von '{pdf_file}'")
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(600)  # 600 Sekunden = 10 Minuten
                        else:
                            print("WARNUNG: Timeout-Signal (SIGALRM) nicht verfügbar auf diesem System.")
                            
                        file_documents = reader.load_data(file=pdf_path)
                        
                        # Deaktiviere Timeout
                        if hasattr(signal, 'SIGALRM'):
                            signal.alarm(0)
                else:
                    # Setze Timeout-Handler (nur unter Linux/Mac)
                    if hasattr(signal, 'SIGALRM'):
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Zeitüberschreitung (>10min) beim Laden von '{pdf_file}'")
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(600)  # 600 Sekunden = 10 Minuten
                    else:
                        print("WARNUNG: Timeout-Signal (SIGALRM) nicht verfügbar auf diesem System.")

                    # Versuche hi_res Strategie mit UnstructuredReader
                    try:
                        print("Verwende UnstructuredReader (ohne hi_res)")
                        file_documents = reader.load_data(
                            file=pdf_path,
                            strategy="hi_res"  # Fordere die High-Resolution-Strategie an
                        )
                    except TypeError as te:
                        if "unexpected keyword argument 'strategy'" in str(te):
                            print("WARNUNG: 'strategy' wird von reader.load_data nicht direkt unterstützt.")
                            print("-> Versuche es ohne explizite Strategie (Standardverhalten des Readers/API).")
                            file_documents = reader.load_data(file=pdf_path)
                        else:
                            raise  # Anderen TypeError weiterleiten

                    # Deaktiviere Timeout
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)

                # Verarbeite die geladenen Dokumente
                if file_documents:
                    documents.extend(file_documents)
                    load_time = time.time() - start_time
                    print(f"[OK] Datei '{pdf_file}' erfolgreich geladen ({load_time:.2f}s)")
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
                print(f"[FEHLER] {str(te)}")
                log_resources("Nach Timeout")
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Timeout deaktivieren im Fehlerfall
            except Exception as e:
                print(f"[FEHLER] Fehler beim Laden von '{pdf_file}': {type(e).__name__} - {str(e)}")
                log_resources("Nach Fehler")
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Timeout deaktivieren im Fehlerfall

        print(f"\n[OK] Insgesamt {len(documents)} Dokument-Element(e) erfolgreich geladen")
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
        print(f"[OK] Metadaten bereinigt in {metadata_time:.2f}s")
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
        print(f"[FEHLER] Globaler Fehler im PDF Lade/Verarbeitungsprozess: {str(e)}")
        print(traceback.format_exc())
        log_resources("Nach globalem Fehler")
        return []

def initialize_chroma(persist_dir: str, collection_name: str, recreate: bool = False) -> Tuple[ChromaVectorStore, Any]:
    """Initialisiert ChromaDB und gibt einen VectorStore zurück.
    
    Args:
        persist_dir: Pfad zum ChromaDB Speicherort.
        collection_name: Name der Collection.
        recreate: Wenn True, wird die Collection gelöscht und neu erstellt.
    """
    logging.info(f"Initialisiere ChromaDB (persist_dir='{persist_dir}', collection='{collection_name}', recreate={recreate})")
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    if recreate:
        try:
            chroma_client.delete_collection(name=collection_name)
            logging.info(f"  -> Existierende Collection '{collection_name}' gelöscht.")
        except Exception as e: # Genauer: chromadb.errors.CollectionNotFoundError oder ähnliches
            logging.info(f"  -> Collection '{collection_name}' nicht gefunden oder Fehler beim Löschen: {e}")
            pass # Ignoriere Fehler, wenn die Collection nicht existiert
    
    # Versuche, die Collection zu bekommen oder zu erstellen
    try:
        chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
        logging.info(f"[OK] ChromaDB Collection '{collection_name}' geladen/erstellt.")
    except Exception as e:
        logging.error(f"[FEHLER] Fehler bei ChromaDB Initialisierung: {str(e)}")
        raise # Fehler weiterleiten, da kritisch

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store, chroma_collection # Gebe auch die Collection zurück für direkte Prüfungen

def configure_llama_index_settings(embed_model_name: str):
    """Konfiguriert die globalen LlamaIndex Settings."""
    logging.info("Konfiguriere LlamaIndex Settings...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Embedding-Berechnung wird auf {device.upper()} ausgeführt")
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        device=device,
        embed_batch_size=128
    )
    Settings.node_parser = create_hierarchical_parser() # Nutzt CHUNK_SIZES_CONFIG aus globalem Scope
    logging.info(f"[OK] Embedding Model '{embed_model_name}' konfiguriert auf {device.upper()}")
    logging.info("[OK] Hierarchischer Parser konfiguriert")

def main():
    print("\n=== Start der PDF-Indexierung mit Unstructured ===")
    
    # GPU-Status prüfen
    check_gpu_status()
    
    # Verwende die neue Hilfsfunktion für die Initialisierung von ChromaDB
    print("\n1. Initialisiere ChromaDB...")
    try:
        vector_store, chroma_collection = initialize_chroma(PERSIST_DIR, COLLECTION_NAME, recreate=True)
        print("[OK] Neue ChromaDB Collection erstellt")
    except Exception as e:
        print(f"[FEHLER] Fehler bei ChromaDB Initialisierung: {str(e)}")
        return

    # 2. Erstelle StorageContext
    print("\n2. Erstelle VectorStore...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Konfiguriere LlamaIndex Settings mit der neuen Hilfsfunktion
    print("\n3. Konfiguriere Embedding Model...")
    configure_llama_index_settings(EMBED_MODEL_NAME)

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
                print(f"  [WARNUNG] 'file_path' fehlt oder ist leer in Metadaten für Node {node.id_}!")
            if 'page_number' not in node.metadata:  # page_number kann auch mal None sein, das ist OK
                print(f"  [INFO] 'page_number' fehlt in Metadaten für Node {node.id_} (kann bei Unstructured vorkommen).")
        print("--- Ende Node-Vorschau ---")
        
        # Speicherverbrauch nach dem Parsen
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**2  # in MB
            print(f"GPU-Speicher nach Node-Parsing: {gpu_mem_after:.2f} MB")
            print(f"GPU-Speicher-Differenz: {gpu_mem_after - gpu_mem_before:.2f} MB")

        if not nodes:
            print("[FEHLER] Parser hat keine Nodes generiert, obwohl Dokument-Elemente vorhanden waren. Indexierung abgebrochen.")
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
                print(f"[WARNUNG] Anzahl Nodes ({len(nodes)}) stimmt nicht mit ChromaDB Count ({count_in_chroma}) überein!")
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
        print(f"\n[FEHLER] Fehler beim Parsen oder Erstellen des Index:")
        print(f"Fehler: {type(e).__name__} - {str(e)}")
        print("\nStack Trace:")
        print(traceback.format_exc())

def action_delete_all_data(persist_dir: str, collection_name: str):
    """Löscht alle Daten aus der spezifizierten ChromaDB Collection."""
    print(f"\n=== AKTION: Lösche alle Daten aus Collection '{collection_name}' ===")
    try:
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_client.delete_collection(name=collection_name)
        print(f"[OK] Collection '{collection_name}' erfolgreich gelöscht.")
    except Exception as e: # Genauer auf ChromaDB Fehler prüfen
        print(f"[FEHLER] Fehler beim Löschen der Collection '{collection_name}': {str(e)}")
        print(f"  -> Möglicherweise existierte die Collection nicht.")

def action_add_data(pdf_folder_path: str, persist_dir: str, collection_name: str):
    """Fügt PDFs aus dem Ordner dem bestehenden Index hinzu oder erstellt einen neuen Index."""
    print(f"\n=== AKTION: Füge Daten aus '{pdf_folder_path}' zu Collection '{collection_name}' hinzu ===")
    
    # Globale Settings konfigurieren (Embedding-Modell, Parser)
    configure_llama_index_settings(EMBED_MODEL_NAME) 

    # ChromaDB initialisieren (nicht neu erstellen, falls vorhanden)
    vector_store, chroma_collection = initialize_chroma(persist_dir, collection_name, recreate=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Lade und verarbeite PDFs...")
    # PDF_FOLDER wird hier aus dem globalen Scope genommen oder besser als pdf_folder_path übergeben
    # Um es konsistent zu machen, überschreiben wir PDF_FOLDER temporär oder geben es direkt weiter
    global PDF_FOLDER # Zugriff auf globale Variable
    original_pdf_folder = PDF_FOLDER
    PDF_FOLDER = pdf_folder_path # Setze den PDF_FOLDER für load_and_process_pdfs
    
    documents = load_and_process_pdfs() # Nutzt den globalen PDF_FOLDER
    
    PDF_FOLDER = original_pdf_folder # Zurücksetzen, falls an anderer Stelle noch gebraucht

    if not documents:
        print("Keine neuen Dokumente zum Indexieren gefunden.")
        return

    print("Parse Dokumente in Nodes...")
    
    # Parse neue Dokumente zu Nodes
    new_nodes = Settings.node_parser.get_nodes_from_documents(documents, show_progress=True)
    if not new_nodes:
        print("Keine neuen Nodes aus den Dokumenten generiert.")
        return

    print(f"Füge {len(new_nodes)} neue Nodes zum VectorStore hinzu...")
    # Der VectorStoreIndex kann direkt mit Nodes initialisiert werden,
    # und wenn der storage_context auf einen bestehenden VectorStore zeigt, werden sie hinzugefügt.
    
    # Sicherster Weg, um sicherzustellen, dass Embeddings korrekt generiert werden:
    # Erstelle einen Index aus den neuen Nodes und lasse ihn in den bestehenden StorageContext schreiben.
    index = VectorStoreIndex(
        new_nodes, # Nur die neu geparsten Nodes
        storage_context=storage_context, # Zeigt auf bestehenden/neuen VectorStore
        show_progress=True
    )

    print(f"[OK] {len(new_nodes)} Nodes erfolgreich zum Index hinzugefügt/Index aktualisiert.")
    print(f"  -> Elemente direkt in ChromaDB Collection '{collection_name}': {chroma_collection.count()}")


def action_rebuild_index(pdf_folder_path: str, persist_dir: str, collection_name: str):
    """Löscht die bestehende Collection und indiziert alle PDFs aus dem Ordner neu."""
    print(f"\n=== AKTION: Baue Index komplett neu auf für '{pdf_folder_path}' in Collection '{collection_name}' ===")
    
    # Globale Settings konfigurieren
    configure_llama_index_settings(EMBED_MODEL_NAME)

    # ChromaDB initialisieren (mit recrate=True)
    vector_store, chroma_collection = initialize_chroma(persist_dir, collection_name, recreate=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Lade und verarbeite PDFs für Neuaufbau...")
    global PDF_FOLDER
    original_pdf_folder = PDF_FOLDER
    PDF_FOLDER = pdf_folder_path
    
    documents = load_and_process_pdfs()
    
    PDF_FOLDER = original_pdf_folder

    if not documents:
        print("Keine Dokumente zum Indexieren gefunden für Neuaufbau.")
        return

    print("Parse Dokumente in Nodes für Neuaufbau...")
    nodes = Settings.node_parser.get_nodes_from_documents(documents, show_progress=True)
    if not nodes:
        print("Keine Nodes aus den Dokumenten generiert für Neuaufbau.")
        return

    print(f"Erstelle Index aus {len(nodes)} Nodes...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    print(f"[OK] Index erfolgreich neu aufgebaut mit {len(nodes)} Nodes.")
    print(f"  -> Elemente direkt in ChromaDB Collection '{collection_name}': {chroma_collection.count()}")

def main_cli():
    """Hauptfunktion für die Kommandozeilenschnittstelle."""
    parser = argparse.ArgumentParser(description="PDF Parser und Indexer mit Unstructured für LlamaIndex.")
    parser.add_argument(
        "action", 
        choices=["add", "delete_all", "rebuild"], 
        help="Aktion, die ausgeführt werden soll: 'add' (Daten hinzufügen), 'delete_all' (alle Daten löschen), 'rebuild' (Index neu aufbauen)."
    )
    parser.add_argument(
        "--pdf-folder", 
        default=PDF_FOLDER,  # Nutzt den globalen Default
        help=f"Ordner mit den PDF-Dateien (Standard: {PDF_FOLDER})."
    )
    parser.add_argument(
        "--persist-dir", 
        default=PERSIST_DIR, # Nutzt den globalen Default
        help=f"Verzeichnis für die ChromaDB-Speicherung (Standard: {PERSIST_DIR})."
    )
    parser.add_argument(
        "--collection-name", 
        default=COLLECTION_NAME, # Nutzt den globalen Default
        help=f"Name der ChromaDB Collection (Standard: {COLLECTION_NAME})."
    )
    
    args = parser.parse_args()

    # GPU-Status einmal am Anfang prüfen
    check_gpu_status()

    if args.action == "add":
        action_add_data(args.pdf_folder, args.persist_dir, args.collection_name)
    elif args.action == "delete_all":
        action_delete_all_data(args.persist_dir, args.collection_name)
    elif args.action == "rebuild":
        # Dein ursprünglicher main() Code entspricht im Wesentlichen 'rebuild'
        action_rebuild_index(args.pdf_folder, args.persist_dir, args.collection_name)

if __name__ == "__main__":
    # Alte main-Funktion bleibt als legacy_main verfügbar
    # main() # Das war die ursprüngliche main
    main_cli() # Die neue CLI-basierte main aufrufen 