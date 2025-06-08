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
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter, HierarchicalNodeParser
# from llama_index.core.postprocessor import MetadataReplacementPostProcessor  # Wird in rag_api.py verwendet
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

def check_embedding_requirements():
    """Prüft, ob alle erforderlichen Bibliotheken für das Embedding-Modell installiert sind."""
    print("\n===== EMBEDDING REQUIREMENTS =====")
    
    requirements_ok = True
    
    # Prüfe sentence-transformers
    try:
        import sentence_transformers
        version = sentence_transformers.__version__
        print(f"✓ sentence-transformers Version: {version}")
        
        # Prüfe ob Version >= 2.0.0 (ausreichend für paraphrase-multilingual-mpnet-base-v2)
        try:
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse("2.0.0"):
                print(f"⚠ sentence-transformers Version {version} ist zu alt!")
                print("  Mindestversion: 2.0.0")
                print("  Upgrade mit: pip install --upgrade sentence-transformers")
                requirements_ok = False
            else:
                print("✓ sentence-transformers Version ist kompatibel")
        except ImportError:
            print("⚠ packaging nicht verfügbar, überspringe Versions-Check")
            
    except ImportError:
        print("❌ sentence-transformers nicht installiert")
        print("  Installiere mit: pip install sentence-transformers")
        requirements_ok = False
    
    # Prüfe transformers (benötigt für HuggingFace Integration)
    try:
        import transformers
        version = transformers.__version__
        print(f"✓ transformers Version: {version}")
            
    except ImportError:
        print("❌ transformers nicht installiert")
        print("  Installiere mit: pip install transformers")
        requirements_ok = False
    
    print("=" * 35)
    
    if not requirements_ok:
        print("\n❌ Nicht alle erforderlichen Bibliotheken sind installiert!")
        print("Bitte installiere die fehlenden Pakete vor dem Fortfahren.")
        return False
    else:
        print("\n✓ Alle erforderlichen Bibliotheken sind verfügbar!")
        return True

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
# Verwende bewährtes multilinguales Embedding-Modell
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Hierarchical Parser Konfiguration - Erweitert für Sentence-basierte Chunks
CHUNK_SIZES_CONFIG = {
    "window_size": 8,  # VERGRÖSSERT: Mehr Kontext-Sätze für bessere semantische Verbindung
    "chunk_size": 1024,  # VERGRÖSSERT: Größere Chunks für mehr Kontext
    "chunk_overlap": 200,  # VERGRÖSSERT: Mehr Überlappung für bessere Verbindung
    "chunk_size_small": 512,  # VERGRÖSSERT: Auch kleine Chunks größer
    "chunk_size_large": 2048,  # VERGRÖSSERT: Große Chunks für umfassenden Kontext
}

def create_semantic_section_parser():
    """
    Erstellt einen semantischen Parser, der PDF-Abschnitte als ganze Einheiten behandelt.
    Viel bessere Performance als sentence-basierte Chunks für Spielregeln.
    """
    return SentenceSplitter(
        chunk_size=CHUNK_SIZES_CONFIG["chunk_size"],
        chunk_overlap=CHUNK_SIZES_CONFIG["chunk_overlap"],
        include_metadata=True,
        include_prev_next_rel=True
    )

def create_hierarchical_backup_parser():
    """
    Erstellt einen hierarchischen Backup-Parser für zusätzliche Struktur.
    Wird verwendet um längere Chunks für bestimmte Anwendungsfälle zu erstellen.
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

def create_enhanced_documents_from_pdf(pdf_path: str) -> List[Document]:
    """
    Erstellt semantisch sinnvolle Dokument-Chunks durch intelligente Gruppierung.
    Kombiniert zusammengehörige PDF-Elemente zu kohärenten Abschnitten.
    Integriert hierarchische Header-Analyse für reichere Metadaten.
    """
    print(f"[SEMANTIC] Verarbeite {pdf_path} mit semantischer Abschnitts-Gruppierung")
    
    # 1. Lade PDF-Elemente
    try:
        from unstructured.partition.pdf import partition_pdf
        
        elements = partition_pdf(
            filename=pdf_path,
            strategy="auto",
            include_page_breaks=True,
            combine_text_under_n_chars=0
        )
        
        print(f"[SEMANTIC] {len(elements)} PDF-Elemente geladen")
        
    except Exception as e:
        print(f"❌ Fehler beim Laden der PDF-Elemente: {str(e)}")
        return []
    
    # 2. Analysiere hierarchische Headers für alle Elemente
    print(f"[SEMANTIC] Starte hierarchische Header-Analyse für bessere Metadaten...")
    header_mapping = analyze_hierarchical_headers(elements)
    
    # 3. Gruppiere Elemente zu semantischen Abschnitten
    semantic_chunks = []
    current_chunk = ""
    current_section = None
    current_page = None
    current_element_indices = []  # Verfolge welche Elemente im aktuellen Chunk sind
    
    for i, element in enumerate(elements):
        element_text = str(element).strip()
        element_type = type(element).__name__
        
        # Extrahiere Seitennummer
        page_number = None
        if hasattr(element, 'metadata') and element.metadata and hasattr(element.metadata, 'page_number'):
            page_number = element.metadata.page_number
        
        # Erkenne Abschnittswechsel (Title-Elemente mit ausreichender Länge)
        is_new_section = (element_type == "Title" and len(element_text) > 5)
        
        if is_new_section:
            # Speichere den vorherigen Chunk wenn er Inhalt hat
            if current_chunk.strip() and len(current_chunk.strip()) > 50:
                semantic_chunks.append({
                    'text': current_chunk.strip(),
                    'section': current_section or "Unnamed Section",
                    'page_number': current_page,
                    'length': len(current_chunk.strip()),
                    'element_indices': current_element_indices.copy()  # Kopiere die Element-Indices
                })
            
            # Starte neuen Chunk
            current_section = element_text
            current_page = page_number
            current_chunk = element_text + "\n\n"
            current_element_indices = [i]  # Neuer Chunk startet mit diesem Element
        else:
            # Füge zum aktuellen Chunk hinzu
            if element_text and len(element_text.strip()) > 2:
                current_chunk += element_text + " "
                current_element_indices.append(i)  # Verfolge dieses Element
                # Update page number if available
                if page_number and not current_page:
                    current_page = page_number
    
    # Letzten Chunk speichern
    if current_chunk.strip() and len(current_chunk.strip()) > 50:
        semantic_chunks.append({
            'text': current_chunk.strip(),
            'section': current_section or "Final Section",
            'page_number': current_page,
            'length': len(current_chunk.strip()),
            'element_indices': current_element_indices.copy()
        })
    
    # 4. Konvertiere zu LlamaIndex Documents mit erweiterten Header-Metadaten
    documents = []
    filename = os.path.basename(pdf_path)
    
    for i, chunk in enumerate(semantic_chunks):
        # Basis-Metadaten
        metadata = {
            "file_path": pdf_path,
            "file_directory": os.path.dirname(pdf_path),
            "filename": filename,
            "section_title": chunk['section'],
            "chunk_index": i,
            "semantic_chunk": True  # Markierung für semantische Chunks
        }
        
        # Füge Seitennummer hinzu wenn verfügbar
        if chunk['page_number'] is not None:
            metadata["page_number"] = chunk['page_number']
        
        # Integriere hierarchische Header-Metadaten aus dem Chunk
        # Sammle alle Header-Informationen der Elemente in diesem Chunk
        chunk_header_info = {
            'section_h1': None,
            'section_h2': None, 
            'section_h3': None,
            'current_section': None,
            'header_elements': []
        }
        
        for element_idx in chunk['element_indices']:
            if element_idx in header_mapping:
                element_headers = header_mapping[element_idx]
                
                # Sammle Header-Informationen (verwende die letzten gefundenen)
                if element_headers.get('section_h1'):
                    chunk_header_info['section_h1'] = element_headers['section_h1']
                if element_headers.get('section_h2'):
                    chunk_header_info['section_h2'] = element_headers['section_h2']
                if element_headers.get('section_h3'):
                    chunk_header_info['section_h3'] = element_headers['section_h3']
                if element_headers.get('current_section'):
                    chunk_header_info['current_section'] = element_headers['current_section']
                
                # Sammle Header-Elemente
                if element_headers.get('is_header'):
                    chunk_header_info['header_elements'].append({
                        'text': str(elements[element_idx]).strip(),
                        'type': element_headers.get('element_type'),
                        'index': element_idx
                    })
        
        # Füge Header-Metadaten zu den Dokument-Metadaten hinzu
        if chunk_header_info['section_h1']:
            metadata['section_h1'] = chunk_header_info['section_h1']
        if chunk_header_info['section_h2']:
            metadata['section_h2'] = chunk_header_info['section_h2']
        if chunk_header_info['section_h3']:
            metadata['section_h3'] = chunk_header_info['section_h3']
        if chunk_header_info['current_section']:
            metadata['current_section'] = chunk_header_info['current_section']
        
        # Zusätzliche Header-Statistiken
        metadata['header_count'] = len(chunk_header_info['header_elements'])
        if chunk_header_info['header_elements']:
            metadata['contains_headers'] = True
            metadata['first_header'] = chunk_header_info['header_elements'][0]['text']
        else:
            metadata['contains_headers'] = False
        
        document = Document(
            text=chunk['text'],
            metadata=metadata
        )
        documents.append(document)
    
    print(f"[SEMANTIC] {len(semantic_chunks)} semantische Abschnitte erstellt:")
    for i, chunk in enumerate(semantic_chunks):
        print(f"  - Abschnitt {i+1}: '{chunk['section']}' ({chunk['length']} Zeichen)")
    
    # Zeige Header-Analyse Zusammenfassung
    header_enriched_chunks = [d for d in documents if d.metadata.get('contains_headers')]
    print(f"[HEADER] Header-Integration erfolgreich:")
    print(f"  - Chunks mit Header-Metadaten: {len(header_enriched_chunks)}/{len(documents)}")
    print(f"  - H1-Überschriften gefunden: {len(set(d.metadata.get('section_h1') for d in documents if d.metadata.get('section_h1')))}")
    print(f"  - H2-Überschriften gefunden: {len(set(d.metadata.get('section_h2') for d in documents if d.metadata.get('section_h2')))}")
    print(f"  - H3-Überschriften gefunden: {len(set(d.metadata.get('section_h3') for d in documents if d.metadata.get('section_h3')))}")
    
    return documents

def create_hybrid_parser_system():
    """
    Erstellt ein hybrides Parser-System mit semantischen Abschnitten als Hauptlogik.
    """
    # Haupt-Parser: Semantische Abschnitte für beste Retrieval-Qualität  
    semantic_parser = create_semantic_section_parser()
    
    # Backup-Parser: Hierarchisch für sehr lange Abschnitte falls benötigt
    hierarchical_parser = create_hierarchical_backup_parser()
    
    # Fallback: Standard-Splitter
    sentence_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZES_CONFIG["chunk_size"],
        chunk_overlap=CHUNK_SIZES_CONFIG["chunk_overlap"]
    )
    
    return {
        'main': semantic_parser,
        'hierarchical': hierarchical_parser, 
        'fallback': sentence_splitter
    }

# MetadataReplacementPostProcessor wird in rag_api.py verwendet
# def create_metadata_replacement_processor():
#     """
#     Erstellt einen MetadataReplacementPostProcessor für optimale Kontext-Nutzung.
#     Ersetzt einzelne Sätze durch ihr Kontext-Fenster während der Abfrage.
#     """
#     return MetadataReplacementPostProcessor(
#         target_metadata_key="window"
#     )

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

def analyze_hierarchical_headers(elements: list) -> dict:
    """
    Analysiert Unstructured-Elemente und erstellt eine Header-Zuordnung.
    Da ElementMetadata read-only ist, geben wir eine Mapping-Tabelle zurück.
    
    Args:
        elements: Liste von Unstructured-Elementen
        
    Returns:
        dict: Mapping von Element-Index zu Header-Metadaten
    """
    print("[HEADER-ANALYSE] Starte hierarchische Header-Analyse...")
    
    current_headers = {
        'h1': None,
        'h2': None, 
        'h3': None,
        'current_section': None
    }
    
    header_stats = {
        'titles_found': 0,
        'headers_found': 0,
        'elements_with_headers': 0
    }
    
    # Mapping von Element-Index zu Header-Metadaten
    header_mapping = {}
    
    # Identifiziere Header-Typen basierend auf Text-Eigenschaften
    def classify_header_level(element_text: str, element_type: str) -> str:
        """Klassifiziert Header-Ebene basierend auf Text und Typ"""
        text_lower = element_text.lower().strip()
        text_length = len(element_text.strip())
        
        # Sehr kurze, allgemeine Titel = H1 (Hauptüberschriften)
        if text_length < 50 and any(keyword in text_lower for keyword in 
                                   ['spielregeln', 'anleitung', 'inhalt', 'ziel', 'vorbereitung']):
            return 'h1'
        
        # Mittlere Überschriften = H2 (Abschnitte)
        elif text_length < 100 and any(keyword in text_lower for keyword in 
                                      ['spielverlauf', 'aktionskarten', 'sonderkarten', 'punkte']):
            return 'h2'
        
        # Detailüberschriften = H3 (Unterabschnitte)
        elif text_length < 150:
            return 'h3'
        
        # Fallback für sehr lange "Titel" -> wahrscheinlich kein echter Header
        else:
            return 'content'
    
    for i, element in enumerate(elements):
        element_text = str(element).strip()
        element_type = type(element).__name__
        
        # Prüfe ob Element eine Überschrift ist
        is_header = element_type in ['Title', 'Header'] and len(element_text) > 0
        
        if is_header:
            header_level = classify_header_level(element_text, element_type)
            
            if header_level == 'h1':
                # Neue Hauptüberschrift - Reset aller Sub-Header
                current_headers['h1'] = element_text
                current_headers['h2'] = None
                current_headers['h3'] = None
                current_headers['current_section'] = element_text
                header_stats['titles_found'] += 1
                print(f"  [H1] Neue Hauptüberschrift: '{element_text}'")
                
            elif header_level == 'h2':
                # Neue Unterüberschrift - Reset nur H3
                current_headers['h2'] = element_text
                current_headers['h3'] = None
                current_headers['current_section'] = element_text
                header_stats['headers_found'] += 1
                print(f"  [H2] Neue Unterüberschrift: '{element_text}'")
                
            elif header_level == 'h3':
                # Neue Detail-Überschrift
                current_headers['h3'] = element_text
                current_headers['current_section'] = element_text
                header_stats['headers_found'] += 1
                print(f"  [H3] Neue Detail-Überschrift: '{element_text}'")
        
        # Erstelle Header-Metadaten für dieses Element
        element_header_metadata = {
            'element_type': element_type,
            'is_header': is_header
        }
        
        # Füge aktuelle Header hinzu
        if current_headers['h1']:
            element_header_metadata['section_h1'] = current_headers['h1']
        if current_headers['h2']:
            element_header_metadata['section_h2'] = current_headers['h2']
        if current_headers['h3']:
            element_header_metadata['section_h3'] = current_headers['h3']
        if current_headers['current_section']:
            element_header_metadata['current_section'] = current_headers['current_section']
        
        # Speichere Metadaten für diesen Element-Index
        header_mapping[i] = element_header_metadata
        
        if any(current_headers.values()):
            header_stats['elements_with_headers'] += 1
    
    # Statistik ausgeben
    print(f"[HEADER-ANALYSE] Analyse abgeschlossen:")
    print(f"  - H1-Überschriften erkannt: {header_stats['titles_found']}")
    print(f"  - H2/H3-Überschriften erkannt: {header_stats['headers_found']}")
    print(f"  - Elemente mit Header-Zuordnung: {header_stats['elements_with_headers']}")
    print(f"  - Gesamt-Elemente verarbeitet: {len(elements)}")
    
    return header_mapping

def process_pdf_with_local_unstructured(pdf_path: str, strategy: str = "auto") -> List[Document]:
    """
    Verarbeitet ein PDF direkt mit der lokalen Unstructured-Installation.
    Jedes extrahierte Element wird als separates Document behandelt.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        strategy: Strategie für die Extraktion ('auto', 'hi_res', 'fast', etc.)
        
    Returns:
        Liste von LlamaIndex Document-Objekten
    """
    print(f"[LOKALES UNSTRUCTURED] Verarbeite {pdf_path} mit Strategie '{strategy}'")
    
    if not os.path.exists(pdf_path):
        print(f"❌ Datei nicht gefunden: {pdf_path}")
        return []
    
    try:
        # Importiere Unstructured direkt
        from unstructured.partition.pdf import partition_pdf
        
        # Dateiname für Metadaten
        filename = os.path.basename(pdf_path)
        
        print(f"[LOKALES UNSTRUCTURED] Starte Partitionierung mit Strategie '{strategy}'...")
        start_time = time.time()
        
        # Direkte Partitionierung des PDFs
        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            include_page_breaks=True,
            infer_table_structure=True,  # Tabellen erkennen
            combine_text_under_n_chars=0,  # Keine Textkombination
            new_after_n_chars=None,  # Keine erzwungene Aufteilung
            max_characters=10000,  # Moderate Chunk-Größe
        )
        
        processing_time = time.time() - start_time
        print(f"✓ Lokale Partitionierung abgeschlossen in {processing_time:.2f}s mit {len(elements)} Elementen")
        
        # Analysiere hierarchische Headers
        print(f"[HEADER-ANALYSE] Analysiere {len(elements)} Elemente für Header-Metadaten...")
        header_mapping = analyze_hierarchical_headers(elements)
        
        # Elemente in Document-Objekte umwandeln
        documents = []
        page_numbers = set()
        element_types = {}
        
        for i, element in enumerate(elements):
            # Extrahiere wichtige Metadaten
            metadata = {
                "file_path": pdf_path,
                "file_directory": os.path.dirname(pdf_path),
                "filename": filename
            }
            
            # Extrahiere Seitennummer aus Element-Metadaten
            if hasattr(element, 'metadata') and element.metadata:
                if hasattr(element.metadata, 'page_number'):
                    page_num = element.metadata.page_number
                    if page_num is not None:  # Nur hinzufügen wenn nicht None
                        metadata["page_number"] = page_num
                        page_numbers.add(page_num)
                
                # Weitere Metadaten hinzufügen
                if hasattr(element.metadata, 'coordinates'):
                    metadata["coordinates"] = str(element.metadata.coordinates)
                if hasattr(element.metadata, 'category'):
                    metadata["element_category"] = element.metadata.category
            
            # Füge Header-Metadaten hinzu (aus unserem Mapping)
            if i in header_mapping:
                metadata.update(header_mapping[i])
            
            # Zähle Element-Typen für Statistik
            element_type = type(element).__name__
            element_types[element_type] = element_types.get(element_type, 0) + 1
            
            # Erstelle LlamaIndex Document aus dem Element
            text = str(element)  # Konvertiere Element zu Text
            if text.strip():  # Nur Elemente mit Inhalt
                document = Document(
                    text=text,
                    metadata=metadata
                )
                documents.append(document)
        
        # Statistik ausgeben
        print(f"Extrahierte Elemente nach Typ:")
        for elem_type, count in element_types.items():
            print(f"  - {elem_type}: {count}")
        
        print(f"Gefundene Seitennummern: {sorted(list(page_numbers))}")
        print(f"Gültige Dokumente erstellt: {len(documents)}")
        
        return documents
        
    except ImportError:
        print("❌ Unstructured ist nicht installiert. Installiere es mit: pip install unstructured[pdf]")
        return []
    except Exception as e:
        print(f"❌ Fehler bei der lokalen Unstructured-Verarbeitung: {str(e)}")
        print(traceback.format_exc())
        return []

def load_and_process_pdfs():
    """
    Lädt PDFs mit direkter Unstructured-Nutzung für bessere Elementtrennung.
    Der LlamaIndex UnstructuredReader fasst alle Elemente zu einem Document zusammen,
    was zu falschen Seitenzuordnungen führt. Deshalb verwenden wir Unstructured direkt.
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

    # Teste lokale Unstructured-Installation
    print("\nTeste direkte lokale Unstructured-Installation...")
    local_reader_works = False
    try:
        if pdf_files:
            test_file = os.path.join(PDF_FOLDER, pdf_files[0])
            # Teste direkte Unstructured-Partitionierung
            test_docs = process_pdf_with_local_unstructured(test_file, strategy="auto")
            if test_docs:
                print("✓ Direkte lokale Unstructured-Partitionierung funktioniert!")
                print(f"  Testdatei lieferte {len(test_docs)} separate Dokument-Elemente")
                # Zeige Seitenverteilung
                page_nums = [doc.metadata.get('page_number', 'N/A') for doc in test_docs[:5]]
                print(f"  Erste 5 Seitennummern: {page_nums}")
                local_reader_works = True
            else:
                print("⚠ Direkte Unstructured-Partitionierung lieferte keine Dokumente")
                
        # Fallback-Test mit LlamaIndex UnstructuredReader
        if not local_reader_works:
            print("→ Teste LlamaIndex UnstructuredReader als Fallback...")
            try:
                test_docs = reader.load_data(
                    file=test_file, 
                    unstructured_kwargs={
                        "strategy": "auto",
                        "include_page_breaks": True,
                        "combine_text_under_n_chars": 0
                    }
                )
                if test_docs:
                    print("✓ LlamaIndex UnstructuredReader funktioniert (aber fasst Elemente zusammen)")
                    local_reader_works = True
                else:
                    print("⚠ LlamaIndex UnstructuredReader lieferte keine Dokumente")
            except Exception as e2:
                print(f"❌ LlamaIndex UnstructuredReader funktioniert nicht: {str(e2)}")
                
    except Exception as e:
        print(f"❌ Fehler beim Testen der lokalen Installation: {str(e)}")

    try:
        # Lade jede PDF einzeln mit neuer semantischer Strategie
        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            file_load_success = False
            try:
                print(f"\n[PROZESS] Starte Verarbeitung von '{pdf_file}' mit semantischer Gruppierung")
                start_time = time.time()
                log_resources(f"Vor Laden von '{pdf_file}'")

                file_documents = []
                
                # NEUE STRATEGIE: Direkte semantische Verarbeitung
                try:
                    print(f"Verwende neue semantische Abschnitts-Gruppierung")
                    file_documents = create_enhanced_documents_from_pdf(pdf_path)
                    
                    if file_documents:
                        print(f"✓ Semantische Gruppierung erfolgreich: {len(file_documents)} Abschnitte")
                    else:
                        print("⚠ Semantische Gruppierung lieferte keine Abschnitte")
                        
                except Exception as e:
                    print(f"→ Fehler bei semantischer Gruppierung: {str(e)}")
                    
                # FALLBACK: Alte Strategie falls semantische Gruppierung fehlschlägt
                if not file_documents:
                    print("→ Fallback: Verwende alte Unstructured-Partitionierung")
                    try:
                        # Verwende direkte lokale Unstructured-Partitionierung
                        print(f"Verwende direkte lokale Unstructured-Partitionierung mit 'auto' Strategie")
                        file_documents = process_pdf_with_local_unstructured(pdf_path, strategy="auto")
                        
                        # Falls keine Dokumente, versuche hi_res Strategie
                        if not file_documents:
                            print("→ Fallback: Versuche 'hi_res' Strategie")
                            file_documents = process_pdf_with_local_unstructured(pdf_path, strategy="hi_res")
                            
                    except Exception as e:
                        print(f"→ Fehler bei direkter Unstructured-Nutzung: {str(e)}")
                        
                        # Externe API als weiterer Fallback
                        if use_direct_api and api_available:
                            print(f"→ Fallback: Verwende externe API mit 'auto' Strategie")
                            file_documents = process_pdf_with_direct_api(pdf_path, strategy="auto")
                        
                        # Letzter Fallback: LlamaIndex UnstructuredReader
                        elif local_reader_works:
                            print(f"→ Letzter Fallback: Verwende LlamaIndex UnstructuredReader")
                            try:
                                file_documents = reader.load_data(
                                    file=pdf_path,
                                    unstructured_kwargs={
                                        "strategy": "auto",
                                        "include_page_breaks": True,
                                        "combine_text_under_n_chars": 0,
                                        "max_characters": 100000
                                    }
                                )
                            except Exception:
                                file_documents = reader.load_data(file=pdf_path)
                        else:
                            print("❌ Keine funktionierende Methode gefunden")
                            file_documents = []

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
                # Übernehme nur explizit gewünschte Felder (inkl. neue Header-Metadaten)
                desired_fields = [
                    'file_path', 'file_directory', 'filename', 'page_number',
                    'section_h1', 'section_h2', 'section_h3', 'current_section',
                    'element_type', 'is_header'
                ]
                for key in desired_fields:
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
            
            # Header-Analyse
            headers_h1 = set(d.metadata.get('section_h1') for d in docs_in_file if d.metadata and d.metadata.get('section_h1'))
            headers_h2 = set(d.metadata.get('section_h2') for d in docs_in_file if d.metadata and d.metadata.get('section_h2'))
            elements_with_headers = len([d for d in docs_in_file if d.metadata and d.metadata.get('current_section')])
            
            print(f"Datei: {file_name}")
            print(f"  - Extrahierte Dokument-Elemente: {len(docs_in_file)}")
            print(f"  - Erkannte eindeutige Seitennummern: {sorted(list(page_numbers)) if page_numbers else 'Keine oder nur N/A'}")
            print(f"  - Gesamttextmenge: {sum(len(d.text) for d in docs_in_file):,} Zeichen")
            print(f"  - [HEADER] H1-Überschriften erkannt: {len(headers_h1)} ({list(headers_h1)[:3]}{'...' if len(headers_h1) > 3 else ''})")
            print(f"  - [HEADER] H2-Überschriften erkannt: {len(headers_h2)} ({list(headers_h2)[:3]}{'...' if len(headers_h2) > 3 else ''})")
            print(f"  - [HEADER] Elemente mit Header-Zuordnung: {elements_with_headers}")
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
    print("\n=== Start der PDF-Indexierung mit multilingualen Embeddings ===")
    
    # Prüfe Embedding Requirements
    if not check_embedding_requirements():
        print("\n❌ Abbruch: Nicht alle erforderlichen Bibliotheken verfügbar.")
        return
    
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
        # Collection erstellen mit expliziter Embedding-Funktion für Kompatibilität mit rag_api.py
        from chromadb.utils import embedding_functions
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        chroma_collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=sentence_transformer_ef
        )
        print("✓ Neue ChromaDB Collection mit 768-dim Embeddings erstellt (kompatibel mit rag_api.py)")
    except Exception as e:
        print(f"❌ Fehler bei ChromaDB Initialisierung: {str(e)}")
        return

    # 2. Erstelle ChromaVectorStore
    print("\n2. Erstelle VectorStore...")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Konfiguriere LlamaIndex Settings
    print("\n3. Konfiguriere Embedding Model und Parser...")
    
    # GPU-Nutzung für HuggingFace Embeddings konfigurieren
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding-Berechnung wird auf {device.upper()} ausgeführt")
    
    # Standard Embedding-Konfiguration für sentence-transformers
    embedding_kwargs = {
        "device": device,
    }
    
    # Für GPU: Optimiere für bessere Performance
    if device == "cuda":
        embedding_kwargs["model_kwargs"] = {
            "torch_dtype": torch.float16,  # float16 für GPU-Performance
        }
        print("✓ GPU-Optimierung mit float16 aktiviert")
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        **embedding_kwargs
    )
    
    # Erstelle hybrides Parser-System
    parser_system = create_hybrid_parser_system()
    Settings.node_parser = parser_system['main']  # Setze Haupt-Parser für Settings
    
    print(f"✓ Embedding Model '{EMBED_MODEL_NAME}' konfiguriert auf {device.upper()}")
    print("  - Unterstützt über 50 Sprachen")
    print("  - 768 Embedding-Dimensionen")
    print("  - Bewährtes multilinguales Modell")
    print("  - Optimiert für semantische Ähnlichkeit")
    print("✓ Hybrides Parser-System konfiguriert:")
    print(f"  - Haupt-Parser: Semantische Abschnitte")
    print("  - Backup-Parser: HierarchicalNodeParser für längere Kontexte")
    print("  - Sentence-basierte Chunks für beste semantische Qualität")

    # 4. PDFs laden mit Unstructured
    print("\n4. Lade und verarbeite PDFs mit Unstructured...")
    documents = load_and_process_pdfs()

    if not documents:
        print("Keine Dokumente zum Indexieren gefunden. Beende Programm.")
        return

    # 5. Index erstellen (mit sentence-basiertem Parser)
    print("\n5. Parse Dokumente in Nodes mit SentenceWindowNodeParser...")
    try:
        # Speicherverbrauch vor dem Parsen
        gpu_mem_before = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_before = torch.cuda.memory_allocated() / 1024**2  # in MB
            print(f"GPU-Speicher vor Node-Parsing: {gpu_mem_before:.2f} MB")
            
        start_time = time.time()
        nodes = parser_system['main'].get_nodes_from_documents(documents, show_progress=True)
        parsing_time = time.time() - start_time
        print(f"-> Semantische Abschnitte haben {len(nodes)} Sentence-Nodes in {parsing_time:.2f}s generiert.")

        print("\n--- Erste 3 Sentence-Nodes nach dem Parsen (Überprüfung der Metadaten) ---")
        for i, node in enumerate(nodes[:3]):  # Zeige die ersten 3 Nodes
            print(f"\nSentence-Node {i+1} (ID: {node.id_}):")
            text_preview = node.text[:100].replace('\n', ' ')
            print(f"  Satz (Vorschau): {text_preview}...")
            print(f"  Metadaten: {list(node.metadata.keys())}")
            
            # Prüfe auf sentence-spezifische Metadaten
            if 'window' in node.metadata:
                window_preview = node.metadata['window'][:150].replace('\n', ' ')
                print(f"  Kontext-Fenster (Vorschau): {window_preview}...")
            if 'original_sentence' in node.metadata:
                orig_preview = node.metadata['original_sentence'][:100].replace('\n', ' ')
                print(f"  Original-Satz: {orig_preview}...")
            
            # Zeige Header-Metadaten
            if 'current_section' in node.metadata:
                print(f"  [HEADER] Aktuelle Sektion: {node.metadata['current_section']}")
            if 'section_h1' in node.metadata:
                print(f"  [HEADER] H1-Überschrift: {node.metadata['section_h1']}")
            if 'section_h2' in node.metadata:
                print(f"  [HEADER] H2-Überschrift: {node.metadata['section_h2']}")
            if 'element_type' in node.metadata:
                print(f"  [HEADER] Element-Typ: {node.metadata['element_type']}")
                
            if 'file_path' not in node.metadata or not node.metadata['file_path']:
                print(f"  WARNUNG: 'file_path' fehlt oder ist leer in Metadaten für Node {node.id_}!")
            if 'page_number' not in node.metadata:
                print(f"  INFO: 'page_number' fehlt in Metadaten für Node {node.id_} (kann bei Unstructured vorkommen).")
        print("--- Ende Sentence-Node-Vorschau ---")
        
        # Speicherverbrauch nach dem Parsen
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**2  # in MB
            print(f"GPU-Speicher nach Node-Parsing: {gpu_mem_after:.2f} MB")
            print(f"GPU-Speicher-Differenz: {gpu_mem_after - gpu_mem_before:.2f} MB")

        if not nodes:
            print("❌ Parser hat keine Nodes generiert, obwohl Dokument-Elemente vorhanden waren. Indexierung abgebrochen.")
            return

        print("\n6. Erstelle Index aus Sentence-Nodes...")
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

        print("\n=== Semantische Indexierung erfolgreich abgeschlossen ===")
        print(f"- Verarbeitete Dokument-Elemente (aus Reader): {len(documents)}")
        print(f"- Verarbeitete Sentence-Nodes durch Parser: {len(nodes)}")

        # Prüfe den Docstore *direkt* nach der Indexierung
        final_nodes_in_docstore = index.docstore.docs
        print(f"- Sentence-Nodes im Index Docstore: {len(final_nodes_in_docstore)}")

        # Optional: Prüfe ChromaDB direkt
        try:
            count_in_chroma = chroma_collection.count()
            print(f"- Elemente direkt in ChromaDB Collection '{COLLECTION_NAME}': {count_in_chroma}")
            if count_in_chroma != len(nodes):
                print(f"WARNUNG: Anzahl Nodes ({len(nodes)}) stimmt nicht mit ChromaDB Count ({count_in_chroma}) überein!")
        except Exception as chroma_err:
            print(f"Fehler beim Abfragen von ChromaDB Count: {chroma_err}")

        # Debug-Information über die ersten Sentence-Nodes im Docstore
        print("\n--- Sentence-Node-Struktur im Docstore (erste 3 Nodes) ---")
        for i, (node_id, node) in enumerate(list(final_nodes_in_docstore.items())[:3]):
            print(f"\nSentence-Node {i+1}:")
            print(f"- ID: {node_id}")
            print(f"- Textlänge: {len(node.text)} Zeichen")
            print(f"- Text-Vorschau: {node.text[:100]}...")
            print(f"- Metadaten: {list(node.metadata.keys())}")
            # Zeige sentence-spezifische Infos
            if hasattr(node, 'metadata'):
                if 'window' in node.metadata:
                    print(f"- Hat Kontext-Fenster: Ja ({len(node.metadata['window'])} Zeichen)")
                if 'original_sentence' in node.metadata:
                    print(f"- Hat Original-Satz: Ja ({len(node.metadata['original_sentence'])} Zeichen)")
        print("--- Ende Sentence-Node-Struktur ---")
        
        # Abschließender GPU-Status
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print("\nAbschließender GPU-Status:")
            print(f"- GPU-Speichernutzung: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"- GPU-Speicher reserviert: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        print("\n=== VEKTORDATENBANK ERFOLGREICH ERSTELLT ===")
        print("✓ Semantische Indexierung abgeschlossen")
        print("✓ Vektordatenbank ist bereit für rag_api.py")
        print("\n=== Qualitätsverbesserungen ===")
        print("1. ✓ Semantische Abschnitte für beste Retrieval-Qualität")
        print("2. ✓ Hierarchische Backup-Struktur für verschiedene Anwendungsfälle")
        print("3. ✓ Multilinguale Embedding-Unterstützung für deutsche Inhalte")
        print("4. ✓ GPU-beschleunigte Verarbeitung für bessere Performance")
        print("\n=== Nächste Schritte ===")
        print("- Starten Sie rag_api.py für die LLM-basierte Abfrage")
        print("- Die erstellte ChromaDB wird automatisch von rag_api.py verwendet")
        print(f"- Collection '{COLLECTION_NAME}' enthält {len(nodes)} optimierte Sentence-Nodes")

    except Exception as e:
        print(f"\n❌ Fehler beim Parsen oder Erstellen des Index:")
        print(f"Fehler: {type(e).__name__} - {str(e)}")
        print("\nStack Trace:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
 