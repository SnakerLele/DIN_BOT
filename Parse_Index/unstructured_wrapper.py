"""
Unstructured Wrapper für das Parse_Index System als Fallback

Dieses Modul kapselt die direkte Interaktion mit der Unstructured-Bibliothek
für die PDF-Verarbeitung und -Partitionierung.

Funktionen:
- Direkte lokale Unstructured-Partitionierung
- Element-Extraktion und -Konvertierung
- Fallback-Strategien für verschiedene PDF-Typen
"""

import os
import time
import traceback
from typing import List
from pathlib import Path

from llama_index.core import Document
from .config import UNSTRUCTURED_STRATEGIES

def fallback_local_unstructured_pdf(pdf_path: str, strategy: str = "auto") -> List[Document]:
    """
    Verarbeitet ein PDF direkt mit der lokalen Unstructured-Installation.
    
    Jedes extrahierte Element wird als separates Document behandelt, was eine
    bessere Granularität und Seitenzuordnung ermöglicht als der LlamaIndex Reader.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        strategy: Strategie für die Extraktion ('auto', 'hi_res', 'fast', etc.)
        
    Returns:
        Liste von LlamaIndex Document-Objekten mit Element-spezifischen Metadaten
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
        
        # Direkte Partitionierung des PDFs mit optimierten Parametern
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
        
        # Konvertiere Elemente in LlamaIndex Documents
        documents = convert_elements_to_documents(elements, pdf_path, filename)
        
        # Statistik ausgeben
        print_element_statistics(elements, documents)
        
        return documents
        
    except ImportError:
        print("❌ Unstructured ist nicht installiert. Installiere es mit: pip install unstructured[pdf]")
        return []
    except Exception as e:
        print(f"❌ Fehler bei der lokalen Unstructured-Verarbeitung: {str(e)}")
        print(traceback.format_exc())
        return []

def convert_elements_to_documents(elements: list, pdf_path: str, filename: str) -> List[Document]:
    """
    Konvertiert Unstructured-Elemente in LlamaIndex Document-Objekte.
    
    Args:
        elements: Liste von Unstructured-Elementen
        pdf_path: Vollständiger Pfad zur PDF-Datei
        filename: Dateiname für Metadaten
        
    Returns:
        Liste von Document-Objekten mit angereicherten Metadaten
    """
    print(f"[DOKUMENTKONVERTIERUNG] Konvertiere {len(elements)} Elemente zu Documents...")
    
    documents = []
    
    for i, element in enumerate(elements):
        # Extrahiere wichtige Metadaten
        metadata = {
            "file_path": pdf_path,
            "file_directory": os.path.dirname(pdf_path),
            "filename": filename,
            "element_index": i,  # Index des Elements in der ursprünglichen Liste
            "element_type": type(element).__name__  # Typ des Unstructured-Elements
        }
        
        # Extrahiere Seitennummer aus Element-Metadaten
        if hasattr(element, 'metadata') and element.metadata:
            if hasattr(element.metadata, 'page_number'):
                page_num = element.metadata.page_number
                if page_num is not None:  # Nur hinzufügen wenn nicht None
                    metadata["page_number"] = page_num
            
            # Weitere strukturelle Metadaten hinzufügen
            if hasattr(element.metadata, 'coordinates'):
                metadata["coordinates"] = str(element.metadata.coordinates)
            if hasattr(element.metadata, 'category'):
                metadata["element_category"] = element.metadata.category
            if hasattr(element.metadata, 'filename'):
                metadata["original_filename"] = element.metadata.filename
        
        # Text extrahieren und validieren
        text = str(element).strip()
        if text:  # Nur Elemente mit Inhalt verarbeiten
            document = Document(
                text=text,
                metadata=metadata
            )
            documents.append(document)
    
    print(f"✓ {len(documents)} gültige Dokumente erstellt (von {len(elements)} Elementen)")
    return documents

def print_element_statistics(elements: list, documents: List[Document]):
    """
    Gibt detaillierte Statistiken über die extrahierten Elemente aus.
    
    Args:
        elements: Ursprüngliche Unstructured-Elemente  
        documents: Konvertierte LlamaIndex Documents
    """
    # Element-Typen statistik
    element_types = {}
    page_numbers = set()
    
    for element in elements:
        element_type = type(element).__name__
        element_types[element_type] = element_types.get(element_type, 0) + 1
        
        # Sammle Seitennummern
        if hasattr(element, 'metadata') and element.metadata:
            if hasattr(element.metadata, 'page_number') and element.metadata.page_number is not None:
                page_numbers.add(element.metadata.page_number)
    
    print(f"Extrahierte Elemente nach Typ:")
    for elem_type, count in sorted(element_types.items()):
        print(f"  - {elem_type}: {count}")
    
    print(f"Gefundene Seitennummern: {sorted(list(page_numbers)) if page_numbers else 'Keine'}")
    print(f"Gültige Dokumente erstellt: {len(documents)}")

def test_unstructured_strategies(pdf_path: str) -> dict:
    """
    Testet verschiedene Unstructured-Strategien für eine PDF-Datei.
    
    Args:
        pdf_path: Pfad zur Test-PDF
        
    Returns:
        dict: Ergebnisse der verschiedenen Strategien mit Performance-Metriken
    """
    if not os.path.exists(pdf_path):
        print(f"❌ Test-PDF nicht gefunden: {pdf_path}")
        return {}
    
    print(f"\n[STRATEGIE-TEST] Teste verschiedene Unstructured-Strategien für: {os.path.basename(pdf_path)}")
    
    results = {}
    
    for strategy_name, strategy_value in UNSTRUCTURED_STRATEGIES.items():
        print(f"\n--- Teste Strategie: {strategy_name} ({strategy_value}) ---")
        
        start_time = time.time()
        try:
            documents = fallback_local_unstructured_pdf(pdf_path, strategy_value)
            processing_time = time.time() - start_time
            
            # Analysiere Ergebnisse
            page_count = len(set(doc.metadata.get('page_number') for doc in documents 
                               if doc.metadata.get('page_number') is not None))
            
            results[strategy_name] = {
                'success': True,
                'documents_count': len(documents),
                'pages_found': page_count,
                'processing_time': processing_time,
                'strategy': strategy_value,
                'avg_document_length': sum(len(doc.text) for doc in documents) / len(documents) if documents else 0
            }
            
            print(f"✓ {strategy_name}: {len(documents)} Dokumente in {processing_time:.2f}s")
            print(f"  - Seiten erkannt: {page_count}")
            print(f"  - Durchschnittliche Dokumentlänge: {results[strategy_name]['avg_document_length']:.0f} Zeichen")
            
        except Exception as e:
            processing_time = time.time() - start_time
            results[strategy_name] = {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'strategy': strategy_value
            }
            print(f"❌ {strategy_name} fehlgeschlagen: {str(e)}")
    
    # Empfehlung basierend auf Ergebnissen
    successful_strategies = {k: v for k, v in results.items() if v['success']}
    if successful_strategies:
        best_strategy = max(successful_strategies.items(), 
                           key=lambda x: (x[1]['documents_count'], -x[1]['processing_time']))
        print(f"\n💡 Empfohlene Strategie: {best_strategy[0]} ({best_strategy[1]['strategy']})")
    
    return results

def validate_unstructured_installation():
    """
    Validiert die Unstructured-Installation und deren Abhängigkeiten.
    
    Returns:
        dict: Validierungsergebnisse
    """
    print("\n===== UNSTRUCTURED VALIDIERUNG =====")
    
    validation_results = {
        'unstructured_available': False,
        'pdf_support': False,
        'dependencies_ok': False,
        'version_info': {},
        'errors': []
    }
    
    # Prüfe Hauptbibliothek
    try:
        import unstructured
        validation_results['unstructured_available'] = True
        validation_results['version_info']['unstructured'] = unstructured.__version__
        print(f"✓ Unstructured installiert: Version {unstructured.__version__}")
    except ImportError as e:
        error = "❌ Unstructured nicht installiert"
        print(error)
        validation_results['errors'].append(error)
        return validation_results
    
    # Prüfe PDF-Support
    try:
        from unstructured.partition.pdf import partition_pdf
        validation_results['pdf_support'] = True
        print("✓ PDF-Partitionierung verfügbar")
    except ImportError as e:
        error = f"❌ PDF-Support nicht verfügbar: {str(e)}"
        print(error)
        validation_results['errors'].append(error)
    
    # Prüfe wichtige Abhängigkeiten
    dependencies = ['pdfplumber', 'pdfminer', 'pytesseract']
    available_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            available_deps.append(dep)
            print(f"✓ {dep} verfügbar")
        except ImportError:
            print(f"⚠ {dep} nicht verfügbar (optional)")
    
    validation_results['dependencies_ok'] = len(available_deps) >= 1  # Mindestens eine PDF-Lib
    validation_results['available_dependencies'] = available_deps
    
    print("=" * 37)
    
    if validation_results['unstructured_available'] and validation_results['pdf_support']:
        print("✅ Unstructured-Installation ist funktionsfähig!")
    else:
        print("❌ Unstructured-Installation hat Probleme!")
        print("Installiere mit: pip install unstructured[pdf]")
    
    return validation_results 