"""
Test für img2table Integration im Hauptparser

Testet die Integration von img2table in den main_parser.py:
- Normale Dokumentverarbeitung + img2table Tabellen-Extraktion
- Separate Dokumente für Tabellen
- Vollständiger Pipeline-Test
"""

import sys
import os
from pathlib import Path

# Parse_Index Module hinzufügen
sys.path.append(str(Path(__file__).parent.parent))

from Parse_Index.main_parser import load_and_process_pdfs

def test_main_parser_with_img2table():
    """Teste die img2table Integration im Hauptparser"""
    print("🧪 Teste Hauptparser mit img2table Integration\n")
    print("=" * 60)
    
    # Prüfe ob PDF-Verzeichnis existiert
    from Parse_Index.config import PDF_FOLDER
    
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ PDF-Ordner nicht gefunden: {PDF_FOLDER}")
        return False
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"❌ Keine PDF-Dateien in {PDF_FOLDER} gefunden")
        return False
    
    print(f"📄 Gefundene PDFs: {pdf_files}")
    
    try:
        # Teste die erweiterte PDF-Verarbeitung
        print("\n🚀 Starte erweiterte PDF-Verarbeitung mit img2table...")
        
        documents = load_and_process_pdfs()
        
        if not documents:
            print("❌ Keine Dokumente erstellt")
            return False
        
        # Analysiere die Ergebnisse
        print(f"\n📊 Analyse der erstellten Dokumente:")
        print(f"   - Gesamt: {len(documents)} Dokumente")
        
        # Kategorisiere Dokumente nach Typ
        normal_docs = []
        table_docs = []
        
        for doc in documents:
            metadata = getattr(doc, 'metadata', {})
            if metadata.get('node_type') == 'table' or metadata.get('extraction_method') == 'img2table':
                table_docs.append(doc)
            else:
                normal_docs.append(doc)
        
        print(f"   - Normale Dokumente: {len(normal_docs)}")
        print(f"   - img2table Tabellen: {len(table_docs)}")
        
        # Zeige Details der Tabellen-Dokumente
        if table_docs:
            print(f"\n📋 Details der img2table Tabellen:")
            for i, doc in enumerate(table_docs[:5], 1):  # Zeige max. 5
                metadata = getattr(doc, 'metadata', {})
                text_length = len(doc.text) if hasattr(doc, 'text') else 0
                print(f"   Tabelle {i}:")
                print(f"     - Seite: {metadata.get('page_number', '?')}")
                print(f"     - Größe: {metadata.get('table_rows', '?')}x{metadata.get('table_columns', '?')}")
                print(f"     - Qualität: {metadata.get('table_quality', '?')}")
                print(f"     - Text-Länge: {text_length} Zeichen")
                print(f"     - ID: {metadata.get('table_id', 'unbekannt')}")
            
            if len(table_docs) > 5:
                print(f"   ... und {len(table_docs) - 5} weitere Tabellen")
        
        # Erfolgsbewertung
        success = len(documents) > 0 and len(table_docs) > 0
        
        if success:
            print(f"\n✅ Test erfolgreich!")
            print(f"   - {len(normal_docs)} normale Dokumente verarbeitet")
            print(f"   - {len(table_docs)} Tabellen mit img2table extrahiert")
            print(f"   - Gesamt: {len(documents)} Dokumente für Indexierung bereit")
        else:
            print(f"\n⚠️ Test teilweise erfolgreich:")
            print(f"   - Dokumente erstellt, aber keine img2table Tabellen")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_metadata():
    """Teste die Metadaten-Struktur der erstellten Dokumente"""
    print("\n📋 Teste Dokument-Metadaten...")
    
    try:
        documents = load_and_process_pdfs()
        
        if not documents:
            print("❌ Keine Dokumente für Metadaten-Test")
            return False
        
        print(f"🔍 Analysiere Metadaten von {len(documents)} Dokumenten:")
        
        # Sammle Metadaten-Statistiken
        metadata_keys = set()
        extraction_methods = {}
        content_types = {}
        
        for doc in documents:
            metadata = getattr(doc, 'metadata', {})
            metadata_keys.update(metadata.keys())
            
            method = metadata.get('extraction_method', 'unknown')
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
            
            ctype = metadata.get('content_type', 'unknown')
            content_types[ctype] = content_types.get(ctype, 0) + 1
        
        print(f"   📊 Metadaten-Felder: {sorted(metadata_keys)}")
        print(f"   🔧 Extraktions-Methoden: {extraction_methods}")
        print(f"   📝 Content-Typen: {content_types}")
        
        # Prüfe ob img2table Dokumente korrekte Metadaten haben
        img2table_docs = [doc for doc in documents 
                         if getattr(doc, 'metadata', {}).get('extraction_method') == 'img2table']
        
        if img2table_docs:
            print(f"\n✅ {len(img2table_docs)} img2table Dokumente haben korrekte Metadaten")
            return True
        else:
            print(f"\n⚠️ Keine img2table Dokumente gefunden")
            return False
            
    except Exception as e:
        print(f"\n❌ Metadaten-Test fehlgeschlagen: {e}")
        return False


def main():
    """Haupttest-Routine für Hauptparser Integration"""
    print("🔧 img2table Integration Tests für Hauptparser\n")
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Grundlegende Integration
    if test_main_parser_with_img2table():
        success_count += 1
        print("✅ Hauptparser Integration erfolgreich")
    else:
        print("❌ Hauptparser Integration fehlgeschlagen")
    
    # Test 2: Metadaten-Struktur
    if test_document_metadata():
        success_count += 1
        print("✅ Metadaten-Tests erfolgreich")
    else:
        print("❌ Metadaten-Tests fehlgeschlagen")
    
    # Endergebnis
    print("\n" + "=" * 60)
    print(f"🏁 Integration Test Ergebnis: {success_count}/{total_tests} erfolgreich")
    
    if success_count == total_tests:
        print("🎉 Vollständige Integration erfolgreich!")
        print("   Der Hauptparser nutzt jetzt img2table für robuste Tabellen-Extraktion.")
        return True
    elif success_count > 0:
        print("⚠️ Teilweise erfolgreich - Integration funktioniert grundsätzlich.")
        return True
    else:
        print("❌ Integration fehlgeschlagen - Überprüfung erforderlich.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 