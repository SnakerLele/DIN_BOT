"""
Test-Skript für Tabellen-Extraktion im Document Enhancer

Testet die neue Tabellen-Erkennungs- und Verarbeitungsfunktionalität:
- Erkennt Tabellen in PDF-Dokumenten
- Validiert Tabellen-Metadaten
- Überprüft Markdown-Konvertierung
- Testet ChromaDB-Kompatibilität der Metadaten
"""

import sys
import os
from pathlib import Path

# Füge Parse_Index Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent / "Parse_Index"))

from document_enhancer import semantic_enhanced_pdf, validate_document_quality

def test_table_extraction(pdf_path: str):
    """
    Testet die Tabellen-Extraktion für eine gegebene PDF.
    
    Args:
        pdf_path: Pfad zur Test-PDF
    """
    print(f"🧪 [TEST] Starte Tabellen-Extraktion-Test für: {pdf_path}")
    print("=" * 80)
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF-Datei nicht gefunden: {pdf_path}")
        return False
    
    try:
        # 1. Führe die semantische PDF-Verarbeitung durch
        print("1️⃣ Führe semantische PDF-Verarbeitung durch...")
        documents = semantic_enhanced_pdf(pdf_path)
        
        if not documents:
            print("❌ Keine Dokumente erstellt - Test fehlgeschlagen")
            return False
        
        print(f"✅ {len(documents)} Dokumente erfolgreich erstellt")
        print()
        
        # 2. Analysiere Tabellen-Dokumente
        print("2️⃣ Analysiere Tabellen-Dokumente...")
        table_docs = [d for d in documents if d.metadata.get('is_table', False)]
        text_docs = [d for d in documents if not d.metadata.get('is_table', False)]
        
        print(f"   📊 Tabellen-Dokumente gefunden: {len(table_docs)}")
        print(f"   📄 Text-Dokumente gefunden: {len(text_docs)}")
        
        if len(table_docs) == 0:
            print("⚠️  Keine Tabellen gefunden - prüfe ob PDF Tabellen enthält")
        else:
            print("✅ Tabellen erfolgreich erkannt!")
        print()
        
        # 3. Detailanalyse der Tabellen
        if table_docs:
            print("3️⃣ Detailanalyse der erkannten Tabellen...")
            for i, table_doc in enumerate(table_docs, 1):
                meta = table_doc.metadata
                print(f"   🗃️  Tabelle {i}:")
                print(f"      - Abschnitt: {meta.get('section_title', 'Unknown')}")
                print(f"      - Seite: {meta.get('page_number', 'Unknown')}")
                print(f"      - Dimensionen: {meta.get('table_rows', '?')}x{meta.get('table_cols', '?')}")
                print(f"      - Qualität: {meta.get('table_quality', 'Unknown')}")
                print(f"      - Extraktionsmethode: {meta.get('table_extraction_method', 'Unknown')}")
                print(f"      - Textlänge: {meta.get('table_text_length', 0)} Zeichen")
                
                # **VOLLSTÄNDIGER TABELLEN-INHALT**
                print(f"      📋 TABELLEN-INHALT:")
                print("      " + "="*60)
                
                # Formatiere den Tabellen-Text schön für die Ausgabe
                table_text = table_doc.text.strip()
                if table_text:
                    lines = [line for line in table_text.split('\n') if line.strip()]
                    
                    # Bei sehr langen Tabellen (>30 Zeilen) zeige nur Anfang und Ende
                    if len(lines) > 30:
                        print(f"      [TABELLE HAT {len(lines)} ZEILEN - ZEIGE ERSTE 15 UND LETZTE 5]")
                        print("      ")
                        # Erste 15 Zeilen
                        for line in lines[:15]:
                            print(f"      {line}")
                        print("      ... [Zeilen ausgelassen] ...")
                        # Letzte 5 Zeilen
                        for line in lines[-5:]:
                            print(f"      {line}")
                    else:
                        # Zeige alle Zeilen der Tabelle eingerückt
                        for line in lines:
                            print(f"      {line}")
                else:
                    print("      [LEER - Kein Tabellen-Text verfügbar]")
                
                print("      " + "="*60)
                print()
        
        # 4. Qualitäts-Validierung
        print("4️⃣ Führe Qualitäts-Validierung durch...")
        quality_metrics = validate_document_quality(documents)
        
        print(f"   📋 Qualitäts-Score: {quality_metrics['quality_score']:.1f}/100")
        print(f"   📊 Tabellen-Erfolgsrate: {quality_metrics['table_extraction_success_rate']:.1%}")
        print(f"   ✅ Gute Tabellen: {quality_metrics['tables_good_quality']}")
        print(f"   ⚠️  Mittlere Tabellen: {quality_metrics['tables_medium_quality']}")
        print(f"   ❌ Schlechte Tabellen: {quality_metrics['tables_poor_quality']}")
        
        if quality_metrics['warnings']:
            print("   ⚠️  Warnungen:")
            for warning in quality_metrics['warnings']:
                print(f"      - {warning}")
        print()
        
        # 5. ChromaDB-Kompatibilitäts-Test
        print("5️⃣ Teste ChromaDB-Kompatibilität der Metadaten...")
        chroma_compatible = True
        unsupported_types = []
        
        for doc in documents:
            for key, value in doc.metadata.items():
                if not isinstance(value, (str, int, float, bool, type(None))):
                    chroma_compatible = False
                    unsupported_types.append((key, type(value).__name__))
        
        if chroma_compatible:
            print("   ✅ Alle Metadaten sind ChromaDB-kompatibel")
        else:
            print("   ❌ Nicht-kompatible Metadaten gefunden:")
            for key, type_name in set(unsupported_types):
                print(f"      - {key}: {type_name}")
        print()
        
        # 6. Test-Zusammenfassung
        print("6️⃣ Test-Zusammenfassung:")
        success_criteria = {
            "Dokumente erstellt": len(documents) > 0,
            "Tabellen erkannt": len(table_docs) > 0,
            "Qualität >= 70": quality_metrics['quality_score'] >= 70,
            "ChromaDB-kompatibel": chroma_compatible,
            "Keine schweren Warnungen": len([w for w in quality_metrics['warnings'] if 'schlechter Qualität' in w]) == 0
        }
        
        passed_tests = sum(success_criteria.values())
        total_tests = len(success_criteria)
        
        print(f"   📊 Tests bestanden: {passed_tests}/{total_tests}")
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion}")
        
        overall_success = passed_tests == total_tests
        print()
        print(f"🎯 [GESAMT-ERGEBNIS] Test {'ERFOLGREICH' if overall_success else 'FEHLGESCHLAGEN'}")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Fehler während des Tests: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def find_test_pdfs(base_path: str = ".") -> list:
    """
    Sucht nach Test-PDF-Dateien in verschiedenen Verzeichnissen.
    
    Args:
        base_path: Basis-Suchpfad
        
    Returns:
        Liste von gefundenen PDF-Pfaden
    """
    search_paths = [
        os.path.join(base_path, "Sample_PDFs"),
        os.path.join(base_path, "test_files"),
        os.path.join(base_path, "data"),
        os.path.join(base_path, "PDFs"),  # Gefundenes PDFs Verzeichnis
        
        base_path
    ]
    
    found_pdfs = []
    for search_path in search_paths:
        if os.path.exists(search_path):
            for file in os.listdir(search_path):
                if file.lower().endswith('.pdf'):
                    found_pdfs.append(os.path.join(search_path, file))
    
    return found_pdfs

def main():
    """Hauptfunktion des Test-Skripts."""
    print("🔬 TABELLEN-EXTRAKTION TESTSKRIPT")
    print("=" * 50)
    
    # Suche nach Test-PDFs
    print("🔍 Suche nach Test-PDF-Dateien...")
    test_pdfs = find_test_pdfs()
    
    if not test_pdfs:
        print("❌ Keine PDF-Dateien zum Testen gefunden!")
        print("💡 Lege eine PDF-Datei mit Tabellen in einen dieser Ordner:")
        print("   - Sample_PDFs/")
        print("   - test_files/")
        print("   - data/")
        print("   - aktuelles Verzeichnis")
        return
    
    print(f"📁 {len(test_pdfs)} PDF-Datei(en) gefunden:")
    for i, pdf_path in enumerate(test_pdfs, 1):
        print(f"   {i}. {pdf_path}")
    print()
    
    # Teste jede gefundene PDF
    total_tests = len(test_pdfs)
    successful_tests = 0
    
    for i, pdf_path in enumerate(test_pdfs, 1):
        print(f"📋 Test {i}/{total_tests}: {os.path.basename(pdf_path)}")
        print("-" * 50)
        
        success = test_table_extraction(pdf_path)
        if success:
            successful_tests += 1
        
        print()
    
    # Gesamt-Zusammenfassung
    print("🎯 GESAMT-ZUSAMMENFASSUNG")
    print("=" * 50)
    print(f"Tests durchgeführt: {total_tests}")
    print(f"Erfolgreich: {successful_tests}")
    print(f"Fehlgeschlagen: {total_tests - successful_tests}")
    print(f"Erfolgsrate: {successful_tests/total_tests:.1%}")
    
    if successful_tests == total_tests:
        print("🎉 Alle Tests erfolgreich! Tabellen-Extraktion funktioniert korrekt.")
    else:
        print("⚠️  Einige Tests fehlgeschlagen. Überprüfe die Ausgaben oben.")

if __name__ == "__main__":
    main() 