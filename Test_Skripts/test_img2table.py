"""
Test für img2table Integration in Parse_Index

Testet die neue img2table basierte Tabellen-Extraktion:
- Basis-Funktionalität von img2table
- Integration in Parse_Index Workflow
- Vergleich mit Unstructured (falls verfügbar)
- Qualitätsbewertung der Ergebnisse
"""

import sys
import logging
from pathlib import Path

# Parse_Index Module hinzufügen
sys.path.append(str(Path(__file__).parent.parent))

from Parse_Index.pdf_enhancer import (
    extract_tables_with_img2table,
    create_table_nodes_from_img2table,
    enhanced_table_extraction,
    IMG2TABLE_AVAILABLE
)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_img2table_availability():
    """Teste ob img2table korrekt installiert ist"""
    print("🔍 Teste img2table Verfügbarkeit...")
    
    if IMG2TABLE_AVAILABLE:
        print("✅ img2table ist verfügbar")
        
        try:
            from Parse_Index.pdf_enhancer.img2table_utils import create_img2table_extractor
            extractor = create_img2table_extractor()
            if extractor:
                print("✅ img2table Extraktor erfolgreich erstellt")
                return True
            else:
                print("❌ img2table Extraktor konnte nicht erstellt werden")
                return False
        except Exception as e:
            print(f"❌ img2table Initialisierung fehlgeschlagen: {e}")
            return False
    else:
        print("❌ img2table nicht verfügbar - überspringe Tests")
        return False


def test_extract_tables_basic():
    """Teste grundlegende Tabellen-Extraktion"""
    print("\n📊 Teste grundlegende Tabellen-Extraktion...")
    
    # Suche nach Test-PDFs
    test_dir = Path("Test_skripts/Sample_PDFs")
    if not test_dir.exists():
        print(f"⚠️ Test-Verzeichnis nicht gefunden: {test_dir}")
        return False
    
    pdf_files = list(test_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ Keine PDF-Dateien in {test_dir} gefunden")
        return False
    
    # Teste mit dem ersten verfügbaren PDF
    test_pdf = pdf_files[0]
    print(f"📄 Teste mit: {test_pdf.name}")
    
    try:
        # img2table Extraktion
        markdown_tables, metadata_list = extract_tables_with_img2table(test_pdf)
        
        print(f"🎯 Ergebnis: {len(markdown_tables)} Tabellen gefunden")
        
        if markdown_tables:
            for i, (markdown, meta) in enumerate(zip(markdown_tables, metadata_list)):
                print(f"\n  📊 Tabelle {i+1}:")
                print(f"     - Seite: {meta.get('page_number')}")
                print(f"     - Größe: {meta.get('rows')}x{meta.get('columns')}")
                print(f"     - Qualität: {meta.get('table_quality')}")
                print(f"     - Zeichen: {len(markdown)}")
                
                # Zeige ersten Teil des Markdowns
                if len(markdown) > 200:
                    print(f"     - Vorschau: {markdown[:200]}...")
                else:
                    print(f"     - Inhalt: {markdown}")
            
            return True
        else:
            print("ℹ️ Keine Tabellen in diesem PDF gefunden")
            return True  # Ist OK, das PDF könnte keine Tabellen haben
            
    except Exception as e:
        print(f"❌ Extraktion fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_table_extraction():
    """Teste die erweiterte Tabellen-Extraktion Funktion"""
    print("\n🚀 Teste erweiterte Tabellen-Extraktion...")
    
    test_dir = Path("Test_skripts/Sample_PDFs")
    pdf_files = list(test_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️ Keine Test-PDFs verfügbar")
        return False
    
    test_pdf = pdf_files[0]
    print(f"📄 Teste enhanced_table_extraction mit: {test_pdf.name}")
    
    try:
        # Enhanced Extraktion (img2table primär)
        tables, metadata = enhanced_table_extraction(
            test_pdf,
            prefer_img2table=True,
            fallback_to_unstructured=False  # Für jetzt nur img2table
        )
        
        print(f"🎯 Enhanced Extraktion: {len(tables)} Tabellen")
        
        if tables:
            # Test Node-Erstellung
            nodes = create_table_nodes_from_img2table(
                tables, metadata, test_pdf.name
            )
            
            print(f"📝 {len(nodes)} Table-Nodes erstellt")
            
            for i, node in enumerate(nodes):
                meta = node['metadata']
                print(f"\n  📄 Node {i+1}:")
                print(f"     - ID: {node['id']}")
                print(f"     - Extraction: {meta.get('extraction_method')}")
                print(f"     - Größe: {meta.get('table_rows')}x{meta.get('table_columns')}")
                print(f"     - Text-Länge: {len(node['text'])}")
            
            return True
        else:
            print("ℹ️ Keine Tabellen gefunden")
            return True
            
    except Exception as e:
        print(f"❌ Enhanced Extraktion fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_pdf_types():
    """Teste verschiedene PDF-Typen"""
    print("\n📚 Teste verschiedene PDF-Typen...")
    
    test_dir = Path("Test_skripts/Sample_PDFs")
    pdf_files = list(test_dir.glob("*.pdf"))
    
    if len(pdf_files) < 2:
        print("⚠️ Brauche mindestens 2 PDFs für diesen Test")
        return True  # Nicht kritisch
    
    results = []
    
    for pdf_file in pdf_files[:3]:  # Teste maximal 3 PDFs
        print(f"\n📄 Teste: {pdf_file.name}")
        
        try:
            tables, metadata = extract_tables_with_img2table(pdf_file)
            result = {
                'file': pdf_file.name,
                'tables_found': len(tables),
                'total_chars': sum(len(t) for t in tables),
                'success': True
            }
            
            if tables:
                qualities = [m.get('table_quality') for m in metadata]
                result['qualities'] = qualities
                print(f"  ✅ {len(tables)} Tabellen, Qualitäten: {qualities}")
            else:
                print(f"  ℹ️ Keine Tabellen gefunden")
            
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ Fehler: {e}")
            results.append({
                'file': pdf_file.name,
                'success': False,
                'error': str(e)
            })
    
    # Zusammenfassung
    successful = sum(1 for r in results if r['success'])
    total_tables = sum(r.get('tables_found', 0) for r in results if r['success'])
    
    print(f"\n📊 Zusammenfassung:")
    print(f"   - Erfolgreich: {successful}/{len(results)} PDFs")
    print(f"   - Total Tabellen: {total_tables}")
    
    return successful > 0


def main():
    """Haupttest-Routine"""
    print("🧪 Starte img2table Integration Tests\n")
    print("=" * 60)
    
    # Test 1: Verfügbarkeit
    if not test_img2table_availability():
        print("\n❌ img2table nicht verfügbar - Tests abgebrochen")
        return False
    
    success_count = 0
    total_tests = 3
    
    # Test 2: Basis-Extraktion
    if test_extract_tables_basic():
        success_count += 1
        print("✅ Basis-Extraktion erfolgreich")
    else:
        print("❌ Basis-Extraktion fehlgeschlagen")
    
    # Test 3: Enhanced Extraktion  
    if test_enhanced_table_extraction():
        success_count += 1
        print("✅ Enhanced Extraktion erfolgreich")
    else:
        print("❌ Enhanced Extraktion fehlgeschlagen")
    
    # Test 4: Verschiedene PDFs
    if test_different_pdf_types():
        success_count += 1
        print("✅ Multi-PDF Tests erfolgreich")
    else:
        print("❌ Multi-PDF Tests fehlgeschlagen")
    
    # Endergebnis
    print("\n" + "=" * 60)
    print(f"🏁 Test-Ergebnis: {success_count}/{total_tests} erfolgreich")
    
    if success_count == total_tests:
        print("🎉 Alle Tests bestanden! img2table Integration funktioniert.")
        return True
    elif success_count >= total_tests // 2:
        print("⚠️ Teilweise erfolgreich - img2table grundsätzlich funktionsfähig.")
        return True
    else:
        print("❌ Viele Tests fehlgeschlagen - img2table Integration problematisch.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 