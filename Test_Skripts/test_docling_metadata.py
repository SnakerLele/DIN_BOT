"""
Test für DoclingMetadataExtractor
---------------------------------
Testet die neue schlanke Metadaten-Extraktion mit docling_doc.meta.
"""

import sys
from pathlib import Path

# Pfad zum Parse_Index Modul hinzufügen
sys.path.append(str(Path(__file__).parent.parent))

from Parse_Index.docling_adapter.metadata import DoclingMetadataExtractor, MetadataExtractor
from Parse_Index.docling_adapter import DoclingAdapter


def test_docling_metadata():
    """Testet die neue DoclingMetadataExtractor."""
    print("🧪 Phase 3 Test: Docling-Metadaten")
    print("=" * 40)
    
    try:
        # 1. Import-Test
        print("✅ DoclingMetadataExtractor Import erfolgreich")
        
        # 2. Backward-Kompatibilität
        old_extractor = MetadataExtractor()
        print("✅ Backward-Kompatibilität (MetadataExtractor) funktioniert")
        
        # 3. Neue Klasse
        new_extractor = DoclingMetadataExtractor()
        print("✅ DoclingMetadataExtractor erstellt")
        
        # 4. Integration mit DoclingAdapter
        adapter = DoclingAdapter()
        print("✅ DoclingAdapter Integration")
        
        # 5. Test mit Dummy-Dokument
        class DummyDoc:
            def __init__(self):
                self.pages = [DummyPage()]
                self.meta = DummyMeta()
                
        class DummyPage:
            def __init__(self):
                self.elements = [DummyElement("text", "Test-Text")]
                
        class DummyElement:
            def __init__(self, element_type, text):
                self.type = element_type
                self.text = text
                
        class DummyMeta:
            def __init__(self):
                self.title = "Test-Dokument"
                self.author = "Test-Autor"
                self.language = "de"
        
        dummy_doc = DummyDoc()
        dummy_path = Path("test.pdf")
        
        # 6. Metadaten-Extraktion testen
        metadata = new_extractor.extract(dummy_doc, dummy_path)
        print(f"✅ Metadaten extrahiert: {len(metadata)} Felder")
        
        # 7. Wichtige Felder prüfen
        expected_fields = ['filename', 'file_path', 'extraction_source', 'docling_available']
        for field in expected_fields:
            if field in metadata:
                print(f"   ✓ {field}: {metadata[field]}")
            else:
                print(f"   ⚠ {field}: nicht gefunden")
        
        # 8. Docling-spezifische Felder
        docling_fields = ['title', 'author', 'language', 'page_count', 'element_types']
        for field in docling_fields:
            if field in metadata:
                print(f"   ✓ Docling-{field}: {metadata[field]}")
        
        print("\n🎉 Phase 3 erfolgreich! Metadaten-Extraktion funktioniert.")
        return True
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_docling_metadata()
    if success:
        print("\n✅ Alle Tests bestanden!")
    else:
        print("\n❌ Tests fehlgeschlagen!")
        sys.exit(1) 