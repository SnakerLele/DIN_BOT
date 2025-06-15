#!/usr/bin/env python3
"""
Schneller Test für Phase 2 - Docling-Chunking
"""

import sys
from pathlib import Path

# Parse_Index zum Python-Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_phase2():
    print("🧪 Phase 2 Test: Docling-Chunking")
    print("=" * 40)
    
    try:
        # Test 1: Import
        from Parse_Index.docling_adapter.sectionizer import DoclingChunker
        print("✅ DoclingChunker Import erfolgreich")
        
        # Test 2: Chunker erstellen
        chunker = DoclingChunker('hybrid')
        print("✅ DoclingChunker erstellt")
        
        # Test 3: Info abrufen
        info = chunker.get_strategy_info()
        print(f"✅ Chunker-Info: {info['chunker_type']}")
        print(f"   Nutzt Docling: {info['uses_docling_chunker']}")
        
        # Test 4: DoclingAdapter
        from Parse_Index.docling_adapter import DoclingAdapter
        print("✅ DoclingAdapter Import erfolgreich")
        
        adapter = DoclingAdapter()
        print("✅ DoclingAdapter erstellt")
        
        adapter_info = adapter.chunker.get_strategy_info()
        print(f"✅ Adapter-Chunker: {adapter_info['chunker_type']}")
        
        # Test 5: Backward-Kompatibilität
        from Parse_Index.docling_adapter.sectionizer import SectionExtractor
        old_chunker = SectionExtractor('by_page')
        print("✅ Backward-Kompatibilität (SectionExtractor) funktioniert")
        
        print("\n🎉 Phase 2 erfolgreich! Alle Tests bestanden.")
        return True
        
    except Exception as e:
        print(f"❌ Fehler: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase2()
    sys.exit(0 if success else 1) 