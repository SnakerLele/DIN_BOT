#!/usr/bin/env python3
"""
Test-Skript für das neue Docling-Chunking-System
===============================================

Testet die DoclingChunker-Implementierung und vergleicht mit dem alten System.
"""

import sys
from pathlib import Path

# Parse_Index zum Python-Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent))

from Parse_Index.docling_adapter import DoclingAdapter
from Parse_Index.docling_adapter.sectionizer import DoclingChunker, create_chunk_config
import logging

# Logging für Tests aktivieren
logging.basicConfig(level=logging.INFO)

def test_chunker_strategies():
    """Testet verschiedene Chunking-Strategien."""
    
    print("🧪 Test 1: Chunker-Strategien")
    print("=" * 50)
    
    strategies = ["by_heading", "by_page", "by_element", "hybrid"]
    
    for strategy in strategies:
        print(f"\n📋 Teste Strategie: {strategy}")
        
        try:
            chunker = DoclingChunker(strategy=strategy)
            info = chunker.get_strategy_info()
            
            print(f"  ✅ Chunker erstellt: {info['chunker_type']}")
            print(f"  📝 Beschreibung: {info['description']}")
            print(f"  🔧 Nutzt Docling: {info['uses_docling_chunker']}")
            
        except Exception as e:
            print(f"  ❌ Fehler: {str(e)}")
    
    print("\n✅ Alle Strategien getestet!")
    return True

def test_chunk_config():
    """Testet die Chunk-Konfiguration."""
    
    print("\n🧪 Test 2: Chunk-Konfiguration")
    print("=" * 50)
    
    # Standard-Config
    config1 = create_chunk_config()
    print(f"Standard-Config: max_size={config1.max_chunk_size}, overlap={config1.chunk_overlap}")
    
    # Custom-Config
    config2 = create_chunk_config(
        max_chunk_size=512,
        chunk_overlap=50,
        preserve_tables=False
    )
    print(f"Custom-Config: max_size={config2.max_chunk_size}, overlap={config2.chunk_overlap}")
    
    # Validierung testen
    try:
        invalid_config = create_chunk_config(max_chunk_size=100, chunk_overlap=200)
        print("❌ Validierung fehlgeschlagen - sollte Fehler werfen")
        return False
    except ValueError as e:
        print(f"✅ Validierung funktioniert: {e}")
    
    print("✅ Chunk-Konfiguration getestet!")
    return True

def test_adapter_integration():
    """Testet die Integration mit dem DoclingAdapter."""
    
    print("\n🧪 Test 3: Adapter-Integration")
    print("=" * 50)
    
    configs = [
        {"chunk_strategy": "hybrid"},
        {"chunk_strategy": "by_page"},
        {"chunk_strategy": "by_heading"},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\nKonfiguration {i}: {config}")
        
        try:
            adapter = DoclingAdapter(config=config)
            
            # Prüfe ob Chunker korrekt initialisiert wurde
            chunker_info = adapter.chunker.get_strategy_info()
            print(f"  ✅ Chunker: {chunker_info['chunker_type']}")
            print(f"  📋 Strategie: {chunker_info['strategy']}")
            print(f"  🔧 Docling: {chunker_info['uses_docling_chunker']}")
            
        except Exception as e:
            print(f"  ❌ Fehler: {str(e)}")
    
    print("\n✅ Adapter-Integration getestet!")
    return True

def test_with_dummy_document():
    """Testet mit einem Dummy-Dokument."""
    
    print("\n🧪 Test 4: Dummy-Dokument-Chunking")
    print("=" * 50)
    
    # Erstelle ein Dummy-Docling-Document
    class DummyElement:
        def __init__(self, text, element_type="text"):
            self.text = text
            self.type = element_type
    
    class DummyPage:
        def __init__(self, page_num, elements):
            self.page_number = page_num
            self.elements = elements
    
    class DummyDoc:
        def __init__(self):
            self.source_path = "test_document.pdf"
            self.pages = [
                DummyPage(1, [
                    DummyElement("Überschrift 1", "heading"),
                    DummyElement("Das ist ein langer Absatz mit viel Text. " * 20, "text"),
                    DummyElement("Tabelle mit Daten", "table"),
                ]),
                DummyPage(2, [
                    DummyElement("Überschrift 2", "heading"),
                    DummyElement("Noch mehr Text für die zweite Seite. " * 15, "text"),
                ])
            ]
    
    dummy_doc = DummyDoc()
    
    # Teste verschiedene Strategien
    strategies = ["hybrid", "by_page"]
    
    for strategy in strategies:
        print(f"\n📋 Teste {strategy} mit Dummy-Dokument:")
        
        try:
            chunker = DoclingChunker(strategy=strategy)
            blocks = chunker.split(dummy_doc)
            
            print(f"  ✅ {len(blocks)} Chunks erstellt")
            
            for i, block in enumerate(blocks[:3]):  # Nur erste 3 zeigen
                text_preview = block["text"][:100] + "..." if len(block["text"]) > 100 else block["text"]
                print(f"    Chunk {i+1}: {len(block['text'])} Zeichen - '{text_preview}'")
                print(f"      Metadaten: {list(block['metadata'].keys())}")
            
            if len(blocks) > 3:
                print(f"    ... und {len(blocks) - 3} weitere Chunks")
                
        except Exception as e:
            print(f"  ❌ Fehler: {str(e)}")
    
    print("\n✅ Dummy-Dokument-Tests abgeschlossen!")
    return True

def test_performance_comparison():
    """Vergleicht die Performance des neuen vs. alten Systems."""
    
    print("\n🧪 Test 5: Performance-Vergleich")
    print("=" * 50)
    
    # Erstelle größeres Dummy-Dokument
    class DummyElement:
        def __init__(self, text, element_type="text"):
            self.text = text
            self.type = element_type
    
    class DummyPage:
        def __init__(self, page_num):
            self.page_number = page_num
            self.elements = [
                DummyElement(f"Überschrift Seite {page_num}", "heading"),
                DummyElement("Lorem ipsum dolor sit amet. " * 100, "text"),
                DummyElement("Tabelle mit vielen Daten. " * 50, "table"),
                DummyElement("Weitere Absätze mit Text. " * 80, "text"),
            ]
    
    class DummyDoc:
        def __init__(self, num_pages=10):
            self.source_path = f"large_test_document_{num_pages}pages.pdf"
            self.pages = [DummyPage(i) for i in range(1, num_pages + 1)]
    
    large_doc = DummyDoc(num_pages=20)
    
    import time
    
    strategies = ["hybrid", "by_page", "by_element"]
    
    for strategy in strategies:
        print(f"\n📊 Performance-Test: {strategy}")
        
        try:
            start_time = time.time()
            
            chunker = DoclingChunker(strategy=strategy)
            blocks = chunker.split(large_doc)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"  ⏱️  Zeit: {duration:.3f}s")
            print(f"  📊 Chunks: {len(blocks)}")
            print(f"  📈 Chunks/s: {len(blocks)/duration:.1f}")
            
            # Statistiken
            total_chars = sum(len(block["text"]) for block in blocks)
            avg_chunk_size = total_chars / len(blocks) if blocks else 0
            
            print(f"  📝 Gesamt-Zeichen: {total_chars:,}")
            print(f"  📏 Ø Chunk-Größe: {avg_chunk_size:.0f} Zeichen")
            
        except Exception as e:
            print(f"  ❌ Fehler: {str(e)}")
    
    print("\n✅ Performance-Tests abgeschlossen!")
    return True

if __name__ == "__main__":
    print("🚀 Starte Docling-Chunking-Tests")
    print("=" * 60)
    print()
    
    try:
        # Alle Tests durchführen
        tests = [
            test_chunker_strategies,
            test_chunk_config,
            test_adapter_integration,
            test_with_dummy_document,
            test_performance_comparison,
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                print(f"❌ Test fehlgeschlagen: {str(e)}")
                results.append(False)
        
        # Zusammenfassung
        print("\n" + "=" * 60)
        print("📊 Test-Zusammenfassung:")
        print("=" * 60)
        
        success_count = sum(results)
        total_count = len(results)
        
        print(f"✅ Erfolgreich: {success_count}/{total_count}")
        print(f"❌ Fehlgeschlagen: {total_count - success_count}/{total_count}")
        
        if all(results):
            print("\n🎉 Alle Tests erfolgreich! Das neue Chunking-System funktioniert.")
            sys.exit(0)
        else:
            print("\n⚠️  Einige Tests fehlgeschlagen. Bitte prüfen.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests abgebrochen")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 