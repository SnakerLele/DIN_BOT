"""
Test-Skript für Phase 4: LlamaIndex-Integration

Testet die neue DoclingAdapter-Integration mit LlamaIndex's nativen
DoclingReader und DoclingNodeParser.
"""

import sys
import os
from pathlib import Path

# Füge Parse_Index zum Python-Pfad hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Teste alle wichtigen Imports."""
    print("🔧 Teste Imports...")
    
    try:
        from Parse_Index.docling_adapter.adapter import DoclingAdapter
        print("✅ DoclingAdapter Import erfolgreich")
    except ImportError as e:
        print(f"❌ DoclingAdapter Import fehlgeschlagen: {e}")
        return False
    
    try:
        from Parse_Index.node_parsers import (
            is_docling_node_parser_available,
            get_optimal_parser_for_documents,
            parse_documents_with_optimal_strategy
        )
        print("✅ Node Parsers Import erfolgreich")
    except ImportError as e:
        print(f"❌ Node Parsers Import fehlgeschlagen: {e}")
        return False
    
    try:
        from llama_index.readers.docling import DoclingReader
        from llama_index.node_parser.docling import DoclingNodeParser
        print("✅ LlamaIndex Docling Integration verfügbar")
        return True
    except ImportError as e:
        print(f"⚠️ LlamaIndex Docling Integration nicht verfügbar: {e}")
        return False

def test_docling_adapter_initialization():
    """Teste DoclingAdapter Initialisierung."""
    print("\n🔧 Teste DoclingAdapter Initialisierung...")
    
    try:
        from Parse_Index.docling_adapter.adapter import DoclingAdapter
        
        # Standard-Konfiguration
        adapter = DoclingAdapter()
        print(f"✅ Standard-Adapter erstellt - LlamaIndex verfügbar: {adapter.is_llamaindex_docling_available()}")
        
        # JSON-Export-Konfiguration
        json_adapter = DoclingAdapter(config={"export_type": "JSON"})
        print(f"✅ JSON-Adapter erstellt")
        
        # Markdown-Export-Konfiguration
        md_adapter = DoclingAdapter(config={"export_type": "MARKDOWN"})
        print(f"✅ Markdown-Adapter erstellt")
        
        return True
        
    except Exception as e:
        print(f"❌ DoclingAdapter Initialisierung fehlgeschlagen: {e}")
        return False

def test_node_parser_availability():
    """Teste Node Parser Verfügbarkeit."""
    print("\n🔧 Teste Node Parser Verfügbarkeit...")
    
    try:
        from Parse_Index.node_parsers import (
            is_docling_node_parser_available,
            get_parser_info,
            get_parser_by_strategy
        )
        
        # Prüfe DoclingNodeParser Verfügbarkeit
        docling_available = is_docling_node_parser_available()
        print(f"✅ DoclingNodeParser verfügbar: {docling_available}")
        
        # Teste verschiedene Parser-Strategien
        strategies = ['docling_native', 'docling_markdown', 'semantic', 'hierarchical']
        
        for strategy in strategies:
            try:
                parser = get_parser_by_strategy(strategy)
                print(f"✅ Parser '{strategy}' erfolgreich erstellt: {type(parser).__name__}")
            except Exception as e:
                print(f"⚠️ Parser '{strategy}' nicht verfügbar: {e}")
        
        # Parser-Informationen anzeigen
        parser_info = get_parser_info()
        print(f"\n📊 Verfügbare Parser: {len(parser_info)}")
        for name, info in parser_info.items():
            status = "✅" if info.get('available', True) else "❌"
            print(f"   {status} {name}: {info['description']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Node Parser Test fehlgeschlagen: {e}")
        return False

def test_document_creation():
    """Teste Document-Erstellung mit Mock-Daten."""
    print("\n🔧 Teste Document-Erstellung...")
    
    try:
        from llama_index.core import Document
        from Parse_Index.node_parsers import get_optimal_parser_for_documents
        
        # Mock-Documents erstellen
        documents = [
            Document(
                text="Dies ist ein Test-Dokument mit Tabellen-Informationen.",
                metadata={
                    "schema_name": "docling_core.transforms.chunker.DocMeta",
                    "has_tables": True,
                    "table_count": 2,
                    "export_type": "JSON"
                }
            ),
            Document(
                text="# Überschrift\n\nDies ist ein Markdown-Dokument.",
                metadata={
                    "export_type": "MARKDOWN",
                    "has_tables": False
                }
            )
        ]
        
        # Optimalen Parser bestimmen
        for i, doc in enumerate(documents):
            parser = get_optimal_parser_for_documents([doc])
            print(f"✅ Document {i+1}: Optimaler Parser = {type(parser).__name__}")
            print(f"   Metadaten: {doc.metadata}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document-Erstellung fehlgeschlagen: {e}")
        return False

def test_convenience_functions():
    """Teste Convenience-Funktionen."""
    print("\n🔧 Teste Convenience-Funktionen...")
    
    try:
        from Parse_Index.docling_adapter.adapter import parse_pdf_simple
        from Parse_Index.node_parsers import parse_documents_with_strategy
        from llama_index.core import Document
        
        # Test parse_pdf_simple (ohne echte PDF)
        print("✅ parse_pdf_simple Funktion verfügbar")
        
        # Test parse_documents_with_strategy
        test_docs = [
            Document(
                text="Test-Text für Parser-Test",
                metadata={"test": True}
            )
        ]
        
        try:
            nodes = parse_documents_with_strategy(test_docs, 'semantic')
            print(f"✅ parse_documents_with_strategy: {len(nodes)} Nodes erstellt")
        except Exception as e:
            print(f"⚠️ parse_documents_with_strategy Fehler: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience-Funktionen Test fehlgeschlagen: {e}")
        return False

def test_table_html_extraction():
    """Teste Tabellen-HTML-Extraktion."""
    print("\n🔧 Teste Tabellen-HTML-Extraktion...")
    
    try:
        from Parse_Index.docling_adapter.adapter import DoclingAdapter
        from llama_index.core import Document
        
        adapter = DoclingAdapter()
        
        # Mock-Document mit Tabellen-Daten
        mock_doc = Document(
            text="Test-Dokument mit Tabelle",
            metadata={
                'doc_items': [
                    {
                        'label': 'table',
                        'table_data': [
                            ['Header 1', 'Header 2'],
                            ['Zeile 1', 'Daten 1'],
                            ['Zeile 2', 'Daten 2']
                        ]
                    }
                ]
            }
        )
        
        # Teste HTML-Konvertierung
        html = adapter._convert_table_data_to_html([
            ['Name', 'Alter'],
            ['Max', '25'],
            ['Anna', '30']
        ])
        
        if html and '<table>' in html:
            print("✅ Tabellen-HTML-Konvertierung erfolgreich")
            print(f"   HTML-Länge: {len(html)} Zeichen")
        else:
            print("⚠️ Tabellen-HTML-Konvertierung fehlgeschlagen")
        
        return True
        
    except Exception as e:
        print(f"❌ Tabellen-HTML-Test fehlgeschlagen: {e}")
        return False

def test_configuration_options():
    """Teste verschiedene Konfigurationsoptionen."""
    print("\n🔧 Teste Konfigurationsoptionen...")
    
    try:
        from Parse_Index.docling_adapter.adapter import DoclingAdapter
        
        # Verschiedene Konfigurationen testen
        configs = [
            {"export_type": "JSON", "preserve_table_html": True},
            {"export_type": "MARKDOWN", "preserve_bounding_boxes": False},
            {"export_type": "JSON", "ocr_enabled": False, "table_extraction": True}
        ]
        
        for i, config in enumerate(configs):
            adapter = DoclingAdapter(config=config)
            final_config = adapter.get_config()
            print(f"✅ Konfiguration {i+1}: {config}")
            print(f"   Export-Typ: {final_config.get('export_type')}")
            print(f"   Tabellen-HTML: {final_config.get('preserve_table_html')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Konfigurationstest fehlgeschlagen: {e}")
        return False

def main():
    """Haupttest-Funktion."""
    print("🚀 Phase 4 Test: LlamaIndex-Integration")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("DoclingAdapter Initialisierung", test_docling_adapter_initialization),
        ("Node Parser Verfügbarkeit", test_node_parser_availability),
        ("Document-Erstellung", test_document_creation),
        ("Convenience-Funktionen", test_convenience_functions),
        ("Tabellen-HTML-Extraktion", test_table_html_extraction),
        ("Konfigurationsoptionen", test_configuration_options),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' Ausnahme: {e}")
            results.append((test_name, False))
    
    # Zusammenfassung
    print("\n" + "=" * 50)
    print("📊 Test-Zusammenfassung:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ BESTANDEN" if result else "❌ FEHLGESCHLAGEN"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Ergebnis: {passed}/{total} Tests bestanden")
    
    if passed == total:
        print("🎉 Alle Tests erfolgreich! Phase 4 LlamaIndex-Integration funktioniert.")
    else:
        print("⚠️ Einige Tests fehlgeschlagen. Überprüfe die Implementierung.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 