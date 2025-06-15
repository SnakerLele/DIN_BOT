#!/usr/bin/env python3
"""
Test-Skript für die neue Docling-Konfiguration
==============================================

Testet, ob die Parameter korrekt an Docling weitergereicht werden.
"""

import sys
from pathlib import Path

# Parse_Index zum Python-Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent))

from Parse_Index.docling_adapter import DoclingAdapter
import logging

# Logging für Tests aktivieren
logging.basicConfig(level=logging.DEBUG)

def test_docling_config():
    """Testet verschiedene Docling-Konfigurationen."""
    
    print("🧪 Test 1: Standard-Konfiguration")
    print("=" * 50)
    
    adapter1 = DoclingAdapter()
    print(f"Docling verfügbar: {adapter1.is_docling_available()}")
    print(f"Konfiguration: {adapter1.get_config()}")
    print()
    
    print("🧪 Test 2: OCR deaktiviert")
    print("=" * 50)
    
    config_no_ocr = {
        "ocr_enabled": False,
        "table_extraction": True,
        "image_extraction": False,
    }
    
    adapter2 = DoclingAdapter(config=config_no_ocr)
    print(f"OCR deaktiviert: {not adapter2.get_config()['ocr_enabled']}")
    print()
    
    print("🧪 Test 3: Nur Tabellen-Extraktion")
    print("=" * 50)
    
    config_tables_only = {
        "ocr_enabled": False,
        "table_extraction": True,
        "image_extraction": False,
        "formula_extraction": False,
        "layout_analysis": True,
    }
    
    adapter3 = DoclingAdapter(config=config_tables_only)
    config = adapter3.get_config()
    print(f"Tabellen: {config['table_extraction']}")
    print(f"OCR: {config['ocr_enabled']}")
    print(f"Bilder: {config['image_extraction']}")
    print()
    
    print("✅ Alle Konfigurations-Tests abgeschlossen!")
    return True

def test_with_sample_pdf():
    """Testet mit einer Beispiel-PDF (falls vorhanden)."""
    
    # Suche nach Test-PDFs
    test_pdf_paths = [
        Path("test_documents/sample.pdf"),
        Path("../test_documents/sample.pdf"),
        Path("sample.pdf"),
    ]
    
    test_pdf = None
    for pdf_path in test_pdf_paths:
        if pdf_path.exists():
            test_pdf = pdf_path
            break
    
    if not test_pdf:
        print("⚠️  Keine Test-PDF gefunden. Überspringe PDF-Test.")
        return True
    
    print(f"🧪 Test 4: PDF-Verarbeitung mit {test_pdf.name}")
    print("=" * 50)
    
    try:
        # Test mit verschiedenen Konfigurationen
        configs = [
            {"ocr_enabled": True, "table_extraction": True},
            {"ocr_enabled": False, "table_extraction": True},
            {"ocr_enabled": True, "table_extraction": False},
        ]
        
        for i, config in enumerate(configs, 1):
            print(f"Konfiguration {i}: {config}")
            
            adapter = DoclingAdapter(config=config)
            
            # Nur parsen, nicht vollständig verarbeiten (für schnellere Tests)
            try:
                documents = adapter.parse_pdf(test_pdf, strict_mode=False)
                stats = adapter.get_last_processing_stats()
                
                print(f"  ✅ {len(documents)} Documents erstellt")
                print(f"  📊 Seiten: {stats.get('pages', 0)}")
                print(f"  📊 Blöcke: {stats.get('blocks', 0)}")
                print(f"  📊 Tabellen: {stats.get('tables', 0)}")
                print()
                
            except Exception as e:
                print(f"  ❌ Fehler: {str(e)}")
                print()
        
        print("✅ PDF-Tests abgeschlossen!")
        return True
        
    except Exception as e:
        print(f"❌ PDF-Test fehlgeschlagen: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starte Docling-Konfigurations-Tests")
    print("=" * 60)
    print()
    
    try:
        # Basis-Tests
        success1 = test_docling_config()
        
        # PDF-Tests (optional)
        success2 = test_with_sample_pdf()
        
        if success1 and success2:
            print("🎉 Alle Tests erfolgreich!")
            sys.exit(0)
        else:
            print("❌ Einige Tests fehlgeschlagen!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests abgebrochen")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 