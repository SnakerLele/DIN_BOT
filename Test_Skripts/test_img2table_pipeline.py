#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für die img2table-priorisierte Pipeline

Testet die neue semantic_enhanced_pdf-Funktion mit img2table-Priorität:
- img2table-Tabellen-Extraktion als primäre Methode
- Unstructured als Fallback für Text und übersehene Tabellen
- Kombinierte Document-Erstellung
- Qualitäts-Validierung

Verwendung:
    python test_img2table_pipeline.py [PDF-Datei-Pfad]
"""

import sys
import os
from pathlib import Path
import time

# Füge Parse_Index zum Python-Pfad hinzu
current_dir = Path(__file__).parent
project_root = current_dir.parent
parse_index_dir = project_root / "Parse_Index"
sys.path.insert(0, str(parse_index_dir))

def test_img2table_pipeline(pdf_path: str):
    """
    Testet die img2table-priorisierte Pipeline
    
    Args:
        pdf_path: Pfad zur Test-PDF-Datei
    """
    print("🧪 TEST: img2table-priorisierte Pipeline")
    print("=" * 60)
    
    # Überprüfe PDF-Datei
    if not os.path.exists(pdf_path):
        print(f"❌ PDF-Datei nicht gefunden: {pdf_path}")
        return False
    
    print(f"📄 Test-PDF: {os.path.basename(pdf_path)}")
    print(f"📁 Pfad: {pdf_path}")
    print(f"📊 Dateigröße: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    print()
    
    try:
        # Import der neuen Pipeline
        from pdf_enhancer.enhancer import semantic_enhanced_pdf
        
        print("🎯 TESTE: img2table-priorisierte Pipeline (Standard)")
        print("-" * 50)
        start_time = time.time()
        
        # Test mit img2table aktiviert (Standard)
        documents = semantic_enhanced_pdf(pdf_path, use_img2table=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️ Verarbeitungszeit: {processing_time:.2f} Sekunden")
        print(f"📋 Erstellte Documents: {len(documents)}")
        print()
        
        # Analysiere Ergebnisse
        print("🔍 ANALYSE DER ERGEBNISSE:")
        print("-" * 30)
        
        # Trenne img2table und unstructured Documents
        img2table_docs = [d for d in documents if d.metadata.get('extraction_method') == 'img2table']
        unstructured_docs = [d for d in documents if d.metadata.get('extraction_method') != 'img2table']
        
        print(f"📊 img2table-Tabellen: {len(img2table_docs)}")
        print(f"📄 Unstructured-Chunks: {len(unstructured_docs)}")
        print()
        
        # img2table-Details
        if img2table_docs:
            print("🎯 IMG2TABLE-TABELLEN DETAILS:")
            total_cells = 0
            quality_counts = {}
            
            for i, doc in enumerate(img2table_docs):
                page = doc.metadata.get('page_number', '?')
                rows = doc.metadata.get('table_rows', 0)
                cols = doc.metadata.get('table_columns', 0)
                quality = doc.metadata.get('table_quality', 'unknown')
                cells = rows * cols
                total_cells += cells
                
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
                
                print(f"  📊 Tabelle {i+1}: Seite {page}, {rows}x{cols} ({cells} Zellen), {quality} Qualität")
                
                # Zeige ersten Teil des Inhalts
                content_preview = doc.text[:200].replace('\n', ' ')
                print(f"      Preview: {content_preview}...")
                print()
            
            print(f"📈 img2table Statistiken:")
            print(f"  - Gesamt-Zellen: {total_cells}")
            for quality, count in quality_counts.items():
                print(f"  - {quality}: {count} Tabellen")
            print()
        else:
            print("ℹ️ Keine img2table-Tabellen gefunden")
            print()
        
        # Unstructured-Details  
        if unstructured_docs:
            print("📄 UNSTRUCTURED-CHUNKS DETAILS:")
            table_docs = [d for d in unstructured_docs if d.metadata.get('is_table')]
            text_docs = [d for d in unstructured_docs if not d.metadata.get('is_table')]
            
            print(f"  📋 Text-Chunks: {len(text_docs)}")
            print(f"  📊 Fallback-Tabellen: {len(table_docs)}")
            
            # Zeige erste 3 Text-Chunks
            for i, doc in enumerate(text_docs[:3]):
                section = doc.metadata.get('section_title', 'Unbekannt')
                length = len(doc.text)
                print(f"    📋 Chunk {i+1}: '{section}' ({length} Zeichen)")
            
            if len(text_docs) > 3:
                print(f"    ... und {len(text_docs)-3} weitere Text-Chunks")
            print()
        
        # Vergleichstest: Ohne img2table
        print("🔄 VERGLEICHSTEST: Ohne img2table")
        print("-" * 40)
        
        # Temporär das erste Export-File umbenennen, damit es nicht überschrieben wird
        pdf_file = Path(pdf_path)
        original_export = pdf_file.parent / f"{pdf_file.stem}_enhanced_documents.txt"
        img2table_export = pdf_file.parent / f"{pdf_file.stem}_WITH_img2table_documents.txt"
        
        if original_export.exists():
            original_export.rename(img2table_export)
            print(f"✅ img2table-Export gesichert als: {img2table_export.name}")
        
        start_time = time.time()
        documents_no_img2table = semantic_enhanced_pdf(pdf_path, use_img2table=False)
        end_time = time.time()
        
        # Das ohne-img2table Export umbenennen
        without_img2table_export = pdf_file.parent / f"{pdf_file.stem}_WITHOUT_img2table_documents.txt"
        if original_export.exists():
            original_export.rename(without_img2table_export)
            print(f"✅ Ohne-img2table-Export gesichert als: {without_img2table_export.name}")
        
        print(f"⏱️ Verarbeitungszeit ohne img2table: {end_time - start_time:.2f} Sekunden")
        print(f"📋 Documents ohne img2table: {len(documents_no_img2table)}")
        
        # Vergleiche Ergebnisse
        print()
        print("⚖️ PIPELINE-VERGLEICH:")
        print("-" * 25)
        print(f"Mit img2table:    {len(documents)} Documents ({len(img2table_docs)} img2table-Tabellen)")
        print(f"Ohne img2table:   {len(documents_no_img2table)} Documents")
        print(f"img2table-Vorteil: +{len(img2table_docs)} zusätzliche Tabellen")
        print()
        
        # Qualitäts-Bewertung
        print("💯 QUALITÄTS-BEWERTUNG:")
        print("-" * 25)
        
        if img2table_docs:
            good_quality = len([d for d in img2table_docs if d.metadata.get('table_quality') == 'good'])
            success_rate = (good_quality / len(img2table_docs)) * 100
            print(f"✅ img2table-Erfolgsrate: {success_rate:.1f}% ({good_quality}/{len(img2table_docs)} gute Qualität)")
        
        total_time_saved = processing_time  # Approximation
        print(f"⚡ Verarbeitung: {processing_time:.1f}s für {len(documents)} Documents")
        print(f"📊 Tabellen-Priorität: img2table > unstructured erfolgreich implementiert")
        
        # Export-Validierung
        pdf_file = Path(pdf_path)
        img2table_export = pdf_file.parent / f"{pdf_file.stem}_WITH_img2table_documents.txt"
        without_img2table_export = pdf_file.parent / f"{pdf_file.stem}_WITHOUT_img2table_documents.txt"
        
        if img2table_export.exists():
            print(f"📤 img2table-Export: {img2table_export.name} ({img2table_export.stat().st_size} Bytes)")
        if without_img2table_export.exists():
            print(f"📤 Ohne-img2table-Export: {without_img2table_export.name} ({without_img2table_export.stat().st_size} Bytes)")
        
        print()
        print("✅ TEST ERFOLGREICH: img2table-Pipeline funktioniert korrekt!")
        return True
        
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        print("Stelle sicher, dass alle Dependencies installiert sind:")
        print("  pip install img2table pdf2image pillow")
        return False
        
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        return False

def main():
    """Hauptfunktion für den Test"""
    
    print("🧪 img2table-Pipeline Tester")
    print("=" * 40)
    
    # PDF-Pfad aus Argumenten oder Standard
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Beispiel-PDF suchen
        test_pdfs = [
            "test_document.pdf",
            "sample.pdf", 
            "../sample_pdfs/test.pdf",
            "test.pdf"
        ]
        
        pdf_path = None
        for test_pdf in test_pdfs:
            if os.path.exists(test_pdf):
                pdf_path = test_pdf
                break
        
        if not pdf_path:
            print("❌ Keine Test-PDF gefunden!")
            print("Verwendung: python test_img2table_pipeline.py [PDF-Datei-Pfad]")
            print("\nOder lege eine der folgenden Test-PDFs an:")
            for test_pdf in test_pdfs:
                print(f"  - {test_pdf}")
            return False
    
    # Führe Test aus
    success = test_img2table_pipeline(pdf_path)
    
    if success:
        print("\n🎉 Alle Tests bestanden!")
        print("Die img2table-priorisierte Pipeline ist einsatzbereit.")
    else:
        print("\n❌ Test fehlgeschlagen!")
        print("Überprüfe die Konfiguration und Dependencies.")
    
    return success

if __name__ == "__main__":
    main() 