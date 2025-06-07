#!/usr/bin/env python3
# Test-Skript für UnstructuredReader mit korrekter Strategy-Verwendung

import os
from llama_index.readers.file import UnstructuredReader
from pathlib import Path

# Prüfe verfügbare PDFs
PDF_FOLDER = "./PDFs"
if os.path.exists(PDF_FOLDER):
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    print(f"Gefundene PDFs: {pdf_files}")
    
    if pdf_files:
        test_file = os.path.join(PDF_FOLDER, pdf_files[0])
        print(f"Teste mit: {test_file}")
        
        # Erstelle UnstructuredReader
        reader = UnstructuredReader()
        
        # Test 1: Standard-Aufruf (Referenz)
        print("\n=== Test 1: Standard-Aufruf (Referenz) ===")
        try:
            docs_standard = reader.load_data(file=test_file)
            print(f"✓ Standard: {len(docs_standard)} Dokumente")
            if docs_standard:
                print(f"  - Text-Länge: {len(docs_standard[0].text)} Zeichen")
                print(f"  - Seitennummer: {docs_standard[0].metadata.get('page_number', 'N/A')}")
        except Exception as e:
            print(f"❌ Standard-Aufruf fehlgeschlagen: {e}")
            
        # Test 2: Mit strategy über unstructured_kwargs
        print("\n=== Test 2: Mit strategy='auto' über unstructured_kwargs ===")
        try:
            docs_auto = reader.load_data(
                file=test_file, 
                unstructured_kwargs={"strategy": "auto"}
            )
            print(f"✓ Strategy='auto': {len(docs_auto)} Dokumente")
            if docs_auto:
                print(f"  - Text-Länge: {len(docs_auto[0].text)} Zeichen")
                print(f"  - Seitennummer: {docs_auto[0].metadata.get('page_number', 'N/A')}")
        except Exception as e:
            print(f"❌ Strategy='auto' fehlgeschlagen: {e}")
            
        # Test 3: Mit strategy='hi_res' für bessere Qualität
        print("\n=== Test 3: Mit strategy='hi_res' ===")
        try:
            docs_hires = reader.load_data(
                file=test_file, 
                unstructured_kwargs={
                    "strategy": "hi_res",
                    "include_page_breaks": True
                }
            )
            print(f"✓ Strategy='hi_res': {len(docs_hires)} Dokumente")
            if docs_hires:
                print(f"  - Text-Länge: {len(docs_hires[0].text)} Zeichen")
                print(f"  - Seitennummer: {docs_hires[0].metadata.get('page_number', 'N/A')}")
        except Exception as e:
            print(f"❌ Strategy='hi_res' fehlgeschlagen: {e}")
            
        # Test 4: Mit strategy='fast' für schnelle Verarbeitung
        print("\n=== Test 4: Mit strategy='fast' ===")
        try:
            docs_fast = reader.load_data(
                file=test_file, 
                unstructured_kwargs={
                    "strategy": "fast",
                    "include_page_breaks": True
                }
            )
            print(f"✓ Strategy='fast': {len(docs_fast)} Dokumente")
            if docs_fast:
                print(f"  - Text-Länge: {len(docs_fast[0].text)} Zeichen")
                print(f"  - Seitennummer: {docs_fast[0].metadata.get('page_number', 'N/A')}")
        except Exception as e:
            print(f"❌ Strategy='fast' fehlgeschlagen: {e}")
            
        print("\n=== Zusammenfassung ===")
        print("Der strategy Parameter muss über 'unstructured_kwargs' übergeben werden!")
        print("Beispiel: reader.load_data(file=path, unstructured_kwargs={'strategy': 'auto'})")
            
    else:
        print("Keine PDF-Dateien im PDFs-Ordner gefunden.")
else:
    print("PDFs-Ordner nicht gefunden.") 