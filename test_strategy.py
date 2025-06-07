#!/usr/bin/env python3
# Test-Skript für UnstructuredReader Strategy Parameter

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
        
        # Test 1: Prüfe, welche Parameter load_data akzeptiert
        print("\n=== Test 1: Signatur der load_data Methode ===")
        import inspect
        sig = inspect.signature(reader.load_data)
        print(f"load_data Parameter: {list(sig.parameters.keys())}")
        
        # Test 2: Versuche mit strategy Parameter
        print("\n=== Test 2: Test mit strategy='auto' ===")
        try:
            docs = reader.load_data(file=test_file, strategy="auto")
            print(f"✓ Strategy Parameter funktioniert! Dokumente erhalten: {len(docs)}")
        except TypeError as e:
            print(f"❌ Strategy Parameter nicht unterstützt: {e}")
            
        # Test 3: Versuche mit anderen Parametern
        print("\n=== Test 3: Test mit anderen möglichen Parametern ===")
        try:
            docs = reader.load_data(file=test_file, hi_res=True)
            print(f"✓ hi_res Parameter funktioniert! Dokumente: {len(docs)}")
        except Exception as e:
            print(f"❌ hi_res Parameter nicht unterstützt: {e}")
            
        # Test 4: Standard-Aufruf ohne zusätzliche Parameter
        print("\n=== Test 4: Standard-Aufruf ===")
        try:
            docs = reader.load_data(file=test_file)
            print(f"✓ Standard-Aufruf funktioniert! Dokumente: {len(docs)}")
            if docs:
                print(f"Erstes Dokument - Länge: {len(docs[0].text)} Zeichen")
                print(f"Metadaten: {docs[0].metadata}")
        except Exception as e:
            print(f"❌ Standard-Aufruf fehlgeschlagen: {e}")
            
        # Test 5: Prüfe verfügbare Attribute des Readers
        print("\n=== Test 5: Verfügbare Reader-Attribute ===")
        attributes = [attr for attr in dir(reader) if not attr.startswith('_')]
        print(f"Verfügbare Attribute: {attributes}")
        
    else:
        print("Keine PDF-Dateien im PDFs-Ordner gefunden.")
else:
    print("PDFs-Ordner nicht gefunden.") 