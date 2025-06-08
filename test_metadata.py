#!/usr/bin/env python3
# test_metadata.py - Inspiziere Header-Metadaten in ChromaDB

import chromadb
from chromadb.utils import embedding_functions

def inspect_metadata():
    print("=== METADATEN-INSPEKTION ===")
    
    # ChromaDB laden mit korrekter Embedding-Funktion
    client = chromadb.PersistentClient(path='./chroma_db_store')
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    
    collection = client.get_collection('test_collection', embedding_function=sentence_transformer_ef)
    
    print(f"Collection Count: {collection.count()}")
    
    # Hole alle Dokumente mit Metadaten
    all_data = collection.get(include=['documents', 'metadatas'])
    
    print(f"\n=== ALLE VERFÜGBAREN METADATEN ===")
    
    for i, (doc, metadata) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
        print(f"\n--- DOKUMENT {i+1} ---")
        print(f"Text (Vorschau): {doc[:100]}...")
        print(f"Metadaten:")
        
        if metadata:
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        else:
            print("  Keine Metadaten")
        
        # Prüfe speziell auf Header-Metadaten
        header_keys = ['section_h1', 'section_h2', 'section_h3', 'current_section', 'contains_headers', 'first_header']
        found_headers = {k: v for k, v in (metadata or {}).items() if k in header_keys}
        
        if found_headers:
            print(f"  🎯 HEADER-METADATEN GEFUNDEN:")
            for key, value in found_headers.items():
                print(f"    {key}: {value}")
        else:
            print(f"  ❌ Keine Header-Metadaten gefunden")
            
        print("-" * 60)

if __name__ == "__main__":
    inspect_metadata() 