#!/usr/bin/env python3
# test_better_chunking.py - Test einer besseren Chunking-Strategie

import chromadb
from chromadb.utils import embedding_functions
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

def test_better_chunking():
    """
    Teste eine alternative Chunking-Strategie mit größeren, semantisch sinnvolleren Chunks
    """
    print("=== TEST BESSERE CHUNKING-STRATEGIE ===")
    
    # 1. Lade das PDF-Rohmaterial (zum Vergleich)
    print("\n1. Lade PDF-Rohmaterial...")
    from unstructured.partition.pdf import partition_pdf
    
    pdf_path = "./PDFs/monopoly-dm-standard-gebraucht-g046200040.pdf"
    elements = partition_pdf(
        filename=pdf_path,
        strategy="auto",
        include_page_breaks=True,
        combine_text_under_n_chars=0
    )
    
    # 2. Kombiniere Elemente zu semantisch sinnvollen Gruppen
    print("\n2. Kombiniere zu semantisch sinnvollen Gruppen...")
    
    semantic_chunks = []
    current_chunk = ""
    current_section = None
    
    for element in elements:
        element_text = str(element).strip()
        element_type = type(element).__name__
        
        # Erkenne Abschnittswechsel
        is_new_section = element_type == "Title" and len(element_text) > 5
        
        if is_new_section:
            # Speichere den vorherigen Chunk
            if current_chunk.strip():
                semantic_chunks.append({
                    'text': current_chunk.strip(),
                    'section': current_section,
                    'length': len(current_chunk.strip())
                })
            
            # Starte neuen Chunk
            current_section = element_text
            current_chunk = element_text + "\n\n"
        else:
            # Füge zum aktuellen Chunk hinzu
            if element_text:
                current_chunk += element_text + " "
    
    # Letzten Chunk speichern
    if current_chunk.strip():
        semantic_chunks.append({
            'text': current_chunk.strip(),
            'section': current_section,
            'length': len(current_chunk.strip())
        })
    
    # 3. Zeige die semantischen Chunks
    print(f"\n3. Erstellt {len(semantic_chunks)} semantische Chunks:")
    for i, chunk in enumerate(semantic_chunks):
        print(f"\nChunk {i+1}: {chunk['section']}")
        print(f"  Länge: {chunk['length']} Zeichen")
        print(f"  Vorschau: {chunk['text'][:200]}...")
    
    # 4. Teste Embedding-Qualität mit semantischen Chunks
    print("\n4. Teste Embedding-Qualität...")
    
    embedding_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device="cuda"
    )
    
    # Finde den "Vorbereitung"-Chunk
    vorbereitung_chunk = None
    for chunk in semantic_chunks:
        if chunk['section'] and 'vorbereitung' in chunk['section'].lower():
            vorbereitung_chunk = chunk
            break
    
    if vorbereitung_chunk:
        print(f"\n5. Gefunden: Vorbereitung-Chunk mit {vorbereitung_chunk['length']} Zeichen")
        print(f"Vollständiger Text:\n{vorbereitung_chunk['text']}")
        
        # Teste Embedding-Ähnlichkeit
        test_queries = [
            "Wie bereitet man Monopoly vor?",
            "Monopoly Spielvorbereitung",
            "Spiel aufbauen",
            "Bankhalter wählen"
        ]
        
        chunk_embedding = embedding_model.get_text_embedding(vorbereitung_chunk['text'])
        
        print(f"\n6. Embedding-Ähnlichkeitstest:")
        for query in test_queries:
            query_embedding = embedding_model.get_text_embedding(query)
            
            # Berechne Cosine-Ähnlichkeit manuell
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            distance = 1 - similarity  # ChromaDB Distance
            
            print(f"  Query: '{query}'")
            print(f"    Similarity: {similarity:.4f} (höher = besser)")
            print(f"    Distance: {distance:.4f} (niedriger = besser)")
            print()
    else:
        print("❌ Kein Vorbereitung-Chunk gefunden!")

if __name__ == "__main__":
    test_better_chunking() 