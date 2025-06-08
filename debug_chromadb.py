#!/usr/bin/env python3
# debug_chromadb.py - Debug ChromaDB Collection

import chromadb
from chromadb.utils import embedding_functions

def debug_chromadb():
    print("=== CHROMADB DEBUG ===")
    
    try:
        # ChromaDB Client laden
        client = chromadb.PersistentClient(path='./chroma_db_store')
        
        # WICHTIG: Verwende das GLEICHE Embedding-Modell wie in parse_pdf.py und rag_api.py
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
        collection = client.get_collection('test_collection')
        
        print(f"Collection Count: {collection.count()}")
        print(f"Collection Name: {collection.name}")
        
        # Teste verschiedene deutsche Suchbegriffe
        test_queries = [
            "Monopoly vorbereiten",
            "Spiel aufbauen", 
            "Vorbereitung des Spiels",
            "Spielregeln",
            "preparation",
            "game setup"
        ]
        
        for query in test_queries:
            print(f"\n--- SUCHE: '{query}' ---")
            try:
                results = collection.query(
                    query_texts=[query], 
                    n_results=3
                )
                
                if results['documents'] and results['documents'][0]:
                    for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
                        print(f"{i+1}. Score: {dist:.4f}")
                        print(f"   Text: {doc[:150]}...")
                        print()
                else:
                    print("❌ Keine Ergebnisse gefunden")
                    
            except Exception as e:
                print(f"❌ Fehler bei Suche: {str(e)}")
        
        # Hole ein paar zufällige Dokumente aus der Collection
        print("\n--- SAMPLE DOKUMENTE ---")
        try:
            sample_results = collection.get(limit=5)
            if sample_results['documents']:
                for i, doc in enumerate(sample_results['documents'][:3]):
                    print(f"Dokument {i+1}: {doc[:200]}...")
                    print()
            else:
                print("❌ Keine Dokumente in Collection")
        except Exception as e:
            print(f"❌ Fehler beim Abrufen der Sample-Dokumente: {str(e)}")
            
        # Prüfe Embedding-Funktion
        print("\n--- EMBEDDING-FUNKTION TEST ---")
        try:
            # Verwende die KORREKTE Embedding-Funktion für den Test
            embedding_function = sentence_transformer_ef
            print(f"Embedding Function: {type(embedding_function)}")
            
            # Teste ein Sample-Embedding
            test_text = "Monopoly Spielregeln"
            embedding = embedding_function([test_text])
            print(f"Test-Embedding für '{test_text}':")
            print(f"  Dimensionen: {len(embedding[0])}")
            print(f"  Erste 5 Werte: {embedding[0][:5]}")
            
            # DIREKTE ÄHNLICHKEITSSUCHE mit korrektem Embedding
            print(f"\n--- DIREKTE EMBEDDING-ÄHNLICHKEIT TEST ---")
            test_query = "Wie bereitet man Monopoly vor?"
            print(f"Test-Query: '{test_query}'")
            
            # Erzeuge Embedding für die Test-Query
            query_embedding = embedding_function([test_query])
            print(f"Query-Embedding Dimensionen: {len(query_embedding[0])}")
            
            # Führe direkte Ähnlichkeitssuche durch
            direct_results = collection.query(
                query_embeddings=query_embedding,
                n_results=5
            )
            
            print("Top 5 Ergebnisse mit direktem Embedding:")
            if direct_results['documents'] and direct_results['documents'][0]:
                for i, (doc, dist) in enumerate(zip(direct_results['documents'][0], direct_results['distances'][0])):
                    print(f"{i+1}. Score: {dist:.4f}")
                    print(f"   Text: {doc[:150]}...")
                    print()
            
        except Exception as e:
            print(f"❌ Fehler beim Embedding-Test: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Fehler beim Verbinden mit ChromaDB: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chromadb() 