"""
Main Parser für das Parse_Index System

Hauptorchestrierung des PDF-Parsing und Indexierung-Prozesses.
"""

import os
import chromadb
import torch
import traceback
from typing import List

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .config import PDF_FOLDER, PERSIST_DIR, COLLECTION_NAME, EMBED_MODEL_NAME
from .utils import validate_environment
from .node_parsers import create_hybrid_parser_system
from .document_enhancer import semantic_enhanced_pdf
from .unstructured_wrapper import fallback_local_unstructured_pdf

def main():
    """Hauptfunktion für die PDF-Indexierung."""
    print("=== Start der PDF-Indexierung ===")
    
    # Validiere Umgebung
    validation_results = validate_environment()
    if not validation_results["requirements_ok"]:
        print("❌ Abbruch: Nicht alle erforderlichen Bibliotheken verfügbar.")
        return
    
    try:
        # 1. ChromaDB initialisieren
        chroma_client, chroma_collection = setup_chromadb()
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 2. LlamaIndex konfigurieren
        parser_system = setup_llama_index_settings()
        
        # 3. PDFs verarbeiten
        documents = load_and_process_pdfs()
        if not documents:
            print("Keine Dokumente gefunden.")
            return
        
        # 4. Index erstellen
        create_vector_index(documents, storage_context, parser_system)
        
        print("✅ Indexierung erfolgreich abgeschlossen!")
        
    except Exception as e:
        print(f"❌ Fehler: {str(e)}")
        print(traceback.format_exc())

def setup_chromadb():
    """Initialisiert ChromaDB."""
    print("1. Initialisiere ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except:
        pass
    
    from chromadb.utils import embedding_functions
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    chroma_collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    print("✓ ChromaDB Collection erstellt")
    return chroma_client, chroma_collection

def setup_llama_index_settings():
    """Konfiguriert LlamaIndex Settings."""
    print("2. Konfiguriere Embedding Model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_kwargs = {"device": device}
    
    if device == "cuda":
        embedding_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        **embedding_kwargs
    )
    
    parser_system = create_hybrid_parser_system()
    Settings.node_parser = parser_system['main']
    
    print(f"✓ Embedding Model konfiguriert auf {device.upper()}")
    return parser_system

def load_and_process_pdfs():
    """Lädt und verarbeitet alle PDFs im PDF-Ordner."""
    print("3. Lade PDFs...")
    
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"PDF-Ordner '{PDF_FOLDER}' erstellt. Bitte PDFs hinzufügen.")
        return []

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        print("Keine PDF-Dateien gefunden!")
        return []

    documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        try:
            # Verwende semantische Dokumentverbesserung
            file_documents = semantic_enhanced_pdf(pdf_path)
            if not file_documents:
                # Fallback auf normale Unstructured-Verarbeitung
                file_documents = fallback_local_unstructured_pdf(pdf_path)
            
            documents.extend(file_documents)
            print(f"✓ {pdf_file}: {len(file_documents)} Dokumente")
        except Exception as e:
            print(f"❌ Fehler bei {pdf_file}: {str(e)}")
    
    print(f"Gesamt: {len(documents)} Dokumente geladen")
    return documents

def create_vector_index(documents, storage_context, parser_system):
    """Erstellt den Vektor-Index."""
    print("4. Erstelle Index...")
    
    try:
        # Parse zu Nodes
        nodes = parser_system['main'].get_nodes_from_documents(documents, show_progress=True)
        print(f"✓ {len(nodes)} Nodes erstellt")
        
        # Index erstellen
        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
        print(f"✓ Index mit {len(index.docstore.docs)} Nodes erstellt")
        
    except Exception as e:
        print(f"❌ Index-Erstellung fehlgeschlagen: {str(e)}")
        raise

if __name__ == "__main__":
    main() 