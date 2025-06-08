# 1. Imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.settings import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
# Qwen3-Reranker Integration
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
# Alternative: SentenceTransformerRerank falls LLMRerank nicht funktioniert
try:
    from llama_index.core.postprocessor import SentenceTransformerRerank
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False
    print("WARNUNG: SentenceTransformerRerank nicht verfügbar - Fallback auf LLMRerank")

from typing import List, Optional
import time
import logging
import sys
from typing import List, Optional
import uuid
import json
from datetime import datetime
import os
import torch
# (Weitere Imports ggf. nötig)

# 2. Logging einrichten
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_api.log')
    ]
)
logger = logging.getLogger(__name__)

# RAG-spezifisches Logging für raglog.txt
def reset_rag_log():
    """Setzt die raglog.txt für eine neue Anfrage zurück"""
    try:
        with open('raglog.txt', 'w', encoding='utf-8') as f:
            f.write("")  # Datei leeren
    except Exception as e:
        logger.error(f"Fehler beim Zurücksetzen der raglog.txt: {str(e)}")

def setup_rag_logger():
    """Erstellt einen speziellen Logger für raglog.txt"""
    rag_logger = logging.getLogger('rag_request')
    rag_logger.setLevel(logging.DEBUG)
    
    # Entferne alle bestehenden Handler
    for handler in rag_logger.handlers[:]:
        rag_logger.removeHandler(handler)
    
    # Erstelle neuen FileHandler für raglog.txt
    rag_handler = logging.FileHandler('raglog.txt', mode='a', encoding='utf-8')
    rag_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    rag_logger.addHandler(rag_handler)
    rag_logger.propagate = False  # Verhindert doppelte Ausgabe
    
    return rag_logger

# 3. Qwen3-Reranker Konfiguration
def setup_qwen3_reranker(device: str = "auto"):
    """
    Konfiguriert den Qwen3-Reranker für optimale Performance mit CPU-Fallback.
    
    Args:
        device: "auto", "cuda" oder "cpu"
    
    Returns:
        Konfigurierter Reranker-Postprocessor
    """
    if device == "auto":
        # Prüfe CUDA-Verfügbarkeit und Memory
        if torch.cuda.is_available():
            # Prüfe verfügbares GPU-Memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.debug(f"Verfügbares GPU-Memory: {gpu_memory:.1f} GB")
            
            # 4B Modell benötigt mindestens 8GB GPU-Memory
            if gpu_memory >= 8.0:
                device = "cuda"
                logger.debug("Verwende CUDA für Qwen3-Reranker (genug GPU-Memory)")
            else:
                device = "cpu"
                logger.warning(f"GPU hat nur {gpu_memory:.1f} GB - verwende CPU für Reranker")
        else:
            device = "cpu"
            logger.debug("CUDA nicht verfügbar - verwende CPU")
    
    logger.debug(f"Konfiguriere Qwen3-Reranker auf {device.upper()}...")
    
    try:
        # Option 1: SentenceTransformerRerank mit Qwen3-Reranker (bevorzugt)
        if RERANK_AVAILABLE:
            # Zuerst versuchen wir das 4B Modell
            try:
                reranker = SentenceTransformerRerank(
                    model="Qwen/Qwen3-Reranker-4B",
                    device=device,
                    top_n=10  # Finale Top-10 nach Reranking
                )
                logger.debug("Qwen3-Reranker-4B erfolgreich geladen")
                logger.debug("  - Cross-Encoder Architecture für präzise Relevanz-Bewertung")
                logger.debug("  - 4B Parameter für optimale Balance")
                logger.debug("  - Multilinguale Unterstützung")
                logger.debug(f"  - Device: {device}")
                return reranker
            except Exception as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    logger.warning(f"4B Modell zu groß für {device} - versuche CPU...")
                    if device == "cuda":
                        # Fallback auf CPU
                        try:
                            reranker = SentenceTransformerRerank(
                                model="Qwen/Qwen3-Reranker-4B",
                                device="cpu",
                                top_n=10
                            )
                            logger.debug("Qwen3-Reranker-4B erfolgreich auf CPU geladen")
                            logger.debug("  - Device: CPU (GPU-Memory nicht ausreichend)")
                            return reranker
                        except Exception as cpu_e:
                            logger.warning(f"Auch CPU-Fallback fehlgeschlagen: {cpu_e}")
                            raise e  # Original-Fehler weiterwerfen
                    else:
                        raise e
                else:
                    raise e
            
    except Exception as e:
        logger.warning(f"Fehler bei SentenceTransformerRerank: {str(e)}")
        logger.warning("Fallback auf LLMRerank...")
    
    # Fallback: LLMRerank (weniger optimal, aber funktional)
    try:
        # Verwende das bereits geladene LLM für Reranking
        # Das ist nicht optimal, aber ein Fallback
        reranker = LLMRerank(
            llm=llm,  # Wird später definiert
            top_n=10
        )
        logger.debug("LLMRerank als Fallback konfiguriert")
        logger.warning("  - Nutzt allgemeines LLM statt speziellem Reranker")
        logger.warning("  - Performance und Qualität sind suboptimal")
        return reranker
        
    except Exception as e:
        logger.error(f"Auch LLMRerank fehlgeschlagen: {str(e)}")
        logger.error("Kein Reranking verfügbar - verwende nur Embedding-Similarity")
        return None

def create_reranking_retriever(index, reranker=None):
    """
    Erstellt einen zweistufigen Retriever:
    1. Dual-Encoder für breite Vorauswahl (100 Kandidaten)  
    2. Cross-Encoder Reranker für finale Auswahl (10 beste)
    
    Args:
        index: VectorStoreIndex
        reranker: Reranker-Postprocessor (optional)
    
    Returns:
        Konfigurierter Retriever
    """
    logger.debug("Erstelle zweistufigen Reranking-Retriever...")
    
    # Basis-Retriever für Dual-Encoder Vorauswahl
    base_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=100,  # Breite Vorauswahl für Reranking
    )
    
    if reranker is not None:
        logger.debug("Reranking-Retriever mit Qwen3-Reranker erstellt")
        logger.debug("  - Stufe 1: Dual-Encoder Vorauswahl (Top-100)")
        logger.debug("  - Stufe 2: Cross-Encoder Reranking (Top-10)")
        logger.debug("  - Erwartete Verbesserung: +20-30% Relevanz-Praezision")
        
        # Retriever mit Postprocessor (Reranker)
        base_retriever._node_postprocessors = [reranker]
        return base_retriever
    else:
        logger.warning("Kein Reranker verfügbar - reduziere similarity_top_k auf 10")
        base_retriever._similarity_top_k = 10  # Fallback ohne Reranking
        return base_retriever

# 4. Initialisierung (wird nur beim Serverstart ausgeführt)
logger.debug("Starte Initialisierung der API-Komponenten...")
try:
    # ChromaDB initialisieren
    logger.debug("Initialisiere ChromaDB...")
    db_client = chromadb.PersistentClient(path="./chroma_db_store")
    
    # Versuche die Collection zu laden oder erstelle sie
    COLLECTION_NAME = "test_collection"  # Muss mit index_data.py übereinstimmen
    try:
        chroma_collection = db_client.get_collection(COLLECTION_NAME)
        logger.debug(f"Bestehende Collection gefunden: {chroma_collection.count()} Dokumente")
    except Exception as e:
        logger.warning(f"Collection nicht gefunden, erstelle neue: {str(e)}")
        chroma_collection = db_client.create_collection(COLLECTION_NAME)
        logger.debug("Neue Collection erstellt")
    
    # Embedding-Modell initialisieren
    logger.debug("Lade Embedding-Modell...")
    # Upgrade auf Qwen3-Embedding für bessere Performance und Kompatibilität
    embed_model = HuggingFaceEmbedding(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        tokenizer_kwargs={"padding_side": "left"}
    )
    logger.debug("Qwen3-Embedding-0.6B erfolgreich geladen")
    logger.debug("  - 1024 Embedding-Dimensionen")
    logger.debug("  - Multilinguale Unterstuetzung (100+ Sprachen)")
    logger.debug("  - 32k Token Kontext")
    
    # LLM initialisieren
    logger.debug("Verbinde mit Ollama...")
    llm = Ollama(model="llama3.2:1b", base_url="http://localhost:11434", request_timeout=120.0)
    # Test Ollama Verbindung mit robusterer Fehlerbehandlung
    try:
        logger.debug("Teste Ollama-Verbindung...")
        test_response = llm.complete("Test")
        logger.debug("Ollama-Verbindung erfolgreich")
    except Exception as e:
        error_str = str(e).lower()
        logger.error(f"Ollama-Verbindungstest fehlgeschlagen: {str(e)}")
        
        # Prüfe verfügbare Modelle für bessere Diagnose
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                logger.info(f"Verfügbare Modelle: {available_models}")
                
                if 'llama3.2:1b' in available_models:
                    logger.info("llama3.2:1b ist installiert - Fehler könnte temporär sein")
                    if "runner process has terminated" in error_str:
                        logger.warning("Model runner crashed - könnte Speicher-Problem sein")
                        logger.warning("Versuche Ollama neu zu starten: ollama serve")
                else:
                    logger.error("PROBLEM: llama3.2:1b ist nicht installiert!")
                    logger.error("LÖSUNG: Führe aus: ollama pull llama3.2:1b")
            else:
                logger.error("Ollama-Server antwortet nicht korrekt")
                logger.error("LÖSUNG: Starte Ollama: ollama serve")
        except Exception as model_check_error:
            logger.error(f"Kann Ollama-Status nicht prüfen: {model_check_error}")
            logger.error("LÖSUNG: Starte Ollama manuell: ollama serve")
        
        # Bei Memory-Problemen oder Runner-Crash: Weiterlaufen aber warnen
        if "runner process has terminated" in error_str or "out of memory" in error_str:
            logger.warning("WARNUNG: Ollama-Model-Runner Problem erkannt")
            logger.warning("System startet trotzdem - API-Calls könnten fehlschlagen")
        else:
            logger.warning("WARNUNG: Starte ohne Ollama-Verbindung (nur für Debugging)")
            logger.warning("Das System wird NICHT funktional sein!")
        # raise  # Auskommentiert für Debugging
    
    # Settings konfigurieren
    logger.debug("Konfiguriere Settings...")
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Vector Store und Index initialisieren
    logger.debug("Initialisiere Vector Store...")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    logger.debug("Erstelle Index...")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # Qwen3-Reranker für verbesserte Retrieval-Qualität
    logger.debug("Konfiguriere Qwen3-Reranker für verbesserte Retrieval-Qualität...")
    qwen3_reranker = setup_qwen3_reranker(device="auto")
    
    # Erstelle zweistufigen Reranking-Retriever
    logger.debug("Erstelle Reranking-Retriever...")
    reranking_retriever = create_reranking_retriever(index, qwen3_reranker)
    
    # Erstelle Query Engine mit Reranking-Retriever
    logger.debug("Erstelle Query Engine mit Qwen3-Reranker...")
    from llama_index.core.query_engine import RetrieverQueryEngine
    
    # Manuell Query Engine mit unserem Reranking-Retriever erstellen
    query_engine = RetrieverQueryEngine.from_args(
        retriever=reranking_retriever,
        llm=llm
    )
    
    # Chat Engine mit Query Engine erstellen
    logger.debug("Erstelle Context Chat Engine mit Qwen3-Reranker...")
    from llama_index.core.chat_engine import ContextChatEngine
    
    chat_engine = ContextChatEngine.from_defaults(
        retriever=reranking_retriever,  # Verwende den retriever statt query_engine
        memory=None,  # Der Chat-Verlauf wird innerhalb der Session verwaltet
        system_prompt="""Du bist ein hilfreicher und präziser KI-Assistent. Deine Aufgabe ist es, Fragen professionell und ausschließlich auf Basis der dir als Kontext bereitgestellten Textabschnitte zu beantworten.

Wichtige Anweisungen für deine Antworten:

1. **Strikte Kontextbasierung:** Antworte *nur* mit Informationen, die direkt in den bereitgestellten Textabschnitten enthalten sind. Verwende kein externes Wissen oder eigene Annahmen.

2. **Präzision und Professionalität:** Formuliere deine Antworten konkret, sachlich und professionell.

3. **Umgang mit unzureichenden Informationen:** Wenn die bereitgestellten Textabschnitte die Frage nicht beantworten können, gib dies klar an. Erfinde keine Antworten.

4. **Quellenangabe:** Nenne am Ende deiner Antwort *immer* das Quelldokument und die Seitenzahl für jeden relevanten Textabschnitt, aus dem du Informationen entnommen hast, sofern diese Metadaten verfügbar sind. Nutze ein klares Format, z.B.: (Quelle: [Dokumentname], Seite: [Seitenzahl]).

5. **Sprache:** Antworte immer auf Deutsch.

Beginne jetzt mit der Beantwortung der Frage."""
    )
    
    # Logge Retrieval-System Konfiguration
    if qwen3_reranker is not None:
        logger.info("Zweistufiges Retrieval-System erfolgreich konfiguriert:")
        logger.info("  1. Qwen3-Embedding-0.6B fuer Dual-Encoder Vorauswahl (Top-100)")
        logger.info("  2. Qwen3-Reranker-4B fuer Cross-Encoder Reranking (Top-10)")
        logger.info("  -> Erwartete Verbesserung: +15-25% Relevanz-Praezision")
        logger.info("  -> Reduzierte RAM-Anforderungen gegenueber 8B Modell")
    else:
        logger.warning("Nur einstufiges Retrieval-System aktiv:")
        logger.warning("  - Qwen3-Embedding-0.6B fuer Dual-Encoder (Top-10)")
        logger.warning("  - Kein Reranking verfuegbar")
    
    # Speichere Chat-Verläufe in einem Dictionary
    # Key: Chat-ID, Value: Liste von ChatMessage Objekten
    chat_sessions = {}
    
    logger.info("Initialisierung erfolgreich abgeschlossen")
except Exception as e:
    logger.error(f"Fehler bei der Initialisierung: {str(e)}")
    logger.exception("Detaillierter Fehler:")
    raise

# 4. FastAPI App erstellen
app = FastAPI(
    title="RAG API",
    description="API für RAG-basierte Chat-Anwendungen",
    version="1.0.0"
)

# CORS Middleware konfigurieren
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Erlaubt alle Ursprünge - für Entwicklung
    allow_credentials=True,
    allow_methods=["*"],  # Erlaubt alle Methoden
    allow_headers=["*"],  # Erlaubt alle Header
)

# 5. Request/Response Modelle
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "llama3.2:1b"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    chat_id: Optional[str] = None  # Optionale Chat-ID zur Session-Verwaltung

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict
    chat_id: Optional[str] = None  # Gibt die Chat-ID zurück

# 6. API Endpunkte
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    try:
        # RAG-Log für neue Anfrage zurücksetzen
        reset_rag_log()
        rag_logger = setup_rag_logger()
        
        rag_logger.info("=" * 60)
        rag_logger.info("NEUE RAG-ANFRAGE GESTARTET")
        rag_logger.info("=" * 60)
        rag_logger.info(f"Zeitstempel: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        rag_logger.info(f"Chat-ID: {request.chat_id}")
        rag_logger.info(f"Model: {request.model}")
        rag_logger.info(f"Temperature: {request.temperature}")
        rag_logger.info(f"Max Tokens: {request.max_tokens}")
        rag_logger.info(f"Anzahl eingehender Messages: {len(request.messages)}")
        
        logger.debug(f"=== NEUE ANFRAGE EMPFANGEN ===")
        logger.debug(f"Chat-ID: {request.chat_id}")
        logger.debug(f"Anzahl Messages: {len(request.messages)}")
        
        # Detailliertes Logging aller eingehenden Messages
        rag_logger.info("\n--- EINGEHENDE MESSAGES ---")
        for i, msg in enumerate(request.messages):
            rag_logger.info(f"Message {i+1}:")
            rag_logger.info(f"  Role: {msg.role}")
            rag_logger.info(f"  Content: {msg.content}")
            rag_logger.info(f"  Länge: {len(msg.content)} Zeichen")
            
            logger.debug(f"Message {i+1}: Role='{msg.role}', Content='{msg.content[:100]}{'...' if len(msg.content) > 100 else ''}'")
            if "tag" in msg.content.lower() or "generierung" in msg.content.lower() or "erstelle" in msg.content.lower():
                rag_logger.warning(f"[WARNUNG] VERDÄCHTIGE MESSAGE ERKANNT in Message {i+1}: Enthält möglicherweise Tag-Generierung oder andere Prompt-Anweisungen")
                logger.warning(f"[WARNUNG] VERDÄCHTIGE MESSAGE ERKANNT in Message {i+1}: Enthält möglicherweise Tag-Generierung oder andere Prompt-Anweisungen")
        
        # Chat-ID ermitteln oder neue erzeugen
        chat_id = request.chat_id
        if not chat_id:
            chat_id = str(uuid.uuid4())
            rag_logger.info(f"\n--- CHAT-SESSION VERWALTUNG ---")
            rag_logger.info(f"Neue Chat-ID erstellt: {chat_id}")
            logger.debug(f"Neue Chat-ID erstellt: {chat_id}")
            chat_sessions[chat_id] = []
        elif chat_id not in chat_sessions:
            rag_logger.info(f"\n--- CHAT-SESSION VERWALTUNG ---")
            rag_logger.info(f"Chat-ID {chat_id} nicht gefunden, erstelle neue Session")
            logger.debug(f"Chat-ID {chat_id} nicht gefunden, erstelle neue Session")
            chat_sessions[chat_id] = []
        else:
            rag_logger.info(f"\n--- CHAT-SESSION VERWALTUNG ---")
            rag_logger.info(f"Existierende Chat-ID verwendet: {chat_id}")
        
        # Chat-History aus der Session laden
        chat_history = chat_sessions[chat_id]
        rag_logger.info(f"Chat-History geladen: {len(chat_history)} vorherige Nachrichten")
        logger.debug(f"Chat-History geladen für ID {chat_id}: {len(chat_history)} Nachrichten")
        
        # Konvertiere alle Nachrichten in ChatMessage-Objekte
        for msg in request.messages:
            # Füge nur neue Nachrichten hinzu, die noch nicht in der History sind
            if not any(existing_msg.content == msg.content for existing_msg in chat_history):
                role = MessageRole.USER if msg.role.lower() == "user" else MessageRole.ASSISTANT
                chat_history.append(ChatMessage(role=role, content=msg.content))
        
        # Extrahiere und validiere die letzte Benutzeranfrage
        user_messages = [msg for msg in request.messages if msg.role.lower() == "user"]
        if not user_messages:
            rag_logger.error("[FEHLER] FEHLER: Keine User-Messages gefunden!")
            logger.error("[FEHLER] FEHLER: Keine User-Messages gefunden!")
            raise HTTPException(status_code=400, detail="Keine Benutzeranfrage gefunden")
        
        # Nimm die letzte User-Message
        last_user_message = user_messages[-1]
        user_message = last_user_message.content
        
        rag_logger.info(f"\n--- USER-MESSAGE ANALYSE ---")
        rag_logger.info(f"Anzahl User-Messages gefunden: {len(user_messages)}")
        rag_logger.info(f"Letzte User-Message:")
        rag_logger.info(f"  Content: {user_message}")
        rag_logger.info(f"  Länge: {len(user_message)} Zeichen")
        
        logger.debug(f"=== LETZTE USER-MESSAGE ANALYSE ===")
        logger.debug(f"Content: '{user_message}'")
        logger.debug(f"Länge: {len(user_message)} Zeichen")
        
        # [ZIEL] META-ANFRAGE ERKENNUNG (OpenWebUI Title/Tag-Generierung)
        meta_info = is_meta_request(user_message)
        
        rag_logger.info(f"\n--- META-ANFRAGE ERKENNUNG ---")
        rag_logger.info(f"Meta-Anfrage erkannt: {meta_info['is_meta']}")
        if meta_info['is_meta']:
            rag_logger.info(f"Meta-Type: {meta_info['type']}")
            rag_logger.info(f"Confidence: {meta_info['confidence']:.2f}")
        
        if meta_info["is_meta"]:
            rag_logger.warning(f"[ZIEL] META-ANFRAGE ERKANNT: {meta_info['type']} (Confidence: {meta_info['confidence']:.2f})")
            rag_logger.info("[BLITZ] Umgehung von RAG - Direkte LLM-Verarbeitung")
            logger.warning(f"[ZIEL] META-ANFRAGE ERKANNT: {meta_info['type']} (Confidence: {meta_info['confidence']:.2f})")
            logger.info("[BLITZ] Umgehung von RAG - Direkte LLM-Verarbeitung")
            
            # Verarbeite Meta-Anfrage direkt mit LLM (ohne RAG)
            try:
                rag_logger.info(f"\n--- META-ANFRAGE VERARBEITUNG ---")
                rag_logger.info("Starte direkte LLM-Verarbeitung ohne RAG...")
                response_text = process_meta_request_with_llm(user_message, meta_info)
                rag_logger.info(f"LLM-Response erhalten: {response_text}")
                
                # Erstelle OpenAI-kompatible Antwort für Meta-Anfragen
                response = ChatResponse(
                    id=f"chatcmpl-{chat_id[:8]}-meta",
                    created=int(time.time()),
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }],
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    },
                    chat_id=chat_id
                )
                
                # Debug-Datei für Meta-Anfragen (ohne RAG-Daten)
                debug_info = {
                    "meta_request_detected": True,
                    "meta_type": meta_info['type'],
                    "meta_confidence": meta_info['confidence'],
                    "processing_method": "LLM-Only (RAG bypassed)",
                    "chat_history_length": len(chat_history),
                    "source_nodes_count": 0,  # Keine bei Meta-Anfragen
                    "unique_sources_count": 0
                }
                
                debug_file = create_debug_file(
                    request_data=request,
                    user_message_final=user_message,
                    llm_response=None,  # Kein RAG-Response bei Meta-Anfragen
                    response_data=response,
                    debug_info=debug_info
                )
                
                if debug_file:
                    rag_logger.info(f"[DATEI] Meta-Request Debug-Datei erstellt: {debug_file}")
                    logger.info(f"[DATEI] Meta-Request Debug-Datei erstellt: {debug_file}")
                
                rag_logger.info(f"[OK] Meta-Anfrage erfolgreich verarbeitet: {meta_info['type']}")
                rag_logger.info("=" * 60)
                rag_logger.info("RAG-ANFRAGE ABGESCHLOSSEN (META-REQUEST)")
                rag_logger.info("=" * 60)
                logger.info(f"[OK] Meta-Anfrage erfolgreich verarbeitet: {meta_info['type']}")
                return response
                
            except Exception as e:
                rag_logger.error(f"[FEHLER] Fehler bei Meta-Request Verarbeitung: {str(e)}")
                rag_logger.warning("[WECHSEL] Fallback auf normale RAG-Verarbeitung")
                logger.error(f"[FEHLER] Fehler bei Meta-Request Verarbeitung: {str(e)}")
                # Fallback - normale RAG-Verarbeitung
                logger.warning("[WECHSEL] Fallback auf normale RAG-Verarbeitung")
        
        # [SUCHE] NORMALE RAG-VERARBEITUNG (für echte User-Fragen)
        rag_logger.info(f"\n--- NORMALE RAG-VERARBEITUNG ---")
        rag_logger.info("[SUCHE] Starte normale RAG-Verarbeitung für echte User-Fragen")
        logger.info("[SUCHE] NORMALE RAG-VERARBEITUNG")
        
        # Validierung: Prüfe auf verdächtige Inhalte (falls Meta-Erkennung fehlgeschlagen)
        suspicious_keywords = [
            "tag", "generierung", "erstelle tags", "klassifizierung", 
            "kategorisierung", "prompt", "system", "instruction"
        ]
        
        is_suspicious = any(keyword in user_message.lower() for keyword in suspicious_keywords)
        rag_logger.info(f"\n--- VERDÄCHTIGE INHALTE PRÜFUNG ---")
        rag_logger.info(f"Verdächtige Keywords geprüft: {suspicious_keywords}")
        rag_logger.info(f"Verdächtige Inhalte erkannt: {is_suspicious}")
        
        if is_suspicious and not meta_info["is_meta"]:  # Nur warnen wenn nicht bereits als Meta erkannt
            rag_logger.warning(f"[WARNUNG] VERDÄCHTIGE USER-MESSAGE (nicht als Meta erkannt)")
            rag_logger.warning(f"Message: '{user_message}'")
            rag_logger.warning("Diese Message könnte eine unerkannte Meta-Anweisung sein!")
            logger.warning(f"[WARNUNG] VERDÄCHTIGE USER-MESSAGE (nicht als Meta erkannt): '{user_message[:200]}...'")
            logger.warning("Diese Message könnte eine unerkannte Meta-Anweisung sein!")
            
            # Optional: Versuche eine echte Frage zu extrahieren
            # Schaue nach vorherigen User-Messages, die möglicherweise die echte Frage enthalten
            potential_questions = []
            for i, msg in enumerate(reversed(user_messages[:-1])):  # Ohne die letzte (verdächtige)
                if len(msg.content) > 10 and not any(kw in msg.content.lower() for kw in suspicious_keywords):
                    potential_questions.append((len(user_messages) - 1 - i, msg.content))
                    rag_logger.debug(f"Potentielle echte Frage gefunden in Message {len(user_messages) - 1 - i}: '{msg.content[:100]}...'")
                    logger.debug(f"Potentielle echte Frage gefunden in Message {len(user_messages) - 1 - i}: '{msg.content[:100]}...'")
            
            if potential_questions:
                # Nimm die neueste potentielle echte Frage
                question_index, alternative_question = potential_questions[0]
                rag_logger.warning(f"[WECHSEL] VERWENDE ALTERNATIVE FRAGE aus Message {question_index}: '{alternative_question}'")
                logger.warning(f"[WECHSEL] VERWENDE ALTERNATIVE FRAGE aus Message {question_index}: '{alternative_question[:100]}...'")
                user_message = alternative_question
            else:
                rag_logger.error("[FEHLER] Keine alternative echte Frage gefunden!")
                rag_logger.warning("[WARNUNG] Fahre mit verdächtiger Message fort - Ergebnis könnte irrelevant sein!")
                logger.error("[FEHLER] Keine alternative echte Frage gefunden!")
                # Trotzdem fortfahren, aber warnen
                logger.warning("[WARNUNG] Fahre mit verdächtiger Message fort - Ergebnis könnte irrelevant sein!")

        # Finale Validierung: Mindestlänge und sinnvoller Inhalt
        if len(user_message.strip()) < 3:
            rag_logger.error(f"[FEHLER] User-Message zu kurz: '{user_message}'")
            logger.error(f"[FEHLER] User-Message zu kurz: '{user_message}'")
            raise HTTPException(status_code=400, detail="Benutzeranfrage zu kurz oder leer")
        
        rag_logger.info(f"\n--- FINALE USER-MESSAGE ---")
        rag_logger.info(f"Finale Message für RAG: '{user_message}'")
        rag_logger.info(f"Message-Länge: {len(user_message)} Zeichen")
        logger.debug(f"[OK] FINALE USER-MESSAGE für RAG: '{user_message}'")
        
        # Führe die Abfrage mit dem Context Chat Engine durch
        try:
            rag_logger.info(f"\n--- CHAT-ENGINE ABFRAGE ---")
            rag_logger.info("[SUCHE] Starte Chat-Engine Abfrage...")
            rag_logger.info(f"[ZIEL] Query: '{user_message}'")
            rag_logger.info(f"[BUCH] Chat-History Länge: {len(chat_history[:-1]) if chat_history else 0}")
            
            # Embedding der User-Query analysieren (falls möglich)
            try:
                user_query_embedding = embed_model.get_text_embedding(user_message)
                rag_logger.info(f"\n[GEHIRN] USER-QUERY EMBEDDING:")
                rag_logger.info(f"├─ Embedding erfolgreich erstellt")
                rag_logger.info(f"├─ Dimensionen: {len(user_query_embedding)}")
                rag_logger.info(f"├─ Embedding-Vektor (erste 10 Werte): {user_query_embedding[:10]}")
                rag_logger.info(f"└─ Embedding-Norm: {sum(x*x for x in user_query_embedding)**0.5:.6f}")
            except Exception as e:
                rag_logger.warning(f"[WARNUNG] Konnte User-Query Embedding nicht erstellen: {str(e)}")
            
            # ChromaDB Collection Status prüfen
            try:
                collection_count = chroma_collection.count()
                rag_logger.info(f"\n[STATISTIK] CHROMADB STATUS:")
                rag_logger.info(f"├─ Collection Name: {COLLECTION_NAME}")
                rag_logger.info(f"├─ Anzahl Dokumente in Collection: {collection_count}")
                rag_logger.info(f"└─ Collection verfügbar: {'[OK]' if collection_count > 0 else '[FEHLER]'}")
                
                if collection_count == 0:
                    rag_logger.error("[KRITISCH] KRITISCHER FEHLER: ChromaDB Collection ist leer!")
                    rag_logger.error("Das RAG-System kann keine Daten abrufen, da keine Dokumente indexiert sind.")
                    
            except Exception as e:
                rag_logger.error(f"[FEHLER] Fehler beim Prüfen der ChromaDB Collection: {str(e)}")
            
            # Logge Chat-Engine Konfiguration
            rag_logger.info(f"\n[CONFIG] CHAT-ENGINE KONFIGURATION:")
            rag_logger.info(f"├─ Chat Mode: context")
            
            # Erweiterte Chat-Engine Details mit Reranking-Info
            try:
                if hasattr(chat_engine, '_retriever'):
                    retriever = chat_engine._retriever
                    rag_logger.info(f"├─ Retriever Type: {type(retriever).__name__}")
                    rag_logger.info(f"├─ Similarity Top K: {getattr(retriever, 'similarity_top_k', 'N/A')}")
                    
                    # Reranking-spezifische Details
                    if hasattr(retriever, '_node_postprocessors') and retriever._node_postprocessors:
                        rag_logger.info(f"├─ [RERANKING] Postprocessors gefunden: {len(retriever._node_postprocessors)}")
                        for i, postprocessor in enumerate(retriever._node_postprocessors):
                            pp_type = type(postprocessor).__name__
                            rag_logger.info(f"│  └─ Postprocessor {i+1}: {pp_type}")
                            
                            # Qwen3-Reranker spezifische Details
                            if hasattr(postprocessor, 'model'):
                                model_name = getattr(postprocessor, 'model', 'N/A')
                                rag_logger.info(f"│     ├─ Model: {model_name}")
                            if hasattr(postprocessor, 'top_n'):
                                top_n = getattr(postprocessor, 'top_n', 'N/A')
                                rag_logger.info(f"│     ├─ Top N: {top_n}")
                            if hasattr(postprocessor, 'device'):
                                device = getattr(postprocessor, 'device', 'N/A')
                                rag_logger.info(f"│     └─ Device: {device}")
                        
                        rag_logger.info(f"├─ [RERANKING] Zweistufiges Retrieval aktiv:")
                        rag_logger.info(f"│  ├─ Stufe 1: Dual-Encoder (Qwen3-Embedding)")
                        rag_logger.info(f"│  │  └─ Vorauswahl: Top-{getattr(retriever, 'similarity_top_k', 100)} Kandidaten")
                        rag_logger.info(f"│  └─ Stufe 2: Cross-Encoder (Qwen3-Reranker)")
                        rag_logger.info(f"│     └─ Finale Auswahl: Top-10 nach Reranking")
                    else:
                        rag_logger.warning(f"├─ [WARNUNG] Kein Reranking aktiv - nur Embedding-Similarity")
                    
                    # Vector Index Details
                    if hasattr(retriever, '_index'):
                        vector_index = retriever._index
                        rag_logger.info(f"├─ Vector Index Type: {type(vector_index).__name__}")
                        
                        # Vector Store Details
                        if hasattr(vector_index, '_vector_store'):
                            vector_store_info = vector_index._vector_store
                            rag_logger.info(f"├─ Vector Store Type: {type(vector_store_info).__name__}")
                            
                if hasattr(chat_engine, '_llm'):
                    llm_info = chat_engine._llm
                    rag_logger.info(f"├─ LLM Type: {type(llm_info).__name__}")
                    rag_logger.info(f"├─ LLM Model: {getattr(llm_info, 'model', 'N/A')}")
                    rag_logger.info(f"├─ LLM Base URL: {getattr(llm_info, 'base_url', 'N/A')}")
                    
                if hasattr(chat_engine, '_system_prompt'):
                    system_prompt_preview = chat_engine._system_prompt[:100] + "..." if len(chat_engine._system_prompt) > 100 else chat_engine._system_prompt
                    rag_logger.info(f"└─ System Prompt (Vorschau): {system_prompt_preview}")
                else:
                    rag_logger.info(f"└─ System Prompt: N/A")
                    
            except Exception as e:
                rag_logger.warning(f"[WARNUNG] Konnte Chat-Engine Details nicht vollständig ermitteln: {str(e)}")
            
            logger.debug("[SUCHE] Starte Chat-Engine Abfrage...")
            # Verwende die aktuelle Chat-History für den Kontext
            llm_response = chat_engine.chat(
                message=user_message,
                chat_history=chat_history[:-1] if chat_history else None  # Letzte Nachricht ausschließen (ist die aktuelle Anfrage)
            )
            response_text = llm_response.response
            rag_logger.info(f"[OK] Chat-Engine Antwort erhalten")
            rag_logger.info(f"Response-Länge: {len(response_text)} Zeichen")
            rag_logger.info(f"Response-Vorschau: {response_text[:200]}...")
            
            # LLM Response Qualitäts-Analyse
            rag_logger.info(f"\n[ROBOTER] LLM-RESPONSE QUALITÄTS-ANALYSE:")
            rag_logger.info(f"├─ Response-Länge: {len(response_text)} Zeichen")
            rag_logger.info(f"├─ Anzahl Wörter: {len(response_text.split())}")
            rag_logger.info(f"├─ Anzahl Zeilen: {len(response_text.split('\n'))}")
            
            # Prüfe auf typische Probleme
            quality_issues = []
            if len(response_text) < 50:
                quality_issues.append("Response sehr kurz (möglicherweise unvollständig)")
            if "ich weiß nicht" in response_text.lower() or "keine information" in response_text.lower():
                quality_issues.append("LLM gibt an, keine Information zu haben")
            if "quelle:" not in response_text.lower() and "referenz" not in response_text.lower():
                quality_issues.append("Keine Quellenangaben in Response erkennbar")
            if len(response_text.split()) > 500:
                quality_issues.append("Response sehr lang (möglicherweise zu ausschweifend)")
                
            if quality_issues:
                rag_logger.warning("[WARNUNG] QUALITÄTSPROBLEME ERKANNT:")
                for issue in quality_issues:
                    rag_logger.warning(f"  - {issue}")
            else:
                rag_logger.info("├─ [OK] Keine offensichtlichen Qualitätsprobleme")
                
            # Sprach-Analyse
            german_indicators = ["der", "die", "das", "und", "oder", "aber", "mit", "von", "zu", "auf"]
            german_count = sum(1 for word in german_indicators if word in response_text.lower())
            rag_logger.info(f"├─ Deutsche Sprache erkannt: {'[OK]' if german_count >= 3 else '[FEHLER]'} ({german_count} Indikatoren)")
            rag_logger.info(f"└─ Response-Sprache entspricht User-Query: {'[OK]' if german_count >= 3 else '[WARNUNG]'}")
            
            logger.debug(f"[OK] Chat-Engine Antwort erhalten: {response_text[:100]}...")
            
            # FINALER KONTEXT ANALYSE - Was wurde tatsächlich an das LLM geschickt
            rag_logger.info(f"\n[ZIEL] FINALER KONTEXT AN LLM:")
            rag_logger.info("=" * 80)
            
            # Versuche den finalen Kontext zu extrahieren
            if hasattr(llm_response, 'source_nodes') and llm_response.source_nodes:
                combined_context = ""
                for i, node in enumerate(llm_response.source_nodes):
                    combined_context += f"[CONTEXT {i+1}]\n{node.text}\n\n"
                
                rag_logger.info(f"[KONTEXT] KOMBINIERTER KONTEXT (was an LLM gesendet wurde):")
                rag_logger.info(f"├─ Anzahl Kontext-Blöcke: {len(llm_response.source_nodes)}")
                rag_logger.info(f"├─ Gesamtlänge: {len(combined_context)} Zeichen")
                rag_logger.info(f"├─ Vollständiger Kontext:")
                rag_logger.info("│  " + "─" * 70)
                
                # Kontext mit Markierungen loggen
                context_lines = combined_context.split('\n')
                for line_num, line in enumerate(context_lines, 1):
                    rag_logger.info(f"│  {line_num:4d}: {line}")
                
                rag_logger.info("│  " + "─" * 70)
                rag_logger.info(f"└─ KONTEXT ENDE")
            
            # System Prompt und finale Query-Konstruktion
            if hasattr(chat_engine, '_system_prompt'):
                rag_logger.info(f"\n[ROBOTER] SYSTEM PROMPT AN LLM:")
                rag_logger.info("─" * 80)
                rag_logger.info(chat_engine._system_prompt)
                rag_logger.info("─" * 80)
            
            # Vollständige Anfrage-Rekonstruktion
            rag_logger.info(f"\n[NACHRICHT] FINALE LLM-ANFRAGE (rekonstruiert):")
            rag_logger.info("=" * 80)
            rag_logger.info("SYSTEM:")
            if hasattr(chat_engine, '_system_prompt'):
                rag_logger.info(chat_engine._system_prompt)
            rag_logger.info("\nKONTEXT:")
            if hasattr(llm_response, 'source_nodes') and llm_response.source_nodes:
                for i, node in enumerate(llm_response.source_nodes):
                    rag_logger.info(f"[KONTEXT {i+1}]: {node.text}")
            rag_logger.info(f"\nUSER QUERY: {user_message}")
            rag_logger.info("=" * 80)
            
            # Extrahiere die SourceNode-Objekte
            source_nodes = llm_response.source_nodes
            unique_source_strings = set()  # Um doppelte Quellenangaben zu vermeiden

            rag_logger.info(f"\n--- QUELLEN-ANALYSE ---")
            if source_nodes:
                rag_logger.info(f"[DATEI] Anzahl der Source Nodes: {len(source_nodes)}")
                rag_logger.info(f"[SUCHE] DETAILLIERTE CHUNK-ANALYSE:")
                rag_logger.info("=" * 80)
                
                logger.info(f"[DATEI] Anzahl der Source Nodes vom Chat Engine: {len(source_nodes)}")
                for i, node in enumerate(source_nodes):
                    file_name = node.metadata.get('filename', 'Unbekanntes Dokument')
                    page_number = node.metadata.get('page_number', 'N/A')  # 'N/A' wenn keine Seitenzahl verfügbar
                    score = node.score if hasattr(node, 'score') else 0.0

                    rag_logger.info(f"\n[ZIEL] CHUNK {i+1} von {len(source_nodes)}:")
                    rag_logger.info(f"┌─ METADATEN:")
                    rag_logger.info(f"│  [ORDNER] Datei: {file_name}")
                    rag_logger.info(f"│  [DATEI] Seite: {page_number}")
                    rag_logger.info(f"│  [STERN] Similarity Score: {score:.6f}")
                    
                    # Reranking-Score falls verfügbar (zeigt Verbesserung durch Cross-Encoder)
                    if hasattr(node, 'rerank_score'):
                        rerank_score = node.rerank_score
                        rag_logger.info(f"│  [QWEN] Reranking Score: {rerank_score:.6f}")
                        score_improvement = rerank_score - score if isinstance(rerank_score, (int, float)) and isinstance(score, (int, float)) else "N/A"
                        rag_logger.info(f"│  [BOOST] Score-Verbesserung: {score_improvement}")
                    else:
                        rag_logger.info(f"│  [INFO] Reranking Score: Nicht verfügbar")
                    
                    rag_logger.info(f"│  [ID] Node ID: {node.node_id if hasattr(node, 'node_id') else 'N/A'}")
                    
                    # Vollständige Metadaten loggen
                    if hasattr(node, 'metadata') and node.metadata:
                        rag_logger.info(f"│  [KONTEXT] Vollständige Metadaten:")
                        for key, value in node.metadata.items():
                            rag_logger.info(f"│     {key}: {value}")
                    
                    rag_logger.info(f"├─ CHUNK-TEXT:")
                    rag_logger.info(f"│  [TEXT] Text-Länge: {len(node.text)} Zeichen")
                    rag_logger.info(f"│  [DATEI] Vollständiger Text:")
                    rag_logger.info(f"│  " + "─" * 70)
                    
                    # Chunk-Text mit Zeilennummern für bessere Lesbarkeit
                    chunk_lines = node.text.split('\n')
                    for line_num, line in enumerate(chunk_lines, 1):
                        rag_logger.info(f"│  {line_num:3d}: {line}")
                    
                    rag_logger.info(f"│  " + "─" * 70)
                    
                    # Hash des Chunks für Eindeutigkeit
                    import hashlib
                    chunk_hash = hashlib.md5(node.text.encode()).hexdigest()[:8]
                    rag_logger.info(f"├─ CHUNK-HASH: {chunk_hash}")
                    
                    # Zusätzliche Node-Eigenschaften wenn verfügbar
                    if hasattr(node, 'embedding') and node.embedding:
                        rag_logger.info(f"├─ EMBEDDING: Verfügbar ({len(node.embedding)} Dimensionen)")
                    else:
                        rag_logger.info(f"├─ EMBEDDING: Nicht verfügbar")
                    
                    if hasattr(node, 'relationships') and node.relationships:
                        rag_logger.info(f"├─ BEZIEHUNGEN: {len(node.relationships)} Beziehungen")
                        for rel_type, rel_info in node.relationships.items():
                            rag_logger.info(f"│     {rel_type}: {rel_info}")
                    
                    rag_logger.info(f"└─ CHUNK {i+1} ENDE")
                    rag_logger.info("=" * 80)

                    # Logge detaillierte Infos zu jedem Node (bestehende Logik)
                    logger.debug(
                        f"  Source Node {i+1}: "
                        f"File='{file_name}', "
                        f"Page='{page_number}', "
                        f"Score={score:.4f}, "
                        f"Node ID='{node.node_id if hasattr(node, 'node_id') else 'N/A'}', "
                        f"Text (Vorschau)='{node.text[:70].replace('\n', ' ')}...'"
                    )
                    unique_source_strings.add(f"(Quelle: {file_name}, Seite: {page_number})")
                
                # Zusammenfassung der Chunks
                rag_logger.info(f"\n[STATISTIK] CHUNK-ZUSAMMENFASSUNG:")
                rag_logger.info(f"├─ Gesamtanzahl Chunks: {len(source_nodes)}")
                
                total_chars = sum(len(node.text) for node in source_nodes)
                rag_logger.info(f"├─ Gesamtzeichen aller Chunks: {total_chars}")
                rag_logger.info(f"├─ Durchschnittliche Chunk-Größe: {total_chars // len(source_nodes)} Zeichen")
                
                scores = [node.score for node in source_nodes if hasattr(node, 'score')]
                if scores:
                    rag_logger.info(f"├─ Höchster Score: {max(scores):.6f}")
                    rag_logger.info(f"├─ Niedrigster Score: {min(scores):.6f}")
                    rag_logger.info(f"├─ Durchschnittlicher Score: {sum(scores)/len(scores):.6f}")
                    
                    # Score-Qualitäts-Analyse
                    rag_logger.info(f"\n[ZIEL] SCORE-QUALITÄTS-ANALYSE:")
                    excellent_threshold = 0.8
                    good_threshold = 0.6
                    poor_threshold = 0.3
                    
                    excellent = [s for s in scores if s >= excellent_threshold]
                    good = [s for s in scores if good_threshold <= s < excellent_threshold]
                    fair = [s for s in scores if poor_threshold <= s < good_threshold]
                    poor = [s for s in scores if s < poor_threshold]
                    
                    rag_logger.info(f"├─ Exzellente Matches (≥{excellent_threshold}): {len(excellent)}")
                    rag_logger.info(f"├─ Gute Matches ({good_threshold}-{excellent_threshold}): {len(good)}")
                    rag_logger.info(f"├─ Mittelmäßige Matches ({poor_threshold}-{good_threshold}): {len(fair)}")
                    rag_logger.info(f"└─ Schlechte Matches (<{poor_threshold}): {len(poor)}")
                    
                    if len(poor) == len(scores):
                        rag_logger.warning("[KRITISCH] ACHTUNG: Alle Chunks haben schlechte Similarity-Scores!")
                        rag_logger.warning("Das deutet auf ein Problem mit:")
                        rag_logger.warning("- Embedding-Modell nicht geeignet für die Daten")
                        rag_logger.warning("- Query-Sprache passt nicht zu indexierten Daten")
                        rag_logger.warning("- Chunk-Größe ungeeignet")
                        rag_logger.warning("- Datenqualität der indexierten Chunks")
                        
                        # Reranking kann bei schlechten Embedding-Scores helfen
                        rerank_scores = [getattr(node, 'rerank_score', None) for node in source_nodes]
                        rerank_scores_available = [s for s in rerank_scores if s is not None]
                        if rerank_scores_available:
                            avg_rerank = sum(rerank_scores_available) / len(rerank_scores_available)
                            rag_logger.info(f"[RERANKING] Durchschnittlicher Reranking-Score: {avg_rerank:.6f}")
                            if avg_rerank > 0.5:
                                                            rag_logger.info("[RERANKING] [OK] Reranker hat relevante Chunks identifiziert trotz schlechter Embedding-Scores!")
                        else:
                            rag_logger.warning("[RERANKING] [WARNUNG] Auch Reranker findet Chunks wenig relevant")
                        
                    elif len(excellent) == 0 and len(good) == 0:
                        rag_logger.warning("[WARNUNG] Warnung: Keine guten Matches gefunden!")
                        rag_logger.warning("Die gefundenen Chunks sind möglicherweise nicht relevant.")
                        
                        # Prüfe ob Reranking bessere Ergebnisse liefert
                        rerank_scores = [getattr(node, 'rerank_score', None) for node in source_nodes]
                        rerank_scores_available = [s for s in rerank_scores if s is not None]
                        if rerank_scores_available:
                            max_rerank = max(rerank_scores_available)
                            rag_logger.info(f"[RERANKING] Hoechster Reranking-Score: {max_rerank:.6f}")
                            if max_rerank > 0.7:
                                rag_logger.info("[RERANKING] [OK] Reranker hat relevantere Chunks gefunden!")
                            elif max_rerank > max(scores):
                                rag_logger.info("[RERANKING] [OK] Reranker hat Score-Verbesserung erzielt")
                            else:
                                rag_logger.warning("[RERANKING] [WARNUNG] Reranker bestaetigt niedrige Relevanz")
                
                unique_files = set(node.metadata.get('filename', 'Unbekannt') for node in source_nodes)
                rag_logger.info(f"├─ Anzahl verschiedener Dateien: {len(unique_files)}")
                rag_logger.info(f"└─ Dateien: {', '.join(unique_files)}")
                
                # Chunk-Verteilungs-Analyse
                rag_logger.info(f"\n[ANALYSE] CHUNK-VERTEILUNGS-ANALYSE:")
                file_chunk_count = {}
                for node in source_nodes:
                    filename = node.metadata.get('filename', 'Unbekannt')
                    file_chunk_count[filename] = file_chunk_count.get(filename, 0) + 1
                    
                for filename, count in file_chunk_count.items():
                    percentage = (count / len(source_nodes)) * 100
                    rag_logger.info(f"├─ {filename}: {count} Chunks ({percentage:.1f}%)")
                    
                if len(file_chunk_count) == 1:
                    rag_logger.info("└─ [OK] Alle Chunks aus einer Datei (gut fokussiert)")
                elif len(file_chunk_count) > 3:
                    rag_logger.warning("└─ [WARNUNG] Chunks aus vielen verschiedenen Dateien (möglicherweise zu unspezifisch)")
                else:
                    rag_logger.info("└─ [OK] Moderate Anzahl verschiedener Quellen")
                
            else:
                rag_logger.info("[FEHLER] Keine Source Nodes gefunden")
                rag_logger.warning("[WARNUNG] Das bedeutet, dass keine relevanten Chunks aus der ChromaDB abgerufen wurden!")
                rag_logger.info("Mögliche Ursachen:")
                rag_logger.info("- Query zu spezifisch oder ungewöhnlich")
                rag_logger.info("- Embedding-Modell findet keine Ähnlichkeiten")
                rag_logger.info("- ChromaDB ist leer oder unvollständig indexiert")
                rag_logger.info("- Similarity-Threshold zu hoch")
                logger.info("Keine Source Nodes gefunden oder keine eindeutigen Quelleninformationen extrahiert.")

            # Der System Prompt instruiert den LLM, die Quellen zu nennen.
            # Die folgende Logik ist ein Fallback oder zur expliziten Darstellung.
            if unique_source_strings:
                rag_logger.info(f"\n--- QUELLEN-INTEGRATION ---")
                rag_logger.info(f"Anzahl eindeutige Quellen: {len(unique_source_strings)}")
                rag_logger.info(f"Gefundene Quellen: {list(unique_source_strings)}")
                
                # Stelle sicher, dass die Quellen nicht bereits sehr ähnlich im Text vom LLM genannt wurden.
                # Einfache Prüfung:
                already_mentioned = False
                for src_str in unique_source_strings:
                    if src_str.lower() in response_text.lower():  # Einfache Substring-Suche
                        already_mentioned = True
                        rag_logger.debug(f"Quelle '{src_str}' scheint bereits vom LLM im Text erwähnt worden zu sein.")
                        logger.debug(f"Quelle '{src_str}' scheint bereits vom LLM im Text erwähnt worden zu sein.")
                        break  # Eine Erwähnung reicht als Indikator

                if not already_mentioned:
                    rag_logger.info("LLM hat Quellen nicht explizit im Text genannt, füge sie manuell hinzu.")
                    logger.info("LLM hat Quellen nicht explizit im Text genannt, füge sie manuell hinzu.")
                    response_text += "\n\n**Referenzierte Quellen:**\n" + "\n".join(sorted(list(unique_source_strings)))
                else:
                    rag_logger.info("LLM scheint Quellen bereits im Text erwähnt zu haben.")
                    logger.info("LLM scheint Quellen bereits im Text erwähnt zu haben oder der System Prompt deckt dies ab. Keine manuelle Ergänzung der Quellen im Text.")
            else:
                rag_logger.info("Keine eindeutige Quelleninformationen für Integration gefunden.")
                logger.info("Keine Source Nodes gefunden oder keine eindeutigen Quelleninformationen extrahiert.")
            
            # Füge die Antwort des Assistenten zur Chat-History hinzu
            chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=response_text))
            # Aktualisiere die Session
            chat_sessions[chat_id] = chat_history
            
            rag_logger.info(f"\n--- CHAT-HISTORY UPDATE ---")
            rag_logger.info(f"Antwort zur Chat-History hinzugefügt")
            rag_logger.info(f"Neue Chat-History Länge: {len(chat_history)} Nachrichten")
            
        except Exception as e:
            rag_logger.error(f"[FEHLER] Fehler bei der Chat-Engine Abfrage: {str(e)}")
            rag_logger.error(f"Chat Engine Status: {chat_engine}")
            logger.error(f"[FEHLER] Fehler bei der Chat-Engine Abfrage: {str(e)}")
            logger.error(f"Chat Engine Status: {chat_engine}")
            logger.exception("Detaillierter Chat-Engine Fehler:")
            raise HTTPException(status_code=500, detail=f"Chat-Engine Fehler: {str(e)}")
        
        # Erstelle OpenAI-kompatible Antwort
        response = ChatResponse(
            id=f"chatcmpl-{chat_id[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            chat_id=chat_id  # Gib die Chat-ID zurück
        )
        
        rag_logger.info(f"\n--- RESPONSE-ERSTELLUNG ---")
        rag_logger.info(f"OpenAI-kompatible Response erstellt")
        rag_logger.info(f"Response-ID: chatcmpl-{chat_id[:8]}")
        rag_logger.info(f"Model: {request.model}")
        rag_logger.info(f"Chat-ID: {chat_id}")
        rag_logger.debug(f"[OK] Antwort erfolgreich erstellt")

        # Debug-Datei erstellen mit allen gesammelten Informationen
        debug_info = {
            "suspicious_message_detected": is_suspicious if 'is_suspicious' in locals() else False,
            "alternative_question_used": 'alternative_question' in locals(),
            "chat_history_length": len(chat_history),
            "source_nodes_count": len(source_nodes) if source_nodes else 0,
            "unique_sources_count": len(unique_source_strings) if unique_source_strings else 0
        }
        
        debug_file = create_debug_file(
            request_data=request,
            user_message_final=user_message,
            llm_response=llm_response,
            response_data=response,
            debug_info=debug_info
        )
        
        if debug_file:
            rag_logger.info(f"[DATEI] Debug-Datei erstellt: {debug_file}")
            logger.info(f"[DATEI] Debug-Datei erstellt: {debug_file}")
        
        rag_logger.info("=" * 60)
        rag_logger.info("RAG-ANFRAGE ERFOLGREICH ABGESCHLOSSEN")
        rag_logger.info("=" * 60)
        
        return response
    except Exception as e:
        rag_logger.error(f"[FEHLER] UNBEHANDELTER FEHLER: {str(e)}")
        rag_logger.error(f"Request Details: {request.dict() if request else 'No request data'}")
        rag_logger.error("=" * 60)
        rag_logger.error("RAG-ANFRAGE MIT FEHLER BEENDET")
        rag_logger.error("=" * 60)
        logger.error(f"[FEHLER] Unbehandelter Fehler: {str(e)}")
        logger.error(f"Request Details: {request.dict() if request else 'No request data'}")
        logger.exception("Detaillierter unbehandelter Fehler:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "llama3.2:1b"}

# Ollama API-kompatible Endpunkte für OpenWebUI
@app.get("/api/tags")
async def api_tags():
    """Listet verfügbare Modelle auf (Ollama API-kompatibel)"""
    return {
        "models": [
            {
                "name": "llama3.2:1b",
                "model": "llama3.2:1b", 
                "modified_at": "2024-06-07T16:00:00Z",
                "size": 1300000000,
                "digest": "baf6a787fdff",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "1B",
                    "quantization_level": "Q4_0"
                }
            }
        ]
    }

@app.get("/api/ps")
async def api_ps():
    """Zeigt laufende Modelle an (Ollama API-kompatibel)"""
    return {
        "models": [
            {
                "name": "llama3.2:1b",
                "model": "llama3.2:1b",
                "size": 1300000000,
                "digest": "baf6a787fdff",
                "expires_at": "2024-06-07T17:00:00Z"
            }
        ]
    }

@app.get("/api/version")
async def api_version():
    """API-Versionsinformationen (Ollama API-kompatibel)"""
    return {
        "version": "0.1.0",
        "build": "rag-api-custom"
    }

# Ollama Chat API-Endpunkt (Alternative zu OpenAI-Format)
@app.post("/api/chat")
async def api_chat(request: dict):
    """Ollama-nativer Chat-Endpunkt"""
    try:
        # Konvertiere Ollama-Format zu internem Format
        messages = []
        if "messages" in request:
            for msg in request["messages"]:
                messages.append(Message(role=msg["role"], content=msg["content"]))
        else:
            # Fallback für einfache Prompt-Anfragen
            messages.append(Message(role="user", content=request.get("prompt", "")))
        
        # Erstelle ChatRequest
        chat_request = ChatRequest(
            messages=messages,
            model=request.get("model", "llama3.2:1b"),
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens", 1000)
        )
        
        # Verwende die bestehende Chat-Logik
        response = await chat_completions(chat_request)
        
        # Konvertiere zurück zu Ollama-Format
        return {
            "model": request.get("model", "llama3.2:1b"),
            "created_at": "2024-06-07T16:00:00Z",
            "message": {
                "role": "assistant",
                "content": response.choices[0]["message"]["content"]
            },
            "done": True
        }
        
    except Exception as e:
        logger.error(f"Fehler im Ollama Chat-Endpunkt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "RAG API ist online",
        "endpoints": {
            "/v1/chat/completions": "Chat-Endpunkt (OpenAI-kompatibel)",
            "/api/chat": "Chat-Endpunkt (Ollama-kompatibel)",
            "/api/tags": "Modell-Liste (Ollama-kompatibel)",
            "/api/ps": "Laufende Modelle (Ollama-kompatibel)",
            "/api/version": "API-Version (Ollama-kompatibel)",
            "/health": "System-Status"
        }
    }

# 7. Startbefehl für uvicorn (wird in der Kommandozeile ausgeführt)
# uvicorn rag_api:app --host 0.0.0.0 --port 8000 --reload

# 8. Debug-Datei Funktion
def create_debug_file(request_data, user_message_final, llm_response, response_data, debug_info):
    """
    Schreibt detaillierte Debug-Informationen ins Log statt in separate Dateien
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        
        # Hauptlogger-Konfiguration anpassen, damit er in die bft_launcher.log schreibt
        launcher_log_handler = logging.FileHandler('bft_launcher.log')
        launcher_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        debug_logger = logging.getLogger("rag_debug")
        debug_logger.setLevel(logging.DEBUG)
        debug_logger.addHandler(launcher_log_handler)
        
        # Debug-Informationen ins Log schreiben
        debug_logger.info(f"========== DEBUG-EINTRAG {timestamp} ==========")
        
        # 1. EINGEHENDE REQUEST-DATEN
        debug_logger.info(f"[SUCHE] EINGEHENDE REQUEST-DATEN: Chat-ID={request_data.chat_id}, Model={request_data.model}")
        
        # 2. VERARBEITUNGS-INFORMATIONEN
        for key, value in debug_info.items():
            debug_logger.info(f"[CONFIG] {key}: {value}")
        
        # 3. FINALE USER-MESSAGE & ANTWORT
        debug_logger.info(f"[OK] USER-MESSAGE: '{user_message_final[:100]}...' (Länge: {len(user_message_final)})")
        
        if llm_response:
            debug_logger.info(f"[ROBOTER] RAG-ANTWORT: '{llm_response.response[:100]}...' (Länge: {len(llm_response.response)})")
            
            if hasattr(llm_response, 'source_nodes') and llm_response.source_nodes:
                sources = []
                for i, node in enumerate(llm_response.source_nodes):
                    filename = node.metadata.get('filename', 'Unbekannt')
                    page = node.metadata.get('page_number', 'N/A')
                    sources.append(f"{filename}:{page}")
                debug_logger.info(f"[DATEI] QUELLEN: {', '.join(sources)}")
        else:
            # Meta-Anfrage ohne RAG
            debug_logger.info(f"[ZIEL] META-ANFRAGE: '{response_data.choices[0]['message']['content'][:100]}...'")
        
        debug_logger.info(f"========== ENDE DEBUG-EINTRAG {timestamp} ==========")
        
        # Handler entfernen, um keine doppelten Einträge zu erzeugen
        debug_logger.removeHandler(launcher_log_handler)
        
        return f"debug_{timestamp}"  # Nur für Rückwärtskompatibilität
        
    except Exception as e:
        logger.error(f"[FEHLER] Fehler beim Schreiben der Debug-Informationen: {str(e)}")
        return None

# 9. Qwen3-Reranker Performance-Optimierung
def log_reranking_performance_tips():
    """
    Loggt Tipps zur Optimierung der Qwen3-Reranker Performance.
    """
    logger.info("\n=== QWEN3-RERANKER PERFORMANCE-TIPPS ===")
    logger.info("📊 Aktuelle Konfiguration:")
    logger.info("  - Similarity Top K: 100 (Dual-Encoder Vorauswahl)")
    logger.info("  - Reranking Top N: 10 (Cross-Encoder finale Auswahl)")
    logger.info("  - Embedding Model: Qwen3-Embedding-0.6B")
    logger.info("  - Reranker Model: Qwen3-Reranker-8B")
    
    logger.info("\nOptimierungsmoeglichkeiten:")
    logger.info("  1. GPU-Performance:")
    logger.info("     - Nutze torch.float16 fuer GPU-Inference")
    logger.info("     - Aktiviere torch.compile() fuer PyTorch 2.0+")
    logger.info("     - Verwende Flash Attention 2 falls verfuegbar")
    
    logger.info("  2. Parameter-Tuning:")
    logger.info("     - Erhoehe similarity_top_k auf 200 fuer komplexe Queries")
    logger.info("     - Reduziere auf 50 fuer bessere Latenz")
    logger.info("     - Teste verschiedene reranking top_n Werte (5-15)")
    
    logger.info("  3. Qualitaets-Verbesserung:")
    logger.info("     - Verwende Qwen3-Reranker-8B fuer maximale Qualitaet")
    logger.info("     - Fallback auf Qwen3-Reranker-4B fuer bessere Performance")
    logger.info("     - Qwen3-Reranker-0.6B fuer ressourcenbegrenzte Umgebungen")
    
    logger.info("  4. Speicher-Optimierung:")
    logger.info("     - Batch-Processing fuer groessere Dokumentensets")
    logger.info("     - Gradient Checkpointing fuer grosse Modelle")
    logger.info("     - Model Sharding bei Multi-GPU Setups")
    
    logger.info("=== ENDE PERFORMANCE-TIPPS ===\n")

# 10. Meta-Anfragen Erkennungs- und Verarbeitungsfunktionen
def is_meta_request(content: str) -> dict:
    """
    Erkennt OpenWebUI Meta-Anfragen (Title/Tag-Generierung) und gibt Details zurück
    
    Returns:
        dict: {
            "is_meta": bool,
            "type": str,  # "title", "tags", "unknown"
            "confidence": float  # 0.0 - 1.0
        }
    """
    content_lower = content.lower()
    
    # Starke Indikatoren für Meta-Anfragen
    meta_indicators = {
        # Title-Generierung Indikatoren
        "title": [
            "### task:",
            "generate a concise",
            "word title with an emoji",
            "summarizing the chat history",
            "### guidelines:",
            "### output:",
            "json format:",
            '"title":',
            "### chat history:",
            "<chat_history>"
        ],
        # Tag-Generierung Indikatoren  
        "tags": [
            "### task:",
            "generate 1-3 broad tags",
            "categorizing the main themes",
            "### guidelines:",
            "high-level domains",
            "### output:",
            "json format:",
            '"tags":',
            "### chat history:",
            "<chat_history>"
        ]
    }
    
    # Prüfe für jeden Meta-Type
    for meta_type, indicators in meta_indicators.items():
        score = 0
        total_indicators = len(indicators)
        
        for indicator in indicators:
            if indicator in content_lower:
                score += 1
        
        confidence = score / total_indicators
        
        # Wenn mehr als 60% der Indikatoren gefunden werden, ist es wahrscheinlich eine Meta-Anfrage
        if confidence >= 0.6:
            return {
                "is_meta": True,
                "type": meta_type,
                "confidence": confidence
            }
    
    # Allgemeine Meta-Anfrage Indikatoren (falls neue Patterns auftauchen)
    general_meta_patterns = [
        "### task:",
        "### guidelines:",
        "### output:",
        "json format:",
        "### chat history:",
        "<chat_history>",
        "generate a",
        "summarizing the chat"
    ]
    
    general_score = sum(1 for pattern in general_meta_patterns if pattern in content_lower)
    general_confidence = general_score / len(general_meta_patterns)
    
    if general_confidence >= 0.5:
        return {
            "is_meta": True,
            "type": "unknown",
            "confidence": general_confidence
        }
    
    return {
        "is_meta": False,
        "type": None,
        "confidence": 0.0
    }

def process_meta_request_with_llm(content: str, meta_info: dict) -> str:
    """
    Verarbeitet Meta-Anfragen direkt mit dem LLM ohne RAG-Datenbank
    """
    try:
        logger.info(f"[ZIEL] VERARBEITE META-ANFRAGE: {meta_info['type']} (Confidence: {meta_info['confidence']:.2f})")
        
        # Direkter LLM-Call ohne RAG
        llm_response = llm.complete(content)
        response_text = str(llm_response)
        
        logger.debug(f"[ROBOTER] LLM-Only Antwort: {response_text[:100]}...")
        return response_text
        
    except Exception as e:
        logger.error(f"[FEHLER] Fehler bei Meta-Request LLM-Verarbeitung: {str(e)}")
        # Fallback für Meta-Anfragen
        if meta_info['type'] == 'title':
            return '{ "title": "[KONTEXT] Geschäftliche Anfrage" }'
        elif meta_info['type'] == 'tags':
            return '{ "tags": ["Business", "General"] }'
        else:
            return "Entschuldigung, ich kann diese Meta-Anfrage derzeit nicht verarbeiten."