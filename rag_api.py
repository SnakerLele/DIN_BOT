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
from typing import List, Optional
import time
import logging
import sys
from typing import List, Optional
import uuid
import json
from datetime import datetime
import os
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

# 3. Initialisierung (wird nur beim Serverstart ausgeführt)
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
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    
    # LLM initialisieren
    logger.debug("Verbinde mit Ollama...")
    llm = Ollama(model="llama3.2:1b", base_url="http://localhost:11434", request_timeout=120.0)
    # Test Ollama Verbindung
    try:
        logger.debug("Teste Ollama-Verbindung...")
        test_response = llm.complete("Test")
        logger.debug("Ollama-Verbindung erfolgreich")
    except Exception as e:
        logger.error(f"Ollama-Verbindungstest fehlgeschlagen: {str(e)}")
        raise
    
    # Settings konfigurieren
    logger.debug("Konfiguriere Settings...")
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Vector Store und Index initialisieren
    logger.debug("Initialisiere Vector Store...")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    logger.debug("Erstelle Index...")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # Erstelle den Context Chat Engine anstelle von query_engine
    logger.debug("Erstelle Context Chat Engine...")
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=None,  # Der Chat-Verlauf wird innerhalb der Session verwaltet
        system_prompt="Du bist ein KI-Assistent für Mitarbeiter eines deutschen Planungsbüros im Baugewerbe. Deine Aufgabe ist es, Fragen präzise zu beantworten, indem du Informationen aus Dokumenten vergangener Projekte nutzt. Fasse die relevanten Fakten strukturiert zusammen. Nenne am Ende deiner Antwort immer das Quelldokument und die Seitenzahl. Antworte professionell und auf Deutsch. Gib an, wenn du etwas nicht weißt.",
        similarity_top_k=4
    )
    
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
        logger.debug(f"=== NEUE ANFRAGE EMPFANGEN ===")
        logger.debug(f"Chat-ID: {request.chat_id}")
        logger.debug(f"Anzahl Messages: {len(request.messages)}")
        
        # Detailliertes Logging aller eingehenden Messages
        for i, msg in enumerate(request.messages):
            logger.debug(f"Message {i+1}: Role='{msg.role}', Content='{msg.content[:100]}{'...' if len(msg.content) > 100 else ''}'")
            if "tag" in msg.content.lower() or "generierung" in msg.content.lower() or "erstelle" in msg.content.lower():
                logger.warning(f"⚠️ VERDÄCHTIGE MESSAGE ERKANNT in Message {i+1}: Enthält möglicherweise Tag-Generierung oder andere Prompt-Anweisungen")
        
        # Chat-ID ermitteln oder neue erzeugen
        chat_id = request.chat_id
        if not chat_id:
            chat_id = str(uuid.uuid4())
            logger.debug(f"Neue Chat-ID erstellt: {chat_id}")
            chat_sessions[chat_id] = []
        elif chat_id not in chat_sessions:
            logger.debug(f"Chat-ID {chat_id} nicht gefunden, erstelle neue Session")
            chat_sessions[chat_id] = []
        
        # Chat-History aus der Session laden
        chat_history = chat_sessions[chat_id]
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
            logger.error("❌ FEHLER: Keine User-Messages gefunden!")
            raise HTTPException(status_code=400, detail="Keine Benutzeranfrage gefunden")
        
        # Nimm die letzte User-Message
        last_user_message = user_messages[-1]
        user_message = last_user_message.content
        
        logger.debug(f"=== LETZTE USER-MESSAGE ANALYSE ===")
        logger.debug(f"Content: '{user_message}'")
        logger.debug(f"Länge: {len(user_message)} Zeichen")
        
        # 🎯 META-ANFRAGE ERKENNUNG (OpenWebUI Title/Tag-Generierung)
        meta_info = is_meta_request(user_message)
        
        if meta_info["is_meta"]:
            logger.warning(f"🎯 META-ANFRAGE ERKANNT: {meta_info['type']} (Confidence: {meta_info['confidence']:.2f})")
            logger.info("⚡ Umgehung von RAG - Direkte LLM-Verarbeitung")
            
            # Verarbeite Meta-Anfrage direkt mit LLM (ohne RAG)
            try:
                response_text = process_meta_request_with_llm(user_message, meta_info)
                
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
                    logger.info(f"📝 Meta-Request Debug-Datei erstellt: {debug_file}")
                
                logger.info(f"✅ Meta-Anfrage erfolgreich verarbeitet: {meta_info['type']}")
                return response
                
            except Exception as e:
                logger.error(f"❌ Fehler bei Meta-Request Verarbeitung: {str(e)}")
                # Fallback - normale RAG-Verarbeitung
                logger.warning("🔄 Fallback auf normale RAG-Verarbeitung")
        
        # 🔍 NORMALE RAG-VERARBEITUNG (für echte User-Fragen)
        logger.info("🔍 NORMALE RAG-VERARBEITUNG")
        
        # Validierung: Prüfe auf verdächtige Inhalte (falls Meta-Erkennung fehlgeschlagen)
        suspicious_keywords = [
            "tag", "generierung", "erstelle tags", "klassifizierung", 
            "kategorisierung", "prompt", "system", "instruction"
        ]
        
        is_suspicious = any(keyword in user_message.lower() for keyword in suspicious_keywords)
        if is_suspicious and not meta_info["is_meta"]:  # Nur warnen wenn nicht bereits als Meta erkannt
            logger.warning(f"⚠️ VERDÄCHTIGE USER-MESSAGE (nicht als Meta erkannt): '{user_message[:200]}...'")
            logger.warning("Diese Message könnte eine unerkannte Meta-Anweisung sein!")
            
            # Optional: Versuche eine echte Frage zu extrahieren
            # Schaue nach vorherigen User-Messages, die möglicherweise die echte Frage enthalten
            potential_questions = []
            for i, msg in enumerate(reversed(user_messages[:-1])):  # Ohne die letzte (verdächtige)
                if len(msg.content) > 10 and not any(kw in msg.content.lower() for kw in suspicious_keywords):
                    potential_questions.append((len(user_messages) - 1 - i, msg.content))
                    logger.debug(f"Potentielle echte Frage gefunden in Message {len(user_messages) - 1 - i}: '{msg.content[:100]}...'")
            
            if potential_questions:
                # Nimm die neueste potentielle echte Frage
                question_index, alternative_question = potential_questions[0]
                logger.warning(f"🔄 VERWENDE ALTERNATIVE FRAGE aus Message {question_index}: '{alternative_question[:100]}...'")
                user_message = alternative_question
            else:
                logger.error("❌ Keine alternative echte Frage gefunden!")
                # Trotzdem fortfahren, aber warnen
                logger.warning("⚠️ Fahre mit verdächtiger Message fort - Ergebnis könnte irrelevant sein!")

        # Finale Validierung: Mindestlänge und sinnvoller Inhalt
        if len(user_message.strip()) < 3:
            logger.error(f"❌ User-Message zu kurz: '{user_message}'")
            raise HTTPException(status_code=400, detail="Benutzeranfrage zu kurz oder leer")
        
        logger.debug(f"✅ FINALE USER-MESSAGE für RAG: '{user_message}'")
        
        # Führe die Abfrage mit dem Context Chat Engine durch
        try:
            logger.debug("🔍 Starte Chat-Engine Abfrage...")
            # Verwende die aktuelle Chat-History für den Kontext
            llm_response = chat_engine.chat(
                message=user_message,
                chat_history=chat_history[:-1] if chat_history else None  # Letzte Nachricht ausschließen (ist die aktuelle Anfrage)
            )
            response_text = llm_response.response
            logger.debug(f"✅ Chat-Engine Antwort erhalten: {response_text[:100]}...")
            
            # Extrahiere die SourceNode-Objekte
            source_nodes = llm_response.source_nodes
            unique_source_strings = set()  # Um doppelte Quellenangaben zu vermeiden

            if source_nodes:
                logger.info(f"📄 Anzahl der Source Nodes vom Chat Engine: {len(source_nodes)}")
                for i, node in enumerate(source_nodes):
                    file_name = node.metadata.get('filename', 'Unbekanntes Dokument')
                    page_number = node.metadata.get('page_number', 'N/A')  # 'N/A' wenn keine Seitenzahl verfügbar
                    score = node.score if hasattr(node, 'score') else 0.0

                    # Logge detaillierte Infos zu jedem Node
                    logger.debug(
                        f"  Source Node {i+1}: "
                        f"File='{file_name}', "
                        f"Page='{page_number}', "
                        f"Score={score:.4f}, "
                        f"Node ID='{node.node_id if hasattr(node, 'node_id') else 'N/A'}', "
                        f"Text (Vorschau)='{node.text[:70].replace('\n', ' ')}...'"
                    )
                    unique_source_strings.add(f"(Quelle: {file_name}, Seite: {page_number})")

            # Der System Prompt instruiert den LLM, die Quellen zu nennen.
            # Die folgende Logik ist ein Fallback oder zur expliziten Darstellung.
            if unique_source_strings:
                # Stelle sicher, dass die Quellen nicht bereits sehr ähnlich im Text vom LLM genannt wurden.
                # Einfache Prüfung:
                already_mentioned = False
                for src_str in unique_source_strings:
                    if src_str.lower() in response_text.lower():  # Einfache Substring-Suche
                        already_mentioned = True
                        logger.debug(f"Quelle '{src_str}' scheint bereits vom LLM im Text erwähnt worden zu sein.")
                        break  # Eine Erwähnung reicht als Indikator

                if not already_mentioned:
                    logger.info("LLM hat Quellen nicht explizit im Text genannt, füge sie manuell hinzu.")
                    response_text += "\n\n**Referenzierte Quellen:**\n" + "\n".join(sorted(list(unique_source_strings)))
                else:
                    logger.info("LLM scheint Quellen bereits im Text erwähnt zu haben oder der System Prompt deckt dies ab. Keine manuelle Ergänzung der Quellen im Text.")
            else:
                logger.info("Keine Source Nodes gefunden oder keine eindeutigen Quelleninformationen extrahiert.")
            
            # Füge die Antwort des Assistenten zur Chat-History hinzu
            chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=response_text))
            # Aktualisiere die Session
            chat_sessions[chat_id] = chat_history
            
        except Exception as e:
            logger.error(f"❌ Fehler bei der Chat-Engine Abfrage: {str(e)}")
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
        logger.debug(f"✅ Antwort erfolgreich erstellt")

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
            logger.info(f"📝 Debug-Datei erstellt: {debug_file}")
        
        return response
    except Exception as e:
        logger.error(f"❌ Unbehandelter Fehler: {str(e)}")
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
        debug_logger.info(f"🔍 EINGEHENDE REQUEST-DATEN: Chat-ID={request_data.chat_id}, Model={request_data.model}")
        
        # 2. VERARBEITUNGS-INFORMATIONEN
        for key, value in debug_info.items():
            debug_logger.info(f"🔧 {key}: {value}")
        
        # 3. FINALE USER-MESSAGE & ANTWORT
        debug_logger.info(f"✅ USER-MESSAGE: '{user_message_final[:100]}...' (Länge: {len(user_message_final)})")
        
        if llm_response:
            debug_logger.info(f"🤖 RAG-ANTWORT: '{llm_response.response[:100]}...' (Länge: {len(llm_response.response)})")
            
            if hasattr(llm_response, 'source_nodes') and llm_response.source_nodes:
                sources = []
                for i, node in enumerate(llm_response.source_nodes):
                    filename = node.metadata.get('filename', 'Unbekannt')
                    page = node.metadata.get('page_number', 'N/A')
                    sources.append(f"{filename}:{page}")
                debug_logger.info(f"📄 QUELLEN: {', '.join(sources)}")
        else:
            # Meta-Anfrage ohne RAG
            debug_logger.info(f"🎯 META-ANFRAGE: '{response_data.choices[0]['message']['content'][:100]}...'")
        
        debug_logger.info(f"========== ENDE DEBUG-EINTRAG {timestamp} ==========")
        
        # Handler entfernen, um keine doppelten Einträge zu erzeugen
        debug_logger.removeHandler(launcher_log_handler)
        
        return f"debug_{timestamp}"  # Nur für Rückwärtskompatibilität
        
    except Exception as e:
        logger.error(f"❌ Fehler beim Schreiben der Debug-Informationen: {str(e)}")
        return None

# 9. Meta-Anfragen Erkennungs- und Verarbeitungsfunktionen
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
        logger.info(f"🎯 VERARBEITE META-ANFRAGE: {meta_info['type']} (Confidence: {meta_info['confidence']:.2f})")
        
        # Direkter LLM-Call ohne RAG
        llm_response = llm.complete(content)
        response_text = str(llm_response)
        
        logger.debug(f"🤖 LLM-Only Antwort: {response_text[:100]}...")
        return response_text
        
    except Exception as e:
        logger.error(f"❌ Fehler bei Meta-Request LLM-Verarbeitung: {str(e)}")
        # Fallback für Meta-Anfragen
        if meta_info['type'] == 'title':
            return '{ "title": "📋 Geschäftliche Anfrage" }'
        elif meta_info['type'] == 'tags':
            return '{ "tags": ["Business", "General"] }'
        else:
            return "Entschuldigung, ich kann diese Meta-Anfrage derzeit nicht verarbeiten."