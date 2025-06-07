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

# GPU-optimierte Imports für Windows
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# (Weitere Imports ggf. nötig)

# 2. Logging einrichten (Windows-kompatibel)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_api.log', encoding='utf-8')  # UTF-8 für Unicode-Zeichen
    ]
)
logger = logging.getLogger(__name__)

# 2.1 CPU-optimierte LLM Klasse für bessere Performance ohne GPU
class CPUOptimizedLLM:
    """
    CPU-optimierte LLM-Klasse für Systeme mit begrenztem VRAM
    Verwendet CPU-spezifische Optimierungen für bessere Performance
    """
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        logger.info(f"Initialisiere CPU-optimiertes LLM: {model_name}")
        
        # Explizit CPU verwenden
        self.device = "cpu"
        logger.info("Verwende CPU für LLM-Verarbeitung (VRAM-sparend)")
        
        try:
            # Tokenizer laden
            logger.debug("Lade Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # CPU-optimiertes Modell laden (ohne device_map für Pipeline-Kompatibilität)
            logger.debug("Lade Modell für CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # CPU arbeitet besser mit float32
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Weniger RAM-Verbrauch
                # CPU-Optimierungen (Cache deaktiviert wegen DynamicCache-Kompatibilität):
                use_cache=False  # Verhindert DynamicCache-Fehler bei Phi-3
            )
            
            # CPU-optimierte Pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device=-1,  # CPU device für Pipeline
                # CPU-Performance Optimierungen:
                batch_size=1,
                use_fast=True  # Schnellere Tokenizer-Variante
            )
            
            logger.info("CPU-LLM erfolgreich initialisiert!")
            
        except Exception as e:
            logger.error(f"Fehler bei CPU-LLM Initialisierung: {str(e)}")
            raise
    
    def complete(self, prompt, **kwargs):
        """
        CPU-optimierte Completion-Methode mit detailliertem Debugging
        """
        try:
            logger.debug("=== CPU-LLM COMPLETE GESTARTET ===")
            
            # CPU-optimierte Parameter
            max_new_tokens = kwargs.get('max_new_tokens', 800)  # Etwas weniger für CPU
            temperature = kwargs.get('temperature', 0.7)
            
            logger.debug(f"Parameter: max_tokens={max_new_tokens}, temp={temperature}")
            logger.debug(f"Prompt-Länge: {len(prompt)} Zeichen")
            
            # Test: Verwende einen sehr kurzen Prompt zuerst
            if len(prompt) > 2000:
                logger.warning(f"Prompt sehr lang ({len(prompt)} Zeichen), kürze auf 1500...")
                prompt = prompt[:1500] + "\n\nAntwort:"
            
            logger.debug("Starte Pipeline-Aufruf...")
            
            # CPU-Inferenz mit minimal notwendigen Parametern (weniger kann mehr sein)
            result = self.pipeline(
                prompt,
                max_new_tokens=min(max_new_tokens, 300),  # Noch kürzer für Stabilität
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                # Minimale Parameter für maximale Stabilität:
                truncation=True
            )
            
            logger.debug("Pipeline-Aufruf abgeschlossen, extrahiere Ergebnis...")
            
            # Extrahiere die Antwort
            if result and len(result) > 0 and 'generated_text' in result[0]:
                generated_text = result[0]['generated_text']
                response = generated_text.strip() if generated_text else "Keine Antwort generiert"
            else:
                logger.error(f"Unerwartetes Pipeline-Ergebnis: {result}")
                response = "Pipeline-Fehler: Unerwartetes Ergebnis-Format"
            
            logger.debug(f"CPU-LLM Antwort generiert: {len(response)} Zeichen")
            logger.debug("=== CPU-LLM COMPLETE BEENDET ===")
            return response
            
        except Exception as e:
            logger.error(f"Fehler bei CPU-LLM Inferenz: {str(e)}")
            logger.exception("Detaillierter Complete-Fehler:")
            return f"Fehler bei der CPU-Verarbeitung: {str(e)}"

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
    
    # Embedding-Modell CPU-optimiert initialisieren
    logger.debug("Lade Embedding-Modell für CPU...")
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device="cpu"  # CPU für Embeddings (VRAM-sparend)
    )
    
    # CPU-optimiertes LLM initialisieren (Ersetzt GPU-Version)
    logger.debug("Initialisiere CPU-optimiertes LLM...")
    llm = CPUOptimizedLLM(model_name="microsoft/Phi-3-mini-4k-instruct")
    logger.info("CPU-LLM bereit - VRAM-freundlich und stabil!")
    
    # Settings konfigurieren (nur Embeddings, LLM wird direkt verwendet)
    logger.debug("Konfiguriere Settings...")
    Settings.embed_model = embed_model
    # Settings.llm wird NICHT gesetzt - unser WindowsGPULLM ist nicht LlamaIndex-kompatibel
    
    # Vector Store und Index initialisieren
    logger.debug("Initialisiere Vector Store...")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    logger.debug("Erstelle Index...")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # Da unser CPUOptimizedLLM nicht direkt mit LlamaIndex kompatibel ist,
    # verwenden wir den Index direkt für Retrieval und unser LLM für Generation
    logger.debug("Chat Engine umgangen - verwende direktes Retrieval + CPU-LLM...")
    
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
    model: Optional[str] = "microsoft/Phi-3-mini-4k-instruct:latest"  # Geändert zu GPU-Modell mit :latest
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
                logger.warning(f"VERDÄCHTIGE MESSAGE ERKANNT in Message {i+1}: Enthält möglicherweise Tag-Generierung oder andere Prompt-Anweisungen")
        
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
                    logger.info(f"Meta-Request Debug-Datei erstellt: {debug_file}")
                
                logger.info(f"Meta-Anfrage erfolgreich verarbeitet: {meta_info['type']}")
                return response
                
            except Exception as e:
                logger.error(f"❌ Fehler bei Meta-Request Verarbeitung: {str(e)}")
                # Fallback - normale RAG-Verarbeitung
                logger.warning("🔄 Fallback auf normale RAG-Verarbeitung")
        
        # 🔍 NORMALE RAG-VERARBEITUNG (für echte User-Fragen)
        logger.info("NORMALE RAG-VERARBEITUNG")
        
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
        
        logger.debug(f"FINALE USER-MESSAGE für RAG: '{user_message}'")
        
        # Führe die Abfrage mit dem Context Chat Engine durch
        try:
            logger.debug("Starte Chat-Engine Abfrage...")
            
            # CPU-optimierte RAG-Pipeline: Direktes Retrieval + CPU-LLM
            logger.debug("Verwende CPU-optimierte RAG-Pipeline...")
            
            # Hol dir relevante Dokumente über den Retriever
            retriever = index.as_retriever(similarity_top_k=4)
            source_nodes = retriever.retrieve(user_message)
            
            # Erstelle Kontext aus den gefundenen Dokumenten
            context_text = ""
            for i, node in enumerate(source_nodes):
                context_text += f"\n\nDokument {i+1}:\n{node.text}"
            
            # CPU-LLM Call mit Kontext
            system_prompt = "Du bist ein KI-Assistent für Mitarbeiter eines deutschen Planungsbüros im Baugewerbe. Deine Aufgabe ist es, Fragen präzise zu beantworten, indem du Informationen aus den bereitgestellten Dokumenten nutzt. Fasse die relevanten Fakten strukturiert zusammen. Nenne am Ende deiner Antwort immer das Quelldokument und die Seitenzahl. Antworte professionell und auf Deutsch."
            
            full_prompt = f"{system_prompt}\n\nKontext aus Dokumenten:{context_text}\n\nFrage: {user_message}\n\nAntwort:"
            
            logger.info("Starte CPU-LLM Inferenz...")
            logger.debug(f"Prompt-Länge: {len(full_prompt)} Zeichen")
            
            # Detailliertes Debug-Logging für LLM-Pipeline
            try:
                llm_response_direct = llm.complete(full_prompt)
                logger.info("CPU-LLM Inferenz abgeschlossen")
                response_text = str(llm_response_direct)
                logger.debug(f"Antwort-Länge: {len(response_text)} Zeichen")
            except Exception as e:
                logger.error(f"FEHLER bei CPU-LLM Inferenz: {str(e)}")
                logger.exception("Detaillierter LLM-Fehler:")
                raise
            
            logger.debug(f"Direkte CPU-LLM-Antwort erhalten: {response_text[:100]}...")
            
            # Erstelle ein Mock-Response-Objekt für Kompatibilität
            class MockLLMResponse:
                def __init__(self, response, source_nodes):
                    self.response = response
                    self.source_nodes = source_nodes
            
            llm_response = MockLLMResponse(response_text, source_nodes)
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
            logger.error(f"Fehler bei der CPU-RAG Pipeline: {str(e)}")
            logger.exception("Detaillierter CPU-RAG Fehler:")
            raise HTTPException(status_code=500, detail=f"CPU-RAG Fehler: {str(e)}")
        
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
            logger.info(f"Debug-Datei erstellt: {debug_file}")
        
        return response
    except Exception as e:
        logger.error(f"❌ Unbehandelter Fehler: {str(e)}")
        logger.error(f"Request Details: {request.dict() if request else 'No request data'}")
        logger.exception("Detaillierter unbehandelter Fehler:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    cpu_count = os.cpu_count()
    return {
        "status": "healthy", 
        "model": "microsoft/Phi-3-mini-4k-instruct:latest",
        "device": f"CPU ({cpu_count} Kerne)",
        "cpu_optimized": True,
        "vram_usage": "0 GB (CPU-Modus)"
    }

# Neue Ollama-kompatible Endpunkte für OpenWebUI
@app.get("/api/tags")
async def get_models():
    """
    Ollama-kompatibler Endpunkt für OpenWebUI - listet verfügbare Modelle auf
    """
    try:
        logger.debug("OpenWebUI fragt verfügbare Modelle ab (/api/tags)")
        return {
            "models": [
                {
                    "name": "microsoft/Phi-3-mini-4k-instruct:latest",
                    "model": "microsoft/Phi-3-mini-4k-instruct:latest", 
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 3800000000,  # ~3.8GB für Phi-3-mini
                    "digest": "phi3mini4k",
                    "details": {
                        "parent_model": "",
                        "format": "transformers",
                        "family": "phi-3",
                        "families": ["phi-3"],
                        "parameter_size": "3.8B",
                        "device": "GPU" if torch.cuda.is_available() else "CPU"
                    }
                }
            ]
        }
    except Exception as e:
        logger.error(f"Fehler bei /api/tags: {str(e)}")
        return {"models": []}

@app.get("/api/version")
async def get_version():
    """
    Ollama-kompatibler Endpunkt für OpenWebUI - zeigt Version
    """
    try:
        logger.debug("OpenWebUI fragt Version ab (/api/version)")
        return {"version": "0.1.0-gpu"}
    except Exception as e:
        logger.error(f"Fehler bei /api/version: {str(e)}")
        return {"version": "0.1.0-gpu"}

@app.get("/api/ps")
async def get_running_models():
    """
    Ollama-kompatibler Endpunkt für OpenWebUI - zeigt laufende Modelle
    """
    try:
        logger.debug("OpenWebUI fragt laufende Modelle ab (/api/ps)")
        return {
            "models": [
                {
                    "name": "microsoft/Phi-3-mini-4k-instruct:latest",
                    "model": "microsoft/Phi-3-mini-4k-instruct:latest",
                    "size": 3800000000,
                    "digest": "phi3mini4k",
                    "expires_at": "2024-12-31T23:59:59Z",
                    "device": "GPU" if torch.cuda.is_available() else "CPU"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Fehler bei /api/ps: {str(e)}")
        return {"models": []}

# Ollama-Chat-Endpunkt für OpenWebUI
class OllamaChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    options: Optional[dict] = {}

@app.post("/api/chat")
async def ollama_chat(request: OllamaChatRequest):
    """
    Ollama-kompatibler Chat-Endpunkt für OpenWebUI
    Konvertiert Ollama-Format zu unserem internen ChatRequest und zurück
    """
    try:
        logger.debug(f"=== OLLAMA-CHAT ANFRAGE ===")
        logger.debug(f"Model: {request.model}")
        logger.debug(f"Stream: {request.stream}")
        logger.debug(f"Messages: {len(request.messages)}")
        
        # Konvertiere Ollama-Request zu unserem internen ChatRequest-Format
        internal_request = ChatRequest(
            messages=request.messages,
            model=request.model,
            temperature=request.options.get('temperature', 0.7) if request.options else 0.7,
            max_tokens=request.options.get('max_tokens', 1000) if request.options else 1000
        )
        
        # Verwende die bestehende Chat-Logik
        internal_response = await chat_completions(internal_request)
        
        # Konvertiere zurück zu Ollama-Format
        assistant_message = internal_response.choices[0]['message']['content']
        
        if request.stream:
            # Streaming-Response für Ollama (vereinfacht)
            return {
                "model": request.model,
                "created_at": datetime.now().isoformat() + "Z",
                "message": {
                    "role": "assistant", 
                    "content": assistant_message
                },
                "done": True
            }
        else:
            # Standard Ollama-Response
            return {
                "model": request.model,
                "created_at": datetime.now().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": assistant_message
                },
                "done": True,
                "total_duration": 1000000000,  # Dummy-Werte
                "load_duration": 100000000,
                "prompt_eval_duration": 200000000,
                "eval_duration": 700000000,
                "eval_count": 50
            }
            
    except Exception as e:
        logger.error(f"❌ Fehler bei Ollama-Chat: {str(e)}")
        logger.exception("Detaillierter Ollama-Chat Fehler:")
        raise HTTPException(status_code=500, detail=f"Ollama-Chat Fehler: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "RAG API ist online",
        "endpoints": {
            "/v1/chat/completions": "Chat-Endpunkt (OpenAI-kompatibel)",
            "/api/tags": "Modelliste (Ollama-kompatibel)",
            "/api/ps": "Laufende Modelle (Ollama-kompatibel)",
            "/health": "System-Status"
        }
    }

# 7. Startbefehl für uvicorn (wird in der Kommandozeile ausgeführt)
# uvicorn rag_api:app --host 0.0.0.0 --port 8000 --reload

# 8. Debug-Datei Funktion
def create_debug_file(request_data, user_message_final, llm_response, response_data, debug_info):
    """
    Erstellt eine Debug-Datei im debug_logs Verzeichnis mit allen Informationen zur Anfrage und Antwort
    """
    try:
        # Stelle sicher, dass der debug_logs Ordner existiert
        debug_dir = "debug_logs"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        debug_filename = os.path.join(debug_dir, f"debug_{timestamp}.txt")
        
        with open(debug_filename, 'w', encoding='utf-8') as f:
            # 1. EINGEHENDE REQUEST-DATEN
            f.write(f"========== DEBUG-EINTRAG {timestamp} ==========\n\n")
            f.write(f"🔍 EINGEHENDE REQUEST-DATEN:\n")
            f.write(f"  Chat-ID: {request_data.chat_id}\n")
            f.write(f"  Model: {request_data.model}\n")
            f.write(f"  Temperature: {request_data.temperature}\n")
            f.write(f"  Max Tokens: {request_data.max_tokens}\n\n")
            
            # Eingehende Messages
            f.write(f"📩 EINGEHENDE MESSAGES ({len(request_data.messages)}):\n")
            for i, msg in enumerate(request_data.messages):
                f.write(f"  Message {i+1} (Role: {msg.role}):\n")
                f.write(f"  {msg.content[:500]}{'...' if len(msg.content) > 500 else ''}\n\n")
            
            # 2. VERARBEITUNGS-INFORMATIONEN
            f.write(f"🔧 VERARBEITUNGS-INFORMATIONEN:\n")
            for key, value in debug_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 3. FINALE USER-MESSAGE & ANTWORT
            f.write(f"✅ FINALE USER-MESSAGE:\n")
            f.write(f"  {user_message_final}\n\n")
            
            if llm_response:
                f.write(f"🤖 RAG-ANTWORT:\n")
                f.write(f"  {llm_response.response}\n\n")
                
                if hasattr(llm_response, 'source_nodes') and llm_response.source_nodes:
                    f.write(f"📄 QUELLEN:\n")
                    for i, node in enumerate(llm_response.source_nodes):
                        filename = node.metadata.get('filename', 'Unbekannt')
                        page = node.metadata.get('page_number', 'N/A')
                        score = node.score if hasattr(node, 'score') else 0.0
                        f.write(f"  Source {i+1}: File='{filename}', Page='{page}', Score={score:.4f}\n")
                        f.write(f"  Vorschau: {node.text[:100].replace(chr(10), ' ')}...\n\n")
            else:
                # Meta-Anfrage ohne RAG
                f.write(f"🎯 META-ANFRAGE ANTWORT:\n")
                f.write(f"  {response_data.choices[0]['message']['content']}\n\n")
            
            f.write(f"========== ENDE DEBUG-EINTRAG {timestamp} ==========\n")
        
        logger.info(f"📝 Debug-Datei erstellt: {debug_filename}")
        return debug_filename
        
    except Exception as e:
        logger.error(f"❌ Fehler beim Erstellen der Debug-Datei: {str(e)}")
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