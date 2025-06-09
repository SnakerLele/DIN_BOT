"""
LlamaIndex Node Parser für das Parse_Index System

Dieses Modul enthält die verschiedenen Node-Parser für die Aufbereitung 
von Dokumenten in optimale Chunks für das Retrieval-System.

Parser-Strategien:
- Semantic Section Parser: Semantische Abschnitte als Hauptstrategie
- Hierarchical Parser: Backup für verschiedene Chunk-Größen  
- Sentence Window Parser: Satz-basierte Chunks mit Kontext-Fenstern
"""

from typing import Dict, Any
from llama_index.core.node_parser import (
    SentenceWindowNodeParser, 
    SentenceSplitter, 
    HierarchicalNodeParser
)
from .config import CHUNK_SIZES_CONFIG

def create_semantic_section_parser():
    """
    Erstellt einen semantischen Parser, der PDF-Abschnitte als ganze Einheiten behandelt.
    
    Diese Strategie liefert die beste Performance für Spielregeln und strukturierte Dokumente,
    da zusammengehörige Inhalte nicht unnötig getrennt werden.
    
    Returns:
        SentenceSplitter: Konfigurierter Parser für semantische Abschnitte
    """
    return SentenceSplitter(
        chunk_size=CHUNK_SIZES_CONFIG["chunk_size"],
        chunk_overlap=CHUNK_SIZES_CONFIG["chunk_overlap"],
        include_metadata=True,
        include_prev_next_rel=True
    )

def create_sentence_window_parser():
    """
    Erstellt einen Sentence Window Node Parser für satz-basierte Chunks.
    
    Jeder Satz wird als separater Node gespeichert, aber mit einem erweiterten
    Kontext-Fenster für bessere semantische Verbindungen.
    
    Returns:
        SentenceWindowNodeParser: Konfigurierter Sentence Window Parser
    """
    return SentenceWindowNodeParser.from_defaults(
        window_size=CHUNK_SIZES_CONFIG["window_size"],
        window_metadata_key="window",
        original_text_metadata_key="original_sentence",
        include_metadata=True,
        include_prev_next_rel=True
    )

def create_hierarchical_backup_parser():
    """
    Erstellt einen hierarchischen Backup-Parser für zusätzliche Struktur.
    
    Wird verwendet um längere Chunks für bestimmte Anwendungsfälle zu erstellen,
    wenn die semantischen Abschnitte zu groß oder zu klein sind.
    
    Returns:
        HierarchicalNodeParser: Konfigurierter hierarchischer Parser
    """
    # Chunk-Größen in absteigender Reihenfolge für hierarchische Struktur
    sizes_list = [
        CHUNK_SIZES_CONFIG["chunk_size_large"],  # 2048 Zeichen
        CHUNK_SIZES_CONFIG["chunk_size"],        # 1024 Zeichen  
        CHUNK_SIZES_CONFIG["chunk_size_small"]   # 512 Zeichen
    ]
    
    return HierarchicalNodeParser.from_defaults(
        chunk_sizes=sizes_list,
        chunk_overlap=CHUNK_SIZES_CONFIG["chunk_overlap"],
        include_metadata=True,
        include_prev_next_rel=True
    )

def create_fine_grained_splitter():
    """
    Erstellt einen fein-granularen Splitter für detaillierte Analyse.
    
    Verwendet kleinere Chunks für Anwendungsfälle, die sehr spezifische
    oder detaillierte Informationen benötigen.
    
    Returns:
        SentenceSplitter: Konfigurierter fine-grained Parser
    """
    return SentenceSplitter(
        chunk_size=CHUNK_SIZES_CONFIG["chunk_size_small"],
        chunk_overlap=CHUNK_SIZES_CONFIG["chunk_overlap"],
        include_metadata=True,
        include_prev_next_rel=True
    )

def create_context_aware_splitter():
    """
    Erstellt einen kontext-bewussten Splitter für große Dokumente.
    
    Verwendet größere Chunks um mehr Kontext zu bewahren, ideal für
    komplexe Regelwerke oder umfangreiche Dokumentationen.
    
    Returns:
        SentenceSplitter: Konfigurierter context-aware Parser
    """
    return SentenceSplitter(
        chunk_size=CHUNK_SIZES_CONFIG["chunk_size_large"],
        chunk_overlap=int(CHUNK_SIZES_CONFIG["chunk_overlap"] * 1.5),  # Mehr Überlappung
        include_metadata=True,
        include_prev_next_rel=True
    )

def create_hybrid_parser_system():
    """
    Erstellt ein hybrides Parser-System mit verschiedenen Strategien.
    
    Das System kombiniert verschiedene Parser-Ansätze für optimale Flexibilität:
    - Hauptstrategie: Semantische Abschnitte für beste Retrieval-Qualität
    - Backup: Hierarchische Struktur für verschiedene Anwendungsfälle
    - Alternativen: Sentence Window und verschiedene Chunk-Größen
    
    Returns:
        dict: Dictionary mit verschiedenen konfigurierten Parsern
    """
    return {
        # Hauptstrategie: Semantische Abschnitte
        'main': create_semantic_section_parser(),
        
        # Backup-Strategien für verschiedene Anwendungsfälle
        'hierarchical': create_hierarchical_backup_parser(),
        'sentence_window': create_sentence_window_parser(),
        
        # Spezielle Parser für verschiedene Anforderungen  
        'fine_grained': create_fine_grained_splitter(),
        'context_aware': create_context_aware_splitter(),
        
        # Standard-Fallback
        'fallback': SentenceSplitter(
            chunk_size=CHUNK_SIZES_CONFIG["chunk_size"],
            chunk_overlap=CHUNK_SIZES_CONFIG["chunk_overlap"]
        )
    }

def get_parser_by_strategy(strategy: str):
    """
    Gibt einen spezifischen Parser basierend auf der gewählten Strategie zurück.
    
    Args:
        strategy: Parser-Strategie ('semantic', 'hierarchical', 'sentence_window', 
                 'fine_grained', 'context_aware', 'fallback')
                 
    Returns:
        Parser-Objekt entsprechend der gewählten Strategie
        
    Raises:
        ValueError: Wenn die Strategie nicht unterstützt wird
    """
    parser_system = create_hybrid_parser_system()
    
    strategy_mapping = {
        'semantic': 'main',
        'hierarchical': 'hierarchical', 
        'sentence_window': 'sentence_window',
        'fine_grained': 'fine_grained',
        'context_aware': 'context_aware',
        'fallback': 'fallback'
    }
    
    if strategy not in strategy_mapping:
        available_strategies = list(strategy_mapping.keys())
        raise ValueError(f"Unbekannte Parser-Strategie: {strategy}. "
                        f"Verfügbare Strategien: {available_strategies}")
    
    return parser_system[strategy_mapping[strategy]]

def get_parser_info():
    """
    Gibt Informationen über alle verfügbaren Parser zurück.
    
    Returns:
        dict: Detaillierte Informationen über alle Parser-Strategien
    """
    return {
        'semantic': {
            'name': 'Semantische Abschnitte',
            'description': 'Behandelt PDF-Abschnitte als ganze Einheiten für beste Retrieval-Qualität',
            'chunk_size': CHUNK_SIZES_CONFIG["chunk_size"],
            'overlap': CHUNK_SIZES_CONFIG["chunk_overlap"],
            'use_case': 'Hauptstrategie für strukturierte Dokumente wie Spielregeln'
        },
        'hierarchical': {
            'name': 'Hierarchische Struktur',
            'description': 'Erstellt mehrere Chunk-Ebenen für verschiedene Detailgrade',
            'chunk_sizes': [
                CHUNK_SIZES_CONFIG["chunk_size_large"],
                CHUNK_SIZES_CONFIG["chunk_size"], 
                CHUNK_SIZES_CONFIG["chunk_size_small"]
            ],
            'overlap': CHUNK_SIZES_CONFIG["chunk_overlap"],
            'use_case': 'Backup für komplexe Dokumente mit verschiedenen Strukturebenen'
        },
        'sentence_window': {
            'name': 'Satz-Fenster',
            'description': 'Satz-basierte Chunks mit erweiterten Kontext-Fenstern',
            'window_size': CHUNK_SIZES_CONFIG["window_size"],
            'use_case': 'Für präzise, satz-spezifische Abfragen'
        },
        'fine_grained': {
            'name': 'Fein-granular',
            'description': 'Kleine Chunks für detaillierte Analyse',
            'chunk_size': CHUNK_SIZES_CONFIG["chunk_size_small"],
            'overlap': CHUNK_SIZES_CONFIG["chunk_overlap"],
            'use_case': 'Für sehr spezifische oder detaillierte Informationen'
        },
        'context_aware': {
            'name': 'Kontext-bewusst',
            'description': 'Große Chunks mit viel Kontext',
            'chunk_size': CHUNK_SIZES_CONFIG["chunk_size_large"],
            'overlap': int(CHUNK_SIZES_CONFIG["chunk_overlap"] * 1.5),
            'use_case': 'Für komplexe Regelwerke oder umfangreiche Dokumentationen'
        },
        'fallback': {
            'name': 'Standard-Fallback',
            'description': 'Standard Sentence Splitter als sicherer Fallback',
            'chunk_size': CHUNK_SIZES_CONFIG["chunk_size"],
            'overlap': CHUNK_SIZES_CONFIG["chunk_overlap"],
            'use_case': 'Fallback wenn andere Parser fehlschlagen'
        }
    } 