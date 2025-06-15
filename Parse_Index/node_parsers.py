"""
LlamaIndex Node Parser für das Parse_Index System mit Docling-Integration

Dieses Modul enthält die verschiedenen Node-Parser für die Aufbereitung 
von Dokumenten in optimale Chunks für das Retrieval-System.

Parser-Strategien:
- Docling Native: Nutzt DoclingNodeParser für beste Docling-Integration
- Semantic Section Parser: Semantische Abschnitte als Hauptstrategie
- Hierarchical Parser: Backup für verschiedene Chunk-Größen  
- Sentence Window Parser: Satz-basierte Chunks mit Kontext-Fenstern
"""

from typing import Dict, Any, List, Optional
from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceWindowNodeParser, 
    SentenceSplitter, 
    HierarchicalNodeParser,
    MarkdownNodeParser
)

# Docling-Integration
try:
    from llama_index.node_parser.docling import DoclingNodeParser
    DOCLING_NODE_PARSER_AVAILABLE = True
except ImportError:
    DoclingNodeParser = None
    DOCLING_NODE_PARSER_AVAILABLE = False

from .config import CHUNK_SIZES_CONFIG

def create_docling_native_parser():
    """
    Erstellt einen nativen Docling Node Parser für optimale Docling-Integration.
    
    Dies ist die beste Strategie für Docling-verarbeitete PDFs, da sie
    die reichen Metadaten (Bounding-Boxes, Tabellen-HTML, etc.) optimal nutzt.
    
    Returns:
        DoclingNodeParser: Konfigurierter Docling-Parser oder None falls nicht verfügbar
    """
    if not DOCLING_NODE_PARSER_AVAILABLE:
        return None
    
    return DoclingNodeParser()

def create_docling_optimized_markdown_parser():
    """
    Erstellt einen für Docling-Markdown optimierten Parser.
    
    Für Fälle wo Docling im Markdown-Export-Modus verwendet wird.
    
    Returns:
        MarkdownNodeParser: Konfigurierter Markdown-Parser
    """
    return MarkdownNodeParser(
        include_metadata=True,
        include_prev_next_rel=True
    )

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
    - Hauptstrategie: Docling Native für beste Docling-Integration
    - Backup: Semantische Abschnitte für beste Retrieval-Qualität
    - Alternativen: Hierarchische Struktur und verschiedene Chunk-Größen
    
    Returns:
        dict: Dictionary mit verschiedenen konfigurierten Parsern
    """
    parsers = {}
    
    # Hauptstrategie: Docling Native (falls verfügbar)
    if DOCLING_NODE_PARSER_AVAILABLE:
        parsers['docling_native'] = create_docling_native_parser()
        parsers['main'] = parsers['docling_native']  # Alias für Hauptstrategie
    else:
        # Fallback: Semantische Abschnitte
        parsers['main'] = create_semantic_section_parser()
    
    # Docling-optimierte Parser
    parsers['docling_markdown'] = create_docling_optimized_markdown_parser()
    
    # Backup-Strategien für verschiedene Anwendungsfälle
    parsers['semantic'] = create_semantic_section_parser()
    parsers['hierarchical'] = create_hierarchical_backup_parser()
    parsers['sentence_window'] = create_sentence_window_parser()
    
    # Spezielle Parser für verschiedene Anforderungen  
    parsers['fine_grained'] = create_fine_grained_splitter()
    parsers['context_aware'] = create_context_aware_splitter()
    
    # Standard-Fallback
    parsers['fallback'] = SentenceSplitter(
        chunk_size=CHUNK_SIZES_CONFIG["chunk_size"],
        chunk_overlap=CHUNK_SIZES_CONFIG["chunk_overlap"]
    )
    
    return parsers

def get_parser_by_strategy(strategy: str):
    """
    Gibt einen spezifischen Parser basierend auf der gewählten Strategie zurück.
    
    Args:
        strategy: Parser-Strategie ('docling_native', 'docling_markdown', 'semantic', 
                 'hierarchical', 'sentence_window', 'fine_grained', 'context_aware', 'fallback')
                 
    Returns:
        Parser-Objekt entsprechend der gewählten Strategie
        
    Raises:
        ValueError: Wenn die Strategie nicht unterstützt wird
    """
    parser_system = create_hybrid_parser_system()
    
    strategy_mapping = {
        'docling_native': 'docling_native',
        'docling_markdown': 'docling_markdown',
        'main': 'main',
        'semantic': 'semantic',
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
    
    parser_key = strategy_mapping[strategy]
    
    # Prüfe ob Parser verfügbar ist
    if parser_key not in parser_system or parser_system[parser_key] is None:
        if strategy == 'docling_native':
            raise ValueError(
                "DoclingNodeParser nicht verfügbar. "
                "Installiere mit: pip install llama-index-node-parser-docling"
            )
        else:
            # Fallback auf semantischen Parser
            return parser_system['semantic']
    
    return parser_system[parser_key]

def get_optimal_parser_for_documents(documents: List[Document]):
    """
    Bestimmt den optimalen Parser basierend auf den Document-Metadaten.
    
    Analysiert die Dokumente und wählt den besten Parser:
    - Docling JSON → DoclingNodeParser
    - Docling Markdown → MarkdownNodeParser  
    - Andere → Semantischer Parser
    
    Args:
        documents: Liste von LlamaIndex Documents
        
    Returns:
        Optimaler Parser für die gegebenen Dokumente
    """
    if not documents:
        return get_parser_by_strategy('fallback')
    
    # Analysiere erste Document-Metadaten
    first_doc = documents[0]
    metadata = first_doc.metadata
    
    # Prüfe auf Docling-spezifische Metadaten
    if 'schema_name' in metadata and 'docling' in metadata.get('schema_name', '').lower():
        # Docling JSON-Format erkannt
        if DOCLING_NODE_PARSER_AVAILABLE:
            return get_parser_by_strategy('docling_native')
        else:
            return get_parser_by_strategy('semantic')
    
    # Prüfe auf Markdown-Export
    if metadata.get('export_type') == 'MARKDOWN' or 'markdown' in metadata.get('export_type', '').lower():
        return get_parser_by_strategy('docling_markdown')
    
    # Prüfe auf Tabellen-reiche Dokumente
    if metadata.get('has_tables') or metadata.get('table_count', 0) > 0:
        # Für Tabellen ist Docling Native optimal
        if DOCLING_NODE_PARSER_AVAILABLE:
            return get_parser_by_strategy('docling_native')
        else:
            return get_parser_by_strategy('context_aware')  # Größere Chunks für Tabellen
    
    # Standard: Semantischer Parser
    return get_parser_by_strategy('semantic')

def get_parser_info():
    """
    Gibt Informationen über alle verfügbaren Parser zurück.
    
    Returns:
        dict: Detaillierte Informationen über alle Parser-Strategien
    """
    info = {
        'docling_native': {
            'name': 'Docling Native',
            'description': 'Nutzt DoclingNodeParser für optimale Docling-Integration',
            'available': DOCLING_NODE_PARSER_AVAILABLE,
            'use_case': 'Beste Wahl für Docling JSON-Export mit reichen Metadaten',
            'features': ['Bounding-Boxes', 'Tabellen-HTML', 'Hierarchie-Kontext']
        },
        'docling_markdown': {
            'name': 'Docling Markdown',
            'description': 'Optimiert für Docling Markdown-Export',
            'available': True,
            'use_case': 'Für Docling Markdown-Export mit Überschriften-Struktur'
        },
        'semantic': {
            'name': 'Semantische Abschnitte',
            'description': 'Behandelt PDF-Abschnitte als ganze Einheiten für beste Retrieval-Qualität',
            'chunk_size': CHUNK_SIZES_CONFIG["chunk_size"],
            'overlap': CHUNK_SIZES_CONFIG["chunk_overlap"],
            'available': True,
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
            'available': True,
            'use_case': 'Backup für komplexe Dokumente mit verschiedenen Strukturebenen'
        },
        'sentence_window': {
            'name': 'Satz-Fenster',
            'description': 'Satz-basierte Chunks mit erweiterten Kontext-Fenstern',
            'window_size': CHUNK_SIZES_CONFIG["window_size"],
            'available': True,
            'use_case': 'Für präzise, satz-spezifische Abfragen'
        },
        'fine_grained': {
            'name': 'Fein-granular',
            'description': 'Kleine Chunks für detaillierte Analyse',
            'chunk_size': CHUNK_SIZES_CONFIG["chunk_size_small"],
            'overlap': CHUNK_SIZES_CONFIG["chunk_overlap"],
            'available': True,
            'use_case': 'Für sehr spezifische oder detaillierte Informationen'
        },
        'context_aware': {
            'name': 'Kontext-bewusst',
            'description': 'Große Chunks mit viel Kontext',
            'chunk_size': CHUNK_SIZES_CONFIG["chunk_size_large"],
            'overlap': int(CHUNK_SIZES_CONFIG["chunk_overlap"] * 1.5),
            'available': True,
            'use_case': 'Für komplexe Regelwerke oder umfangreiche Dokumentationen'
        },
        'fallback': {
            'name': 'Standard-Fallback',
            'description': 'Standard Sentence Splitter als sicherer Fallback',
            'chunk_size': CHUNK_SIZES_CONFIG["chunk_size"],
            'overlap': CHUNK_SIZES_CONFIG["chunk_overlap"],
            'available': True,
            'use_case': 'Fallback wenn andere Parser fehlschlagen'
        }
    }
    
    return info

def is_docling_node_parser_available() -> bool:
    """
    Prüft, ob der DoclingNodeParser verfügbar ist.
    
    Returns:
        bool: True wenn DoclingNodeParser verfügbar ist
    """
    return DOCLING_NODE_PARSER_AVAILABLE

# Convenience-Funktionen für einfache Nutzung
def parse_documents_with_optimal_strategy(documents: List[Document]) -> List:
    """
    Parst Dokumente mit der optimalen Strategie basierend auf ihren Metadaten.
    
    Args:
        documents: Liste von LlamaIndex Documents
        
    Returns:
        Liste von Nodes
    """
    parser = get_optimal_parser_for_documents(documents)
    return parser.get_nodes_from_documents(documents)

def parse_documents_with_strategy(documents: List[Document], strategy: str) -> List:
    """
    Parst Dokumente mit einer spezifischen Strategie.
    
    Args:
        documents: Liste von LlamaIndex Documents
        strategy: Parser-Strategie
        
    Returns:
        Liste von Nodes
    """
    parser = get_parser_by_strategy(strategy)
    return parser.get_nodes_from_documents(documents) 