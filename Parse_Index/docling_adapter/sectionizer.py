"""
DoclingChunker
--------------
Schlanker Wrapper um Doclings native Chunker-Funktionalität.

Nutzt direkt Doclings Chunker-Klassen anstatt eigene Heuristiken:
- HeadingChunker: Chunking basierend auf Überschriften-Hierarchie  
- PageChunker: Ein Chunk pro Seite
- HierarchicalChunker: Intelligente Kombination verschiedener Strategien
- ElementChunker: Jedes Element als eigener Chunk
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Konfiguration für Chunking-Parameter."""
    max_chunk_size: int = 1024  # Optimiert für Embedding-Modelle
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    preserve_tables: bool = True
    preserve_images: bool = True
    
    def __post_init__(self):
        # Validierung der Parameter
        if self.max_chunk_size < self.min_chunk_size:
            raise ValueError("max_chunk_size muss größer als min_chunk_size sein")
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError("chunk_overlap muss kleiner als max_chunk_size sein")


class DoclingChunker:
    """
    Schlanker Wrapper um Doclings native Chunker-Funktionalität.
    
    Nutzt direkt Doclings Chunker-Klassen für optimale Ergebnisse.
    """
    
    def __init__(self, strategy: str = "hybrid", config: Optional[ChunkConfig] = None):
        """
        Initialisiert den DoclingChunker.
        
        Args:
            strategy: Chunking-Strategie ("by_heading", "by_page", "hybrid", "by_element")
            config: Chunking-Konfiguration
        """
        self.strategy = strategy
        self.config = config or ChunkConfig()
        self.logger = logging.getLogger(__name__)
        
        # Validiere Strategie
        valid_strategies = ["by_heading", "by_page", "hybrid", "by_element"]
        if strategy not in valid_strategies:
            raise ValueError(f"Ungültige Strategie: {strategy}. Gültig: {valid_strategies}")
        
        # Initialisiere Docling-Chunker
        self.chunker = self._create_docling_chunker()
    
    def _create_docling_chunker(self):
        """
        Erstellt den passenden Docling-Chunker basierend auf der Strategie.
        
        Returns:
            Docling Chunker-Instanz oder None bei Fallback
        """
        try:
            # Versuche Docling-Chunker zu importieren
            if self.strategy == "by_heading":
                from docling.chunking import HierarchicalChunker
                return HierarchicalChunker(
                    max_tokens=self.config.max_chunk_size,
                    overlap_tokens=self.config.chunk_overlap
                )
            elif self.strategy == "by_page":
                # PageChunker existiert nicht - verwende HybridChunker
                from docling.chunking import HybridChunker
                return HybridChunker()
            elif self.strategy == "by_element":
                # ElementChunker existiert nicht - verwende BaseChunker
                from docling.chunking import BaseChunker
                return BaseChunker()
            else:  # hybrid
                from docling.chunking import HybridChunker
                return HybridChunker()
                
        except ImportError as e:
            self.logger.warning(f"Docling-Chunker nicht verfügbar: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Fehler beim Erstellen des Docling-Chunkers: {e}")
            return None
    
    def split(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Hauptmethode: Zerlegt ein Docling-Document in semantische Blöcke.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            Liste von Blöcken mit Text und Metadaten
        """
        try:
            self.logger.debug(f"Starte Chunking mit Strategie: {self.strategy}")
            
            # Nutze Docling-Chunker wenn verfügbar
            if self.chunker is not None:
                return self._chunk_with_docling(docling_doc)
            else:
                # Fallback: Einfache Implementierung
                return self._chunk_fallback(docling_doc)
                
        except Exception as e:
            self.logger.error(f"Fehler beim Chunking: {str(e)}")
            return self._create_fallback_chunks(docling_doc)
    
    def _chunk_with_docling(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Nutzt Doclings native Chunker für optimale Ergebnisse.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            Liste von Chunks mit Text und Metadaten
        """
        try:
            # Docling-Chunker anwenden
            chunks = list(self.chunker.chunk(docling_doc))
            
            # Chunks in unser Format konvertieren
            blocks = []
            for i, chunk in enumerate(chunks):
                # Chunk-Text extrahieren
                chunk_text = getattr(chunk, 'text', str(chunk))
                
                # Metadaten aus Chunk extrahieren
                chunk_meta = getattr(chunk, 'meta', {})
                
                # Unser Standard-Metadaten-Format erstellen
                metadata = {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_strategy": self.strategy,
                    "text_length": len(chunk_text),
                    "docling_chunker": True,
                }
                
                # Docling-Metadaten hinzufügen
                if chunk_meta:
                    metadata.update(chunk_meta)
                
                # Seiten-Information extrahieren (falls verfügbar)
                if hasattr(chunk, 'page_number'):
                    metadata["page_number"] = chunk.page_number
                elif hasattr(chunk, 'page'):
                    metadata["page_number"] = chunk.page
                else:
                    metadata["page_number"] = 1
                
                # Überschriften-Kontext (falls verfügbar)
                if hasattr(chunk, 'heading_context'):
                    context = chunk.heading_context
                    metadata["section_h1"] = context.get("h1")
                    metadata["section_h2"] = context.get("h2") 
                    metadata["section_h3"] = context.get("h3")
                
                # Element-Typ (falls verfügbar)
                if hasattr(chunk, 'element_type'):
                    metadata["docling_type"] = chunk.element_type
                elif hasattr(chunk, 'type'):
                    metadata["docling_type"] = chunk.type
                else:
                    metadata["docling_type"] = "text"
                
                blocks.append({
                    "text": chunk_text,
                    "metadata": metadata
                })
            
            self.logger.info(f"Docling-Chunking erfolgreich: {len(blocks)} Chunks")
            return blocks
            
        except Exception as e:
            self.logger.error(f"Docling-Chunking fehlgeschlagen: {e}")
            return self._chunk_fallback(docling_doc)
    
    def _chunk_fallback(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Einfache Fallback-Implementierung wenn Docling-Chunker nicht verfügbar.
        
        Extrahiert Text seitenweise und teilt bei Bedarf auf.
        """
        self.logger.warning("Verwende Fallback-Chunking (Docling-Chunker nicht verfügbar)")
        
        blocks = []
        pages = getattr(docling_doc, 'pages', [])
        
        if not pages:
            return self._create_fallback_chunks(docling_doc)
        
        for page_num, page in enumerate(pages, start=1):
            # Text von der Seite extrahieren
            page_text = self._extract_page_text(page)
            
            if not page_text.strip():
                continue
            
            # Bei großen Seiten aufteilen
            if len(page_text) > self.config.max_chunk_size:
                chunks = self._simple_text_split(page_text)
                
                for i, chunk in enumerate(chunks):
                    blocks.append({
                        "text": chunk,
                        "metadata": {
                            "page_number": page_num,
                            "chunk_index": len(blocks),
                            "page_chunk": i + 1,
                            "total_page_chunks": len(chunks),
                            "chunk_strategy": f"{self.strategy}_fallback",
                            "text_length": len(chunk),
                            "docling_chunker": False,
                            "is_fallback": True,
                        }
                    })
            else:
                blocks.append({
                    "text": page_text,
                    "metadata": {
                        "page_number": page_num,
                        "chunk_index": len(blocks),
                        "chunk_strategy": f"{self.strategy}_fallback",
                        "text_length": len(page_text),
                        "docling_chunker": False,
                        "is_fallback": True,
                    }
                })
        
        self.logger.info(f"Fallback-Chunking: {len(blocks)} Chunks")
        return blocks
    
    def _extract_page_text(self, page: Any) -> str:
        """Extrahiert Text von einer Seite."""
        text_parts = []
        
        elements = getattr(page, 'elements', [])
        for element in elements:
            element_text = getattr(element, 'text', '').strip()
            if element_text:
                text_parts.append(element_text)
        
        return '\n\n'.join(text_parts)
    
    def _simple_text_split(self, text: str) -> List[str]:
        """Einfache Text-Aufteilung bei zu großen Chunks."""
        if len(text) <= self.config.max_chunk_size:
            return [text]
        
        chunks = []
        remaining = text
        
        while len(remaining) > self.config.max_chunk_size:
            # Suche nach Absatzgrenze
            split_pos = remaining.rfind('\n\n', 0, self.config.max_chunk_size)
            
            if split_pos == -1:
                # Suche nach Satzgrenze
                split_pos = remaining.rfind('. ', 0, self.config.max_chunk_size)
                if split_pos != -1:
                    split_pos += 2
            
            if split_pos == -1:
                # Hart bei max_size trennen
                split_pos = self.config.max_chunk_size
            
            chunk = remaining[:split_pos].strip()
            if chunk and len(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
            
            # Mit Overlap weitermachen
            overlap_start = max(0, split_pos - self.config.chunk_overlap)
            remaining = remaining[overlap_start:].strip()
            
            # Endlosschleife verhindern
            if split_pos == 0:
                break
        
        # Letzten Teil hinzufügen
        if remaining.strip() and len(remaining) >= self.config.min_chunk_size:
            chunks.append(remaining.strip())
        
        return chunks
    
    def _create_fallback_chunks(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """Erstellt minimale Fallback-Chunks."""
        source_path = getattr(docling_doc, 'source_path', 'unbekanntes Dokument')
        
        return [{
            "text": f"Fallback-Chunk für {source_path}",
            "metadata": {
                "page_number": 1,
                "chunk_index": 0,
                "chunk_strategy": f"{self.strategy}_fallback",
                "text_length": len(f"Fallback-Chunk für {source_path}"),
                "docling_chunker": False,
                "is_fallback": True,
                "is_dummy": True,
            }
        }]
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Gibt Informationen über die aktuelle Chunking-Strategie zurück."""
        strategies = {
            "by_heading": "Docling HierarchicalChunker (Überschriften-basiert)",
            "by_page": "Docling HybridChunker (Seiten-basiert)",
            "by_element": "Docling BaseChunker (Element-basiert)",
            "hybrid": "Docling HybridChunker (Hybrid-Modus)",
        }
        
        return {
            "strategy": self.strategy,
            "description": strategies.get(self.strategy, "Unbekannte Strategie"),
            "uses_docling_chunker": self.chunker is not None,
            "chunker_type": type(self.chunker).__name__ if self.chunker else "Fallback",
            "config": {
                "max_chunk_size": self.config.max_chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "min_chunk_size": self.config.min_chunk_size,
                "preserve_tables": self.config.preserve_tables,
                "preserve_images": self.config.preserve_images,
            }
        }


# Backward-Kompatibilität: Alias für alte SectionExtractor-Klasse
SectionExtractor = DoclingChunker


# Hilfsfunktionen für externe Nutzung
def extract_sections_simple(docling_doc: Any, strategy: str = "hybrid", config: Optional[ChunkConfig] = None) -> List[str]:
    """
    Einfache Extraktion nur der Texte (ohne Metadaten).
    
    Args:
        docling_doc: Docling Document-Objekt
        strategy: Chunking-Strategie
        config: Chunking-Konfiguration
        
    Returns:
        Liste von Text-Strings
    """
    chunker = DoclingChunker(strategy=strategy, config=config)
    blocks = chunker.split(docling_doc)
    return [block["text"] for block in blocks]


def count_sections(docling_doc: Any, strategy: str = "hybrid", config: Optional[ChunkConfig] = None) -> int:
    """
    Zählt die Anzahl der Chunks für eine gegebene Strategie.
    
    Args:
        docling_doc: Docling Document-Objekt
        strategy: Chunking-Strategie
        config: Chunking-Konfiguration
        
    Returns:
        Anzahl der Chunks
    """
    chunker = DoclingChunker(strategy=strategy, config=config)
    blocks = chunker.split(docling_doc)
    return len(blocks)


def create_chunk_config(
    max_chunk_size: int = 1024,
    chunk_overlap: int = 100,
    preserve_tables: bool = True,
    preserve_images: bool = True
) -> ChunkConfig:
    """
    Erstellt eine Chunking-Konfiguration mit den angegebenen Parametern.
    
    Args:
        max_chunk_size: Maximale Chunk-Größe in Zeichen
        chunk_overlap: Überlappung zwischen Chunks
        preserve_tables: Tabellen-Inhalte extrahieren
        preserve_images: Bild-Inhalte extrahieren
        
    Returns:
        ChunkConfig-Objekt
    """
    return ChunkConfig(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_tables=preserve_tables,
        preserve_images=preserve_images
    ) 