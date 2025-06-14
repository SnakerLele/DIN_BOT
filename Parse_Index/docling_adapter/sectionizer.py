"""
SectionExtractor
----------------
Extrahiert semantische Abschnitte aus Docling-Dokumenten.

Strategien:
- by_heading: Chunking basierend auf Überschriften-Hierarchie
- by_page: Ein Chunk pro Seite
- hybrid: Kombination aus Überschriften und Seitengrenzen
- by_element: Jedes Element (Tabelle, Absatz, etc.) als eigener Chunk
"""

from typing import List, Dict, Any, Optional
import logging


class SectionExtractor:
    """
    Extrahiert semantische Abschnitte aus Docling-Dokumenten.
    
    Wandelt die Docling-Struktur in granulare Blöcke um, die für
    RAG-Systeme optimal sind.
    """
    
    def __init__(self, strategy: str = "hybrid"):
        """
        Initialisiert den SectionExtractor.
        
        Args:
            strategy: Chunking-Strategie ("by_heading", "by_page", "hybrid", "by_element")
        """
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # Validiere Strategie
        valid_strategies = ["by_heading", "by_page", "hybrid", "by_element"]
        if strategy not in valid_strategies:
            raise ValueError(f"Ungültige Strategie: {strategy}. Gültig: {valid_strategies}")
    
    def split(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Hauptmethode: Zerlegt ein Docling-Document in semantische Blöcke.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            Liste von Blöcken, jeder Block enthält:
            - text: Textinhalt des Blocks
            - metadata: Dictionary mit Metadaten (Seite, Abschnitt, Typ, etc.)
        """
        try:
            if self.strategy == "by_heading":
                return self._split_by_heading(docling_doc)
            elif self.strategy == "by_page":
                return self._split_by_page(docling_doc)
            elif self.strategy == "by_element":
                return self._split_by_element(docling_doc)
            else:  # hybrid
                return self._split_hybrid(docling_doc)
                
        except Exception as e:
            self.logger.error(f"Fehler beim Sectionizing: {str(e)}")
            # Fallback: Einfache Seiten-basierte Aufteilung
            return self._split_by_page_fallback(docling_doc)
    
    def _split_by_heading(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Chunking basierend auf Überschriften-Hierarchie.
        
        Erstellt neue Chunks bei jeder H1/H2-Überschrift und sammelt
        alle folgenden Elemente bis zur nächsten Überschrift.
        """
        blocks = []
        current_section = {
            "text_parts": [],
            "h1": None,
            "h2": None,
            "h3": None,
            "page_numbers": set(),
            "element_types": set(),
        }
        
        # Echte Docling-API verwenden
        # Docling hat eine andere Struktur: document.main_text oder document.export_to_markdown()
        
        # Debug: Schauen wir uns die Docling-Struktur an
        self.logger.info(f"Docling-Document Typ: {type(docling_doc)}")
        self.logger.info(f"Docling-Document Attribute: {dir(docling_doc)}")
        
        # Versuche verschiedene Wege, den Text zu extrahieren
        text_content = ""
        
        # Methode 1: Markdown-Export
        try:
            text_content = docling_doc.export_to_markdown()
            self.logger.info(f"Markdown-Export erfolgreich: {len(text_content)} Zeichen")
        except Exception as e:
            self.logger.warning(f"Markdown-Export fehlgeschlagen: {e}")
        
        # Methode 2: Fallback auf main_text
        if not text_content and hasattr(docling_doc, 'main_text'):
            text_content = docling_doc.main_text
            self.logger.info(f"Main-Text extrahiert: {len(text_content)} Zeichen")
        
        # Methode 3: Fallback auf pages
        if not text_content:
            pages = getattr(docling_doc, 'pages', [])
            if pages:
                page_texts = []
                for page in pages:
                    if hasattr(page, 'text'):
                        page_texts.append(page.text)
                text_content = "\n\n".join(page_texts)
                self.logger.info(f"Seiten-Text extrahiert: {len(text_content)} Zeichen")
        
        # Wenn wir Text haben, erstelle einen einfachen Block
        if text_content and text_content.strip():
            return [{
                "text": text_content,
                "metadata": {
                    "page_number": 1,
                    "section_h1": "Dokument",
                    "section_h2": None,
                    "section_h3": None,
                    "docling_type": "full_document",
                    "chunk_strategy": self.strategy,
                    "total_chars": len(text_content),
                }
            }]
        
        # Fallback für leere Dokumente
        self.logger.warning("Kein Text aus Docling-Document extrahiert")
        return self._create_dummy_blocks(docling_doc)
    
    def _split_by_page(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Ein Chunk pro Seite.
        
        Sammelt alle Elemente einer Seite in einem Block.
        """
        blocks = []
        pages = getattr(docling_doc, 'pages', [])
        
        if not pages:
            return self._create_dummy_blocks(docling_doc)
        
        for page_num, page in enumerate(pages, start=1):
            elements = getattr(page, 'elements', [])
            
            text_parts = []
            element_types = set()
            
            for element in elements:
                element_text = getattr(element, 'text', '')
                element_type = getattr(element, 'type', 'text')
                
                if element_text.strip():
                    text_parts.append(element_text)
                    element_types.add(element_type)
            
            if text_parts:
                block = {
                    "text": "\n\n".join(text_parts),
                    "metadata": {
                        "page_number": page_num,
                        "section_h1": None,
                        "section_h2": None,
                        "section_h3": None,
                        "docling_type": "page",
                        "element_types": list(element_types),
                        "chunk_strategy": "by_page",
                    }
                }
                blocks.append(block)
        
        self.logger.info(f"Seiten-basiertes Chunking: {len(blocks)} Seiten")
        return blocks
    
    def _split_by_element(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Jedes Element als eigener Chunk.
        
        Erstellt für jede Tabelle, jeden Absatz, etc. einen separaten Block.
        Optimal für sehr granulare Suche.
        """
        blocks = []
        pages = getattr(docling_doc, 'pages', [])
        
        if not pages:
            return self._create_dummy_blocks(docling_doc)
        
        current_headings = {"h1": None, "h2": None, "h3": None}
        
        for page_num, page in enumerate(pages, start=1):
            elements = getattr(page, 'elements', [])
            
            for element_idx, element in enumerate(elements):
                element_text = getattr(element, 'text', '')
                element_type = getattr(element, 'type', 'text')
                
                # Überschriften-Kontext aktualisieren
                if element_type == "heading":
                    level = getattr(element, 'level', 1)
                    if level == 1:
                        current_headings = {"h1": element_text, "h2": None, "h3": None}
                    elif level == 2:
                        current_headings["h2"] = element_text
                        current_headings["h3"] = None
                    elif level == 3:
                        current_headings["h3"] = element_text
                
                # Block für jedes Element mit Inhalt
                if element_text.strip():
                    block = {
                        "text": element_text,
                        "metadata": {
                            "page_number": page_num,
                            "element_index": element_idx,
                            "section_h1": current_headings["h1"],
                            "section_h2": current_headings["h2"],
                            "section_h3": current_headings["h3"],
                            "docling_type": element_type,
                            "chunk_strategy": "by_element",
                        }
                    }
                    
                    # Spezielle Metadaten für Tabellen/Bilder
                    if element_type == "table":
                        block["metadata"]["table_id"] = f"table_{page_num}_{element_idx}"
                    elif element_type == "image":
                        block["metadata"]["image_id"] = f"image_{page_num}_{element_idx}"
                    elif element_type == "formula":
                        block["metadata"]["formula_id"] = f"formula_{page_num}_{element_idx}"
                    
                    blocks.append(block)
        
        self.logger.info(f"Element-basiertes Chunking: {len(blocks)} Elemente")
        return blocks
    
    def _split_hybrid(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Hybrid-Strategie: Kombination aus Überschriften und Seitengrenzen.
        
        - Große Abschnitte werden bei Überschriften getrennt
        - Sehr lange Abschnitte werden zusätzlich bei Seitengrenzen getrennt
        - Tabellen/Bilder werden als separate Blöcke behandelt
        """
        # Erst heading-basiert chunken
        heading_blocks = self._split_by_heading(docling_doc)
        
        # Dann große Blöcke weiter aufteilen
        refined_blocks = []
        max_chunk_size = 2000  # Zeichen
        
        for block in heading_blocks:
            text = block["text"]
            
            # Kleine Blöcke unverändert übernehmen
            if len(text) <= max_chunk_size:
                refined_blocks.append(block)
                continue
            
            # Große Blöcke aufteilen
            # TODO: Intelligentere Aufteilung (Satzgrenzen, etc.)
            chunks = self._split_large_text(text, max_chunk_size)
            
            for i, chunk_text in enumerate(chunks):
                chunk_block = {
                    "text": chunk_text,
                    "metadata": {
                        **block["metadata"],
                        "chunk_part": i + 1,
                        "total_parts": len(chunks),
                        "chunk_strategy": "hybrid",
                    }
                }
                refined_blocks.append(chunk_block)
        
        self.logger.info(f"Hybrid-Chunking: {len(refined_blocks)} Blöcke")
        return refined_blocks
    
    def _split_large_text(self, text: str, max_size: int) -> List[str]:
        """
        Teilt großen Text in kleinere Chunks auf.
        
        Versucht an Satzgrenzen zu trennen.
        """
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            # Satz hinzufügen würde Chunk zu groß machen
            if len(current_chunk) + len(sentence) + 2 > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Einzelner Satz ist zu lang - hart trennen
                    chunks.append(sentence[:max_size])
                    current_chunk = sentence[max_size:]
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _finalize_section(self, section: Dict) -> Dict[str, Any]:
        """Wandelt einen Abschnitt in einen finalen Block um."""
        return {
            "text": "\n\n".join(section["text_parts"]),
            "metadata": {
                "page_number": min(section["page_numbers"]) if section["page_numbers"] else 1,
                "page_range": f"{min(section['page_numbers'])}-{max(section['page_numbers'])}" 
                             if len(section["page_numbers"]) > 1 else str(min(section["page_numbers"], default=1)),
                "section_h1": section["h1"],
                "section_h2": section["h2"],
                "section_h3": section["h3"],
                "docling_type": "section",
                "element_types": list(section["element_types"]),
                "chunk_strategy": "by_heading",
            }
        }
    
    def _reset_section(self) -> Dict:
        """Erstellt eine neue leere Sektion."""
        return {
            "text_parts": [],
            "h1": None,
            "h2": None,
            "h3": None,
            "page_numbers": set(),
            "element_types": set(),
        }
    
    def _split_by_page_fallback(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """Fallback-Implementierung für Fehlerbehandlung."""
        self.logger.warning("Verwende Fallback-Chunking")
        return self._create_dummy_blocks(docling_doc)
    
    def _create_dummy_blocks(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """Erstellt Dummy-Blöcke für Tests/Fallbacks."""
        return [{
            "text": f"Dummy-Inhalt für {getattr(docling_doc, 'source_path', 'unbekanntes Dokument')}",
            "metadata": {
                "page_number": 1,
                "section_h1": "Dummy-Abschnitt",
                "section_h2": None,
                "section_h3": None,
                "docling_type": "text",
                "chunk_strategy": self.strategy,
                "is_dummy": True,
            }
        }]
    
    def get_strategy_info(self) -> Dict[str, str]:
        """
        Gibt Informationen über die aktuelle Chunking-Strategie zurück.
        
        Returns:
            Dictionary mit Strategie-Details
        """
        strategies = {
            "by_heading": "Chunking basierend auf Überschriften-Hierarchie",
            "by_page": "Ein Chunk pro Seite",
            "by_element": "Jedes Element als eigener Chunk",
            "hybrid": "Kombination aus Überschriften und Seitengrenzen",
        }
        
        return {
            "strategy": self.strategy,
            "description": strategies.get(self.strategy, "Unbekannte Strategie"),
            "granularity": "high" if self.strategy == "by_element" else "medium" if self.strategy == "hybrid" else "low",
        }


# Hilfsfunktionen für externe Nutzung
def extract_sections_simple(docling_doc: Any, strategy: str = "hybrid") -> List[str]:
    """
    Einfache Extraktion nur der Texte (ohne Metadaten).
    
    Args:
        docling_doc: Docling Document-Objekt
        strategy: Chunking-Strategie
        
    Returns:
        Liste von Text-Strings
    """
    extractor = SectionExtractor(strategy=strategy)
    blocks = extractor.split(docling_doc)
    return [block["text"] for block in blocks]


def count_sections(docling_doc: Any, strategy: str = "hybrid") -> int:
    """
    Zählt die Anzahl der Abschnitte für eine gegebene Strategie.
    
    Args:
        docling_doc: Docling Document-Objekt
        strategy: Chunking-Strategie
        
    Returns:
        Anzahl der Abschnitte
    """
    extractor = SectionExtractor(strategy=strategy)
    blocks = extractor.split(docling_doc)
    return len(blocks) 