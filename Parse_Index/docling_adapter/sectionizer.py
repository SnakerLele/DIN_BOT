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

from typing import List, Dict, Any, Optional, Union
import logging
import re
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Konfiguration für Chunking-Parameter."""
    max_chunk_size: int = 1024  # Optimiert für Embedding-Modelle
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    sentence_splitters: List[str] = None
    preserve_tables: bool = True
    preserve_images: bool = True
    extract_table_html: bool = True
    
    def __post_init__(self):
        if self.sentence_splitters is None:
            self.sentence_splitters = ['. ', '! ', '? ', '.\n', '!\n', '?\n']


class SectionExtractor:
    """
    Extrahiert semantische Abschnitte aus Docling-Dokumenten.
    
    Wandelt die Docling-Struktur in granulare Blöcke um, die für
    RAG-Systeme optimal sind.
    """
    
    def __init__(self, strategy: str = "hybrid", config: Optional[ChunkConfig] = None):
        """
        Initialisiert den SectionExtractor.
        
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
        
        # Regex für bessere Satz-Erkennung
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])')
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
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
            self.logger.debug(f"Starte Sectionizing mit Strategie: {self.strategy}")
            
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
            # Fallback: Robuste Seiten-basierte Aufteilung
            return self._split_by_page_fallback(docling_doc)
    
    def _split_by_heading(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Chunking basierend auf echter Überschriften-Hierarchie.
        
        Analysiert die Docling-Struktur direkt und erstellt semantische Abschnitte
        basierend auf erkannten Überschriften.
        """
        blocks = []
        
        # Versuche echte Docling-Struktur zu nutzen
        if hasattr(docling_doc, 'pages') and docling_doc.pages:
            blocks = self._extract_heading_based_sections(docling_doc)
            if blocks:
                self.logger.info(f"Heading-basiertes Chunking: {len(blocks)} Abschnitte")
                return blocks
        
        # Fallback: Markdown-basierte Analyse
        blocks = self._extract_from_markdown(docling_doc)
        if blocks:
            self.logger.info(f"Markdown-basiertes Chunking: {len(blocks)} Abschnitte")
            return blocks
        
        # Letzter Fallback: Intelligente Text-Analyse
        self.logger.warning("Verwende Text-basierte Heading-Erkennung als Fallback")
        return self._extract_headings_from_text(docling_doc)
    
    def _extract_heading_based_sections(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Extrahiert Abschnitte basierend auf echter Docling-Struktur.
        """
        blocks = []
        current_section = self._create_empty_section()
        
        for page_num, page in enumerate(docling_doc.pages, start=1):
            elements = getattr(page, 'elements', [])
            
            for element in elements:
                element_text = getattr(element, 'text', '').strip()
                element_type = getattr(element, 'type', 'text')
                
                if not element_text:
                    continue
                
                # Überschrift erkannt - neue Sektion starten
                if element_type in ['heading', 'title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    # Aktuelle Sektion abschließen
                    if current_section["text_parts"]:
                        blocks.append(self._finalize_section(current_section))
                    
                    # Neue Sektion starten
                    current_section = self._create_empty_section()
                    level = self._extract_heading_level(element, element_type)
                    self._update_heading_context(current_section, element_text, level)
                
                # Text zur aktuellen Sektion hinzufügen
                current_section["text_parts"].append(element_text)
                current_section["page_numbers"].add(page_num)
                current_section["element_types"].add(element_type)
                
                # Spezielle Behandlung für Tabellen/Bilder
                if element_type in ['table', 'image', 'figure']:
                    table_content = self._extract_table_content(element, element_type)
                    if table_content:
                        current_section["special_content"] = current_section.get("special_content", [])
                        current_section["special_content"].append(table_content)
        
        # Letzte Sektion abschließen
        if current_section["text_parts"]:
            blocks.append(self._finalize_section(current_section))
        
        return blocks
    
    def _extract_from_markdown(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Fallback: Extrahiert Abschnitte aus Markdown-Export.
        """
        try:
            # Verschiedene Markdown-Export-Methoden versuchen
            markdown_text = None
            
            # Methode 1: Standard export_to_markdown
            if hasattr(docling_doc, 'export_to_markdown'):
                try:
                    markdown_text = docling_doc.export_to_markdown()
                except Exception as e:
                    self.logger.debug(f"export_to_markdown fehlgeschlagen: {e}")
            
            # Methode 2: Alternative Markdown-Methoden
            if not markdown_text:
                for method_name in ['to_markdown', 'as_markdown', 'get_markdown']:
                    if hasattr(docling_doc, method_name):
                        try:
                            method = getattr(docling_doc, method_name)
                            markdown_text = method()
                            break
                        except Exception as e:
                            self.logger.debug(f"{method_name} fehlgeschlagen: {e}")
            
            if not markdown_text:
                return []
            
            return self._parse_markdown_sections(markdown_text)
            
        except Exception as e:
            self.logger.warning(f"Markdown-Extraktion fehlgeschlagen: {e}")
            return []
    
    def _parse_markdown_sections(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        Parst Markdown-Text in semantische Abschnitte.
        """
        blocks = []
        sections = []
        current_section = {"text": "", "level": 0, "title": ""}
        
        lines = markdown_text.split('\n')
        
        for line in lines:
            heading_match = self.heading_pattern.match(line)
            
            if heading_match:
                # Aktuelle Sektion abschließen
                if current_section["text"].strip():
                    sections.append(current_section)
                
                # Neue Sektion starten
                level = len(heading_match.group(1))  # Anzahl #
                title = heading_match.group(2).strip()
                current_section = {"text": "", "level": level, "title": title}
            else:
                current_section["text"] += line + "\n"
        
        # Letzte Sektion hinzufügen
        if current_section["text"].strip():
            sections.append(current_section)
        
        # Sektionen in Blöcke umwandeln
        for i, section in enumerate(sections):
            if section["text"].strip():
                # Große Sektionen aufteilen
                chunks = self._split_text_intelligently(section["text"])
                
                for j, chunk in enumerate(chunks):
                    block = {
                        "text": chunk,
                        "metadata": {
                            "page_number": 1,  # Markdown hat keine Seiten-Info
                            "section_title": section["title"],
                            "section_level": section["level"],
                            "section_index": i,
                            "chunk_index": j,
                            "total_chunks": len(chunks),
                            "docling_type": "markdown_section",
                            "chunk_strategy": "by_heading",
                            "text_length": len(chunk),
                        }
                    }
                    
                    # Überschriften-Hierarchie rekonstruieren
                    if section["level"] == 1:
                        block["metadata"]["section_h1"] = section["title"]
                    elif section["level"] == 2:
                        block["metadata"]["section_h2"] = section["title"]
                    elif section["level"] == 3:
                        block["metadata"]["section_h3"] = section["title"]
                    
                    blocks.append(block)
        
        return blocks
    
    def _extract_headings_from_text(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Letzter Fallback: Heuristische Überschriften-Erkennung aus reinem Text.
        """
        # Text extrahieren
        full_text = self._extract_full_text(docling_doc)
        if not full_text:
            return self._create_dummy_blocks(docling_doc)
        
        # Heuristische Überschriften-Erkennung
        lines = full_text.split('\n')
        sections = []
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Heuristik für Überschriften
            if self._is_likely_heading(line):
                # Aktuelle Sektion abschließen
                if current_text:
                    sections.append({
                        "title": "Abschnitt",
                        "text": '\n'.join(current_text)
                    })
                    current_text = []
                
                # Neue Sektion mit Überschrift
                sections.append({
                    "title": line,
                    "text": ""
                })
            else:
                current_text.append(line)
        
        # Letzte Sektion
        if current_text:
            sections.append({
                "title": "Abschnitt",
                "text": '\n'.join(current_text)
            })
        
        # In Blöcke umwandeln
        blocks = []
        for i, section in enumerate(sections):
            if section["text"].strip():
                chunks = self._split_text_intelligently(section["text"])
                
                for j, chunk in enumerate(chunks):
                    blocks.append({
                        "text": chunk,
                        "metadata": {
                            "page_number": 1,
                            "section_h1": section["title"] if i == 0 else None,
                            "section_h2": section["title"] if i > 0 else None,
                            "section_h3": None,
                            "docling_type": "heuristic_section",
                            "chunk_strategy": "by_heading",
                            "section_index": i,
                            "chunk_index": j,
                            "text_length": len(chunk),
                        }
                    })
        
        return blocks if blocks else self._create_dummy_blocks(docling_doc)
    
    def _split_by_page(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Ein Chunk pro Seite mit verbesserter Element-Extraktion.
        """
        blocks = []
        pages = getattr(docling_doc, 'pages', [])
        
        if not pages:
            return self._create_dummy_blocks(docling_doc)
        
        for page_num, page in enumerate(pages, start=1):
            elements = getattr(page, 'elements', [])
            
            text_parts = []
            element_types = set()
            special_content = []
            
            for element in elements:
                element_text = getattr(element, 'text', '').strip()
                element_type = getattr(element, 'type', 'text')
                
                if element_text:
                    text_parts.append(element_text)
                    element_types.add(element_type)
                
                # Spezielle Inhalte extrahieren
                if element_type in ['table', 'image', 'figure'] and self.config.preserve_tables:
                    content = self._extract_table_content(element, element_type)
                    if content:
                        special_content.append(content)
            
            if text_parts:
                page_text = "\n\n".join(text_parts)
                
                # Große Seiten aufteilen
                if len(page_text) > self.config.max_chunk_size:
                    chunks = self._split_text_intelligently(page_text)
                    
                    for i, chunk in enumerate(chunks):
                        block = {
                            "text": chunk,
                            "metadata": {
                                "page_number": page_num,
                                "page_chunk": i + 1,
                                "total_page_chunks": len(chunks),
                                "section_h1": None,
                                "section_h2": None,
                                "section_h3": None,
                                "docling_type": "page_chunk",
                                "element_types": list(element_types),
                                "chunk_strategy": "by_page",
                                "text_length": len(chunk),
                            }
                        }
                        
                        if special_content:
                            block["metadata"]["special_content"] = special_content
                        
                        blocks.append(block)
                else:
                    block = {
                        "text": page_text,
                        "metadata": {
                            "page_number": page_num,
                            "section_h1": None,
                            "section_h2": None,
                            "section_h3": None,
                            "docling_type": "page",
                            "element_types": list(element_types),
                            "chunk_strategy": "by_page",
                            "text_length": len(page_text),
                        }
                    }
                    
                    if special_content:
                        block["metadata"]["special_content"] = special_content
                    
                    blocks.append(block)
        
        self.logger.info(f"Seiten-basiertes Chunking: {len(blocks)} Blöcke")
        return blocks
    
    def _split_by_element(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Jedes Element als eigener Chunk mit verbesserter Metadaten-Extraktion.
        """
        blocks = []
        pages = getattr(docling_doc, 'pages', [])
        
        if not pages:
            return self._create_dummy_blocks(docling_doc)
        
        current_headings = {"h1": None, "h2": None, "h3": None}
        
        for page_num, page in enumerate(pages, start=1):
            elements = getattr(page, 'elements', [])
            
            for element_idx, element in enumerate(elements):
                element_text = getattr(element, 'text', '').strip()
                element_type = getattr(element, 'type', 'text')
                
                if not element_text:
                    continue
                
                # Überschriften-Kontext aktualisieren
                if element_type in ['heading', 'title', 'h1', 'h2', 'h3']:
                    level = self._extract_heading_level(element, element_type)
                    self._update_heading_context_dict(current_headings, element_text, level)
                
                # Block für jedes Element erstellen
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
                        "text_length": len(element_text),
                    }
                }
                
                # Spezielle Metadaten für verschiedene Element-Typen
                if element_type == "table":
                    block["metadata"]["table_id"] = f"table_{page_num}_{element_idx}"
                    if self.config.extract_table_html:
                        table_content = self._extract_table_content(element, element_type)
                        if table_content:
                            block["metadata"]["table_html"] = table_content.get("html", "")
                            block["metadata"]["table_data"] = table_content.get("data", [])
                            
                elif element_type in ["image", "figure"]:
                    block["metadata"]["image_id"] = f"image_{page_num}_{element_idx}"
                    image_content = self._extract_image_content(element)
                    if image_content:
                        block["metadata"].update(image_content)
                        
                elif element_type in ["formula", "equation"]:
                    block["metadata"]["formula_id"] = f"formula_{page_num}_{element_idx}"
                    formula_content = self._extract_formula_content(element)
                    if formula_content:
                        block["metadata"].update(formula_content)
                
                blocks.append(block)
        
        self.logger.info(f"Element-basiertes Chunking: {len(blocks)} Elemente")
        return blocks
    
    def _split_hybrid(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Verbesserte Hybrid-Strategie mit konfigurierbaren Parametern.
        
        1. Versucht zuerst element-basierte Extraktion für maximale Semantik
        2. Fasst kleine Elemente zu größeren Chunks zusammen
        3. Teilt sehr große Chunks intelligent auf
        """
        # Starte mit element-basierter Extraktion für maximale Semantik
        element_blocks = self._split_by_element(docling_doc)
        
        if not element_blocks:
            # Fallback auf heading-basiert
            return self._split_by_heading(docling_doc)
        
        # Intelligente Zusammenfassung und Aufteilung
        refined_blocks = []
        current_chunk = {"text_parts": [], "metadata_list": [], "total_length": 0}
        
        for block in element_blocks:
            text = block["text"]
            text_length = len(text)
            
            # Sehr große Einzelelemente (z.B. lange Tabellen) separat behandeln
            if text_length > self.config.max_chunk_size * 1.5:
                # Aktuellen Chunk abschließen
                if current_chunk["text_parts"]:
                    refined_blocks.append(self._merge_chunk_parts(current_chunk))
                    current_chunk = {"text_parts": [], "metadata_list": [], "total_length": 0}
                
                # Großes Element aufteilen
                sub_chunks = self._split_text_intelligently(text)
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_block = {
                        "text": sub_chunk,
                        "metadata": {
                            **block["metadata"],
                            "chunk_part": i + 1,
                            "total_parts": len(sub_chunks),
                            "chunk_strategy": "hybrid",
                            "text_length": len(sub_chunk),
                        }
                    }
                    refined_blocks.append(sub_block)
                continue
            
            # Prüfen ob aktueller Chunk + neues Element zu groß wird
            if (current_chunk["total_length"] + text_length > self.config.max_chunk_size 
                and current_chunk["text_parts"]):
                
                # Aktuellen Chunk abschließen
                refined_blocks.append(self._merge_chunk_parts(current_chunk))
                current_chunk = {"text_parts": [], "metadata_list": [], "total_length": 0}
            
            # Element zum aktuellen Chunk hinzufügen
            current_chunk["text_parts"].append(text)
            current_chunk["metadata_list"].append(block["metadata"])
            current_chunk["total_length"] += text_length
        
        # Letzten Chunk abschließen
        if current_chunk["text_parts"]:
            refined_blocks.append(self._merge_chunk_parts(current_chunk))
        
        self.logger.info(f"Hybrid-Chunking: {len(refined_blocks)} Blöcke")
        return refined_blocks
    
    def _split_text_intelligently(self, text: str) -> List[str]:
        """
        Intelligente Text-Aufteilung mit konfigurierbaren Parametern.
        
        Versucht in dieser Reihenfolge zu trennen:
        1. An Absatzgrenzen (\n\n)
        2. An Satzgrenzen (. ! ?)
        3. An Wortgrenzen
        4. Hart bei max_size
        """
        if len(text) <= self.config.max_chunk_size:
            return [text]
        
        chunks = []
        remaining_text = text
        
        while len(remaining_text) > self.config.max_chunk_size:
            # Optimale Trennstelle finden
            split_pos = self._find_best_split_position(
                remaining_text, 
                self.config.max_chunk_size
            )
            
            if split_pos == -1:
                # Notfall: Hart bei max_size trennen
                split_pos = self.config.max_chunk_size
            
            chunk = remaining_text[:split_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            # Overlap berücksichtigen
            overlap_start = max(0, split_pos - self.config.chunk_overlap)
            remaining_text = remaining_text[overlap_start:].strip()
            
            # Endlosschleife verhindern
            if split_pos == 0:
                break
        
        # Letzten Teil hinzufügen
        if remaining_text.strip():
            chunks.append(remaining_text.strip())
        
        return [chunk for chunk in chunks if len(chunk) >= self.config.min_chunk_size]
    
    def _find_best_split_position(self, text: str, max_pos: int) -> int:
        """
        Findet die beste Position zum Trennen des Textes.
        """
        if max_pos >= len(text):
            return len(text)
        
        # 1. Versuche Absatzgrenze
        paragraph_pos = text.rfind('\n\n', 0, max_pos)
        if paragraph_pos > max_pos * 0.5:  # Mindestens 50% der gewünschten Länge
            return paragraph_pos + 2
        
        # 2. Versuche Satzgrenze
        for splitter in self.config.sentence_splitters:
            sentence_pos = text.rfind(splitter, 0, max_pos)
            if sentence_pos > max_pos * 0.7:  # Mindestens 70% der gewünschten Länge
                return sentence_pos + len(splitter)
        
        # 3. Versuche Wortgrenze
        word_pos = text.rfind(' ', 0, max_pos)
        if word_pos > max_pos * 0.8:  # Mindestens 80% der gewünschten Länge
            return word_pos + 1
        
        # 4. Keine gute Trennstelle gefunden
        return -1
    
    def _extract_table_content(self, element: Any, element_type: str) -> Optional[Dict[str, Any]]:
        """
        Extrahiert Tabellen-Inhalt in verschiedenen Formaten.
        """
        if not self.config.preserve_tables:
            return None
        
        content = {"type": element_type}
        
        try:
            # HTML-Repräsentation extrahieren
            if hasattr(element, 'to_html') and self.config.extract_table_html:
                content["html"] = element.to_html()
            elif hasattr(element, 'html'):
                content["html"] = element.html
            
            # Strukturierte Daten extrahieren
            if hasattr(element, 'data'):
                content["data"] = element.data
            elif hasattr(element, 'rows'):
                rows = []
                for row in element.rows:
                    if hasattr(row, 'cells'):
                        row_data = [getattr(cell, 'text', str(cell)) for cell in row.cells]
                        rows.append(row_data)
                content["data"] = rows
            
            # Fallback: Text-Repräsentation
            if hasattr(element, 'text') and element.text:
                content["text"] = element.text
            
            # Zusätzliche Metadaten
            if hasattr(element, 'bbox'):
                content["bbox"] = element.bbox
            if hasattr(element, 'confidence'):
                content["confidence"] = element.confidence
                
        except Exception as e:
            self.logger.warning(f"Fehler bei Tabellen-Extraktion: {e}")
            content["error"] = str(e)
        
        return content if len(content) > 1 else None
    
    def _extract_image_content(self, element: Any) -> Optional[Dict[str, Any]]:
        """
        Extrahiert Bild-Metadaten und -Inhalt.
        """
        if not self.config.preserve_images:
            return None
        
        content = {}
        
        try:
            # Bild-Metadaten
            if hasattr(element, 'alt_text'):
                content["alt_text"] = element.alt_text
            if hasattr(element, 'caption'):
                content["caption"] = element.caption
            if hasattr(element, 'bbox'):
                content["bbox"] = element.bbox
            if hasattr(element, 'size'):
                content["size"] = element.size
            
            # Bild-Pfad oder Base64-Daten
            if hasattr(element, 'image_path'):
                content["image_path"] = element.image_path
            elif hasattr(element, 'image_data'):
                content["has_image_data"] = True  # Nicht die Daten selbst speichern
            
        except Exception as e:
            self.logger.warning(f"Fehler bei Bild-Extraktion: {e}")
            content["error"] = str(e)
        
        return content if content else None
    
    def _extract_formula_content(self, element: Any) -> Optional[Dict[str, Any]]:
        """
        Extrahiert Formel-Inhalt (LaTeX, MathML, etc.).
        """
        content = {}
        
        try:
            if hasattr(element, 'latex'):
                content["latex"] = element.latex
            if hasattr(element, 'mathml'):
                content["mathml"] = element.mathml
            if hasattr(element, 'text'):
                content["text_representation"] = element.text
            
        except Exception as e:
            self.logger.warning(f"Fehler bei Formel-Extraktion: {e}")
            content["error"] = str(e)
        
        return content if content else None
    
    def _extract_heading_level(self, element: Any, element_type: str) -> int:
        """
        Extrahiert das Überschriften-Level aus einem Element.
        """
        # Direkte Level-Eigenschaft
        if hasattr(element, 'level'):
            return element.level
        
        # Aus Element-Typ ableiten
        if element_type in ['h1', 'title']:
            return 1
        elif element_type == 'h2':
            return 2
        elif element_type == 'h3':
            return 3
        elif element_type in ['h4', 'h5', 'h6']:
            return int(element_type[1])
        
        # Heuristik basierend auf Text-Eigenschaften
        if hasattr(element, 'font_size'):
            # Größere Schrift = höheres Level (niedrigere Zahl)
            if element.font_size > 16:
                return 1
            elif element.font_size > 14:
                return 2
            else:
                return 3
        
        return 2  # Standard-Level
    
    def _update_heading_context(self, section: Dict, heading_text: str, level: int):
        """
        Aktualisiert den Überschriften-Kontext einer Sektion.
        """
        if level == 1:
            section["h1"] = heading_text
            section["h2"] = None
            section["h3"] = None
        elif level == 2:
            section["h2"] = heading_text
            section["h3"] = None
        elif level == 3:
            section["h3"] = heading_text
    
    def _update_heading_context_dict(self, headings: Dict, heading_text: str, level: int):
        """
        Aktualisiert ein Überschriften-Dictionary.
        """
        if level == 1:
            headings["h1"] = heading_text
            headings["h2"] = None
            headings["h3"] = None
        elif level == 2:
            headings["h2"] = heading_text
            headings["h3"] = None
        elif level == 3:
            headings["h3"] = heading_text
    
    def _create_empty_section(self) -> Dict:
        """
        Erstellt eine neue leere Sektion.
        """
        return {
            "text_parts": [],
            "h1": None,
            "h2": None,
            "h3": None,
            "page_numbers": set(),
            "element_types": set(),
            "special_content": [],
        }
    
    def _finalize_section(self, section: Dict) -> Dict[str, Any]:
        """
        Wandelt eine Sektion in einen finalen Block um.
        """
        text = "\n\n".join(section["text_parts"])
        
        # Sichere Behandlung leerer page_numbers
        page_numbers = section.get("page_numbers", set())
        if page_numbers:
            min_page = min(page_numbers)
            if len(page_numbers) > 1:
                page_range = f"{min_page}-{max(page_numbers)}"
            else:
                page_range = str(min_page)
        else:
            min_page = 1
            page_range = "1"
        
        metadata = {
            "page_number": min_page,
            "page_range": page_range,
            "section_h1": section.get("h1"),
            "section_h2": section.get("h2"),
            "section_h3": section.get("h3"),
            "docling_type": "section",
            "element_types": list(section.get("element_types", set())),
            "chunk_strategy": "by_heading",
            "text_length": len(text),
        }
        
        # Spezielle Inhalte hinzufügen
        special_content = section.get("special_content", [])
        if special_content:
            metadata["special_content"] = special_content
        
        return {"text": text, "metadata": metadata}
    
    def _merge_chunk_parts(self, chunk_parts: Dict) -> Dict[str, Any]:
        """
        Fügt mehrere Chunk-Teile zu einem Block zusammen.
        """
        text = "\n\n".join(chunk_parts["text_parts"])
        
        # Metadaten aus dem ersten Element als Basis
        base_metadata = chunk_parts["metadata_list"][0].copy()
        
        # Kombinierte Informationen
        all_types = set()
        all_pages = set()
        
        for metadata in chunk_parts["metadata_list"]:
            if "element_types" in metadata:
                if isinstance(metadata["element_types"], list):
                    all_types.update(metadata["element_types"])
                else:
                    all_types.add(metadata["element_types"])
            
            if "page_number" in metadata:
                all_pages.add(metadata["page_number"])
        
        # Aktualisierte Metadaten
        base_metadata.update({
            "docling_type": "merged_chunk",
            "chunk_strategy": "hybrid",
            "element_types": list(all_types),
            "text_length": len(text),
            "merged_elements": len(chunk_parts["text_parts"]),
        })
        
        if all_pages:
            base_metadata["page_number"] = min(all_pages)
            if len(all_pages) > 1:
                base_metadata["page_range"] = f"{min(all_pages)}-{max(all_pages)}"
        
        return {"text": text, "metadata": base_metadata}
    
    def _extract_full_text(self, docling_doc: Any) -> str:
        """
        Extrahiert den kompletten Text aus einem Docling-Document.
        """
        text_parts = []
        
        # Methode 1: Über Seiten und Elemente
        if hasattr(docling_doc, 'pages'):
            for page in docling_doc.pages:
                if hasattr(page, 'elements'):
                    for element in page.elements:
                        element_text = getattr(element, 'text', '')
                        if element_text.strip():
                            text_parts.append(element_text)
        
        # Methode 2: Direkte Text-Eigenschaft
        elif hasattr(docling_doc, 'text'):
            return docling_doc.text
        
        # Methode 3: Main-Text
        elif hasattr(docling_doc, 'main_text'):
            return docling_doc.main_text
        
        return '\n\n'.join(text_parts)
    
    def _is_likely_heading(self, line: str) -> bool:
        """
        Heuristik zur Erkennung von Überschriften.
        """
        line = line.strip()
        
        # Zu kurz oder zu lang
        if len(line) < 3 or len(line) > 100:
            return False
        
        # Endet mit Punkt (wahrscheinlich normaler Text)
        if line.endswith('.'):
            return False
        
        # Beginnt mit Zahl + Punkt/Klammer (Nummerierung)
        if re.match(r'^\d+[\.\)]\s', line):
            return True
        
        # Alle Großbuchstaben (typisch für Überschriften)
        if line.isupper() and len(line) < 50:
            return True
        
        # Beginnt mit Großbuchstaben und hat wenige Satzzeichen
        if (line[0].isupper() and 
            line.count(',') <= 1 and 
            line.count(';') == 0):
            return True
        
        return False
    
    def _split_by_page_fallback(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Robuste Fallback-Implementierung.
        """
        self.logger.warning("Verwende robusten Fallback-Chunking")
        
        # Versuche Text zu extrahieren
        full_text = self._extract_full_text(docling_doc)
        
        if not full_text:
            return self._create_dummy_blocks(docling_doc)
        
        # Einfache text-basierte Aufteilung
        chunks = self._split_text_intelligently(full_text)
        
        blocks = []
        for i, chunk in enumerate(chunks):
            blocks.append({
                "text": chunk,
                "metadata": {
                    "page_number": 1,
                    "section_h1": "Dokument",
                    "section_h2": None,
                    "section_h3": None,
                    "docling_type": "fallback_chunk",
                    "chunk_strategy": "fallback",
                    "chunk_index": i,
                    "text_length": len(chunk),
                    "is_fallback": True,
                }
            })
        
        return blocks
    
    def _create_dummy_blocks(self, docling_doc: Any) -> List[Dict[str, Any]]:
        """
        Erstellt Dummy-Blöcke für Tests/Fallbacks.
        """
        source_path = getattr(docling_doc, 'source_path', 'unbekanntes Dokument')
        
        return [{
            "text": f"Dummy-Inhalt für {source_path}",
            "metadata": {
                "page_number": 1,
                "section_h1": "Dummy-Abschnitt",
                "section_h2": None,
                "section_h3": None,
                "docling_type": "dummy",
                "chunk_strategy": self.strategy,
                "is_dummy": True,
                "text_length": len(f"Dummy-Inhalt für {source_path}"),
            }
        }]
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über die aktuelle Chunking-Strategie zurück.
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
            "config": {
                "max_chunk_size": self.config.max_chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "min_chunk_size": self.config.min_chunk_size,
                "preserve_tables": self.config.preserve_tables,
                "extract_table_html": self.config.extract_table_html,
            }
        }


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
    extractor = SectionExtractor(strategy=strategy, config=config)
    blocks = extractor.split(docling_doc)
    return [block["text"] for block in blocks]


def count_sections(docling_doc: Any, strategy: str = "hybrid", config: Optional[ChunkConfig] = None) -> int:
    """
    Zählt die Anzahl der Abschnitte für eine gegebene Strategie.
    
    Args:
        docling_doc: Docling Document-Objekt
        strategy: Chunking-Strategie
        config: Chunking-Konfiguration
        
    Returns:
        Anzahl der Abschnitte
    """
    extractor = SectionExtractor(strategy=strategy, config=config)
    blocks = extractor.split(docling_doc)
    return len(blocks)


def create_chunk_config(
    max_chunk_size: int = 1024,
    chunk_overlap: int = 100,
    preserve_tables: bool = True,
    extract_table_html: bool = True
) -> ChunkConfig:
    """
    Erstellt eine Chunking-Konfiguration mit den angegebenen Parametern.
    
    Args:
        max_chunk_size: Maximale Chunk-Größe in Zeichen
        chunk_overlap: Überlappung zwischen Chunks
        preserve_tables: Tabellen-Inhalte extrahieren
        extract_table_html: HTML-Repräsentation von Tabellen extrahieren
        
    Returns:
        ChunkConfig-Objekt
    """
    return ChunkConfig(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_tables=preserve_tables,
        extract_table_html=extract_table_html
    ) 