"""
MetadataExtractor
-----------------
Extrahiert Metadaten aus PDF-Dokumenten und Docling-Parsing-Ergebnissen.

Quellen:
- PDF-Metadaten (Titel, Autor, Erstellungsdatum)
- Docling-Metadaten (falls verfügbar)
- Heuristische Extraktion (Titel aus erster Seite, DOI-Erkennung)
- Datei-basierte Metadaten (Pfad, Größe, Änderungsdatum)
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import re
from datetime import datetime


class MetadataExtractor:
    """
    Extrahiert umfassende Metadaten aus PDF-Dokumenten.
    
    Kombiniert verschiedene Quellen für maximale Informationsausbeute.
    """
    
    def __init__(self):
        """Initialisiert den MetadataExtractor."""
        self.logger = logging.getLogger(__name__)
        
        # Regex-Pattern für verschiedene Erkennungen
        self.doi_pattern = re.compile(r'10\.\d{4,}/[^\s]+', re.IGNORECASE)
        self.isbn_pattern = re.compile(r'ISBN[-\s]?(?:97[89][-\s]?)?(?:\d[-\s]?){9}\d', re.IGNORECASE)
        self.year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        
        # DIN-Norm Pattern (für deutsche Normen)
        self.din_pattern = re.compile(r'DIN\s+(?:EN\s+)?(?:ISO\s+)?\d+(?:[-:]\d+)*', re.IGNORECASE)
    
    def extract(self, docling_doc: Any, pdf_path: Path) -> Dict[str, Any]:
        """
        Hauptmethode: Extrahiert alle verfügbaren Metadaten.
        
        Args:
            docling_doc: Docling Document-Objekt
            pdf_path: Pfad zur PDF-Datei
            
        Returns:
            Dictionary mit allen extrahierten Metadaten
        """
        try:
            metadata = {}
            
            # 1. Datei-basierte Metadaten
            file_metadata = self._extract_file_metadata(pdf_path)
            metadata.update(file_metadata)
            
            # 2. PDF-Metadaten (PyPDF2/pypdf)
            pdf_metadata = self._extract_pdf_metadata(pdf_path)
            metadata.update(pdf_metadata)
            
            # 3. Docling-Metadaten (falls verfügbar)
            docling_metadata = self._extract_docling_metadata(docling_doc)
            metadata.update(docling_metadata)
            
            # 4. Heuristische Extraktion aus Inhalt
            content_metadata = self._extract_content_metadata(docling_doc)
            metadata.update(content_metadata)
            
            # 5. Normalisierung und Bereinigung
            metadata = self._normalize_metadata(metadata)
            
            self.logger.debug(f"Metadaten extrahiert: {len(metadata)} Felder")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Fehler bei Metadaten-Extraktion: {str(e)}")
            # Fallback: Minimale Metadaten
            return {
                "filename": pdf_path.name,
                "file_path": str(pdf_path),
                "extraction_error": str(e),
                "extraction_timestamp": datetime.now().isoformat(),
            }
    
    def _extract_file_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extrahiert Metadaten aus dem Dateisystem.
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            
        Returns:
            Dictionary mit Datei-Metadaten
        """
        try:
            stat = pdf_path.stat()
            
            return {
                "filename": pdf_path.name,
                "file_path": str(pdf_path),
                "file_size_bytes": stat.st_size,
                "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                "file_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "file_created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "file_extension": pdf_path.suffix.lower(),
            }
        except Exception as e:
            self.logger.warning(f"Fehler bei Datei-Metadaten: {str(e)}")
            return {
                "filename": pdf_path.name,
                "file_path": str(pdf_path),
            }
    
    def _extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extrahiert Metadaten aus PDF-Properties.
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            
        Returns:
            Dictionary mit PDF-Metadaten
        """
        try:
            # PyPDF2/pypdf für PDF-Metadaten
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                try:
                    from pypdf import PdfReader
                except ImportError:
                    self.logger.warning("Weder PyPDF2 noch pypdf verfügbar")
                    return {}
            
            reader = PdfReader(str(pdf_path))
            pdf_info = reader.metadata or {}
            
            # PDF-Metadaten normalisieren
            metadata = {}
            
            # Standard PDF-Felder
            if '/Title' in pdf_info:
                metadata['pdf_title'] = str(pdf_info['/Title']).strip()
            if '/Author' in pdf_info:
                metadata['pdf_author'] = str(pdf_info['/Author']).strip()
            if '/Subject' in pdf_info:
                metadata['pdf_subject'] = str(pdf_info['/Subject']).strip()
            if '/Creator' in pdf_info:
                metadata['pdf_creator'] = str(pdf_info['/Creator']).strip()
            if '/Producer' in pdf_info:
                metadata['pdf_producer'] = str(pdf_info['/Producer']).strip()
            if '/CreationDate' in pdf_info:
                metadata['pdf_creation_date'] = str(pdf_info['/CreationDate'])
            if '/ModDate' in pdf_info:
                metadata['pdf_modification_date'] = str(pdf_info['/ModDate'])
            
            # Seitenzahl
            metadata['pdf_page_count'] = len(reader.pages)
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Fehler bei PDF-Metadaten-Extraktion: {str(e)}")
            return {}
    
    def _extract_docling_metadata(self, docling_doc: Any) -> Dict[str, Any]:
        """
        Extrahiert Metadaten aus Docling-Document.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            Dictionary mit Docling-Metadaten
        """
        try:
            metadata = {}
            
            # TODO: Echte Docling-Metadaten-API verwenden
            # Für jetzt: Dummy-Implementierung
            
            # Docling kann in Zukunft Titel, Autor, etc. direkt extrahieren
            if hasattr(docling_doc, 'metadata'):
                docling_meta = docling_doc.metadata
                if docling_meta:
                    metadata['docling_title'] = getattr(docling_meta, 'title', None)
                    metadata['docling_authors'] = getattr(docling_meta, 'authors', None)
                    metadata['docling_language'] = getattr(docling_meta, 'language', None)
            
            # Strukturelle Informationen
            if hasattr(docling_doc, 'pages'):
                metadata['docling_page_count'] = len(docling_doc.pages)
                
                # Zähle verschiedene Element-Typen
                element_counts = {}
                for page in docling_doc.pages:
                    if hasattr(page, 'elements'):
                        for element in page.elements:
                            element_type = getattr(element, 'type', 'unknown')
                            element_counts[element_type] = element_counts.get(element_type, 0) + 1
                
                # Konvertiere Element-Counts zu String für ChromaDB-Kompatibilität
                metadata['docling_element_counts'] = str(element_counts)
                
                # Zusätzlich einzelne Counts als separate Felder
                for element_type, count in element_counts.items():
                    metadata[f'docling_{element_type}_count'] = count
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Fehler bei Docling-Metadaten: {str(e)}")
            return {}
    
    def _extract_content_metadata(self, docling_doc: Any) -> Dict[str, Any]:
        """
        Extrahiert Metadaten heuristisch aus dem Dokumentinhalt.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            Dictionary mit inhalts-basierten Metadaten
        """
        try:
            metadata = {}
            
            # Sammle Text von den ersten Seiten für Heuristiken
            first_page_text = self._get_first_page_text(docling_doc)
            
            if first_page_text:
                # Titel-Heuristik (größte Schrift auf erster Seite)
                potential_title = self._extract_title_heuristic(first_page_text)
                if potential_title:
                    metadata['heuristic_title'] = potential_title
                
                # DOI-Erkennung
                doi_match = self.doi_pattern.search(first_page_text)
                if doi_match:
                    metadata['doi'] = doi_match.group()
                
                # ISBN-Erkennung
                isbn_match = self.isbn_pattern.search(first_page_text)
                if isbn_match:
                    metadata['isbn'] = isbn_match.group()
                
                # Jahr-Erkennung
                year_matches = self.year_pattern.findall(first_page_text)
                if year_matches:
                    # Nehme das letzte gefundene Jahr (meist das Publikationsjahr)
                    metadata['heuristic_year'] = int(year_matches[-1])
                
                # DIN-Norm-Erkennung (für deutsche Normen)
                din_match = self.din_pattern.search(first_page_text)
                if din_match:
                    metadata['din_norm'] = din_match.group()
                
                # Sprache-Heuristik (einfach)
                language = self._detect_language_simple(first_page_text)
                if language:
                    metadata['heuristic_language'] = language
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Fehler bei Content-Metadaten: {str(e)}")
            return {}
    
    def _get_first_page_text(self, docling_doc: Any) -> str:
        """
        Extrahiert Text von der ersten Seite für Heuristiken.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            Text der ersten Seite
        """
        try:
            if hasattr(docling_doc, 'pages') and docling_doc.pages:
                first_page = docling_doc.pages[0]
                if hasattr(first_page, 'elements'):
                    texts = []
                    for element in first_page.elements:
                        element_text = getattr(element, 'text', '')
                        if element_text.strip():
                            texts.append(element_text)
                    return '\n'.join(texts)
            
            # Fallback für Dummy-Objekte
            return f"Dummy-Text für {getattr(docling_doc, 'source_path', 'unbekannt')}"
            
        except Exception:
            return ""
    
    def _extract_title_heuristic(self, text: str) -> Optional[str]:
        """
        Versucht den Titel heuristisch zu extrahieren.
        
        Args:
            text: Text der ersten Seite
            
        Returns:
            Potentieller Titel oder None
        """
        lines = text.split('\n')
        
        # Suche nach der ersten nicht-leeren Zeile, die wie ein Titel aussieht
        for line in lines[:10]:  # Nur erste 10 Zeilen betrachten
            line = line.strip()
            if len(line) > 10 and len(line) < 200:  # Plausible Titel-Länge
                # Keine Zahlen am Anfang (wahrscheinlich keine Seitenzahl)
                if not line[0].isdigit():
                    # Nicht nur Großbuchstaben (wahrscheinlich kein Header)
                    if not line.isupper() or len(line) < 50:
                        return line
        
        return None
    
    def _detect_language_simple(self, text: str) -> Optional[str]:
        """
        Einfache Sprach-Erkennung basierend auf häufigen Wörtern.
        
        Args:
            text: Zu analysierender Text
            
        Returns:
            Sprach-Code oder None
        """
        text_lower = text.lower()
        
        # Deutsche Indikatoren
        german_words = ['der', 'die', 'das', 'und', 'oder', 'mit', 'von', 'zu', 'auf', 'für', 'ist', 'sind', 'wird', 'werden']
        german_count = sum(1 for word in german_words if word in text_lower)
        
        # Englische Indikatoren
        english_words = ['the', 'and', 'or', 'with', 'from', 'to', 'on', 'for', 'is', 'are', 'will', 'be', 'of', 'in']
        english_count = sum(1 for word in english_words if word in text_lower)
        
        if german_count > english_count and german_count > 2:
            return 'de'
        elif english_count > german_count and english_count > 2:
            return 'en'
        
        return None
    
    def _normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalisiert und bereinigt die extrahierten Metadaten.
        
        Args:
            metadata: Rohe Metadaten
            
        Returns:
            Bereinigte Metadaten
        """
        normalized = {}
        
        # Titel-Konsolidierung (bevorzuge PDF-Titel, dann heuristischen)
        title = metadata.get('pdf_title') or metadata.get('heuristic_title') or metadata.get('docling_title')
        if title:
            normalized['title'] = str(title).strip()
        
        # Autor-Konsolidierung
        author = metadata.get('pdf_author') or metadata.get('docling_authors')
        if author:
            normalized['author'] = str(author).strip()
        
        # Jahr-Konsolidierung
        year = metadata.get('heuristic_year')
        if year:
            normalized['year'] = year
        
        # Sprache-Konsolidierung
        language = metadata.get('docling_language') or metadata.get('heuristic_language')
        if language:
            normalized['language'] = language
        
        # Alle anderen Metadaten übernehmen
        for key, value in metadata.items():
            if key not in normalized and value is not None:
                normalized[key] = value
        
        # Extraktion-Zeitstempel hinzufügen
        normalized['extraction_timestamp'] = datetime.now().isoformat()
        
        return normalized
    
    def extract_minimal(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Schnelle Extraktion nur der wichtigsten Metadaten.
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            
        Returns:
            Dictionary mit minimalen Metadaten
        """
        try:
            file_metadata = self._extract_file_metadata(pdf_path)
            pdf_metadata = self._extract_pdf_metadata(pdf_path)
            
            return {
                **file_metadata,
                'title': pdf_metadata.get('pdf_title'),
                'author': pdf_metadata.get('pdf_author'),
                'page_count': pdf_metadata.get('pdf_page_count'),
            }
        except Exception as e:
            return {
                "filename": pdf_path.name,
                "file_path": str(pdf_path),
                "error": str(e)
            }


# Hilfsfunktionen für externe Nutzung
def extract_metadata_simple(pdf_path: str | Path) -> Dict[str, Any]:
    """
    Einfache Metadaten-Extraktion ohne Docling-Document.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        
    Returns:
        Dictionary mit Metadaten
    """
    extractor = MetadataExtractor()
    return extractor.extract_minimal(Path(pdf_path))


def has_metadata(pdf_path: str | Path) -> bool:
    """
    Prüft, ob eine PDF-Datei Metadaten enthält.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        
    Returns:
        True wenn Metadaten vorhanden
    """
    metadata = extract_metadata_simple(pdf_path)
    return bool(metadata.get('title') or metadata.get('author')) 