"""
MetadataExtractor
-----------------
Extrahiert Metadaten aus PDF-Dokumenten und Docling-Parsing-Ergebnissen.

Quellen:
- PDF-Metadaten (Titel, Autor, Erstellungsdatum)
- Docling-Metadaten (echte API-Integration)
- Heuristische Extraktion (Titel aus erster Seite, DOI-Erkennung)
- Datei-basierte Metadaten (Pfad, Größe, Änderungsdatum)
- Dateiname-basierte Heuristiken
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
import re
import json
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class DocumentMetadata:
    """Strukturierte Repräsentation von Dokument-Metadaten."""
    # Basis-Informationen
    filename: str
    file_path: str
    file_size_bytes: int
    file_size_mb: float
    
    # Titel und Autor
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    
    # Publikations-Informationen
    year: Optional[int] = None
    language: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    
    # Norm-spezifisch
    din_norm: Optional[str] = None
    iso_norm: Optional[str] = None
    
    # Technische Details
    page_count: Optional[int] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    
    # Docling-spezifisch
    element_counts: Optional[Dict[str, int]] = None
    has_tables: bool = False
    has_images: bool = False
    has_formulas: bool = False
    
    # Qualitäts-Indikatoren
    extraction_confidence: float = 1.0
    extraction_timestamp: Optional[str] = None
    extraction_sources: Optional[List[str]] = None


class MetadataExtractor:
    """
    Extrahiert umfassende Metadaten aus PDF-Dokumenten.
    
    Kombiniert verschiedene Quellen für maximale Informationsausbeute.
    """
    
    def __init__(self):
        """Initialisiert den MetadataExtractor."""
        self.logger = logging.getLogger(__name__)
        
        # Verbesserte Regex-Pattern
        self.doi_pattern = re.compile(
            r'(?:doi:?\s*)?10\.\d{4,}/[^\s\]\)]+', 
            re.IGNORECASE
        )
        self.isbn_pattern = re.compile(
            r'ISBN[-\s]?(?:97[89][-\s]?)?(?:\d[-\s]?){9}[\dxX]', 
            re.IGNORECASE
        )
        
        # Kontextuelles Jahr-Pattern (mit Publikations-Kontext)
        self.year_pattern = re.compile(
            r'(?:©|\(c\)|copyright|published?|edition|ausgabe|jahr)?\s*(?:in\s+)?(?:19|20)\d{2}',
            re.IGNORECASE
        )
        self.simple_year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        
        # Erweiterte Norm-Pattern
        self.din_pattern = re.compile(
            r'DIN\s+(?:EN\s+)?(?:ISO\s+)?\d+(?:[-:]\d+)*(?::\d{4})?', 
            re.IGNORECASE
        )
        self.iso_pattern = re.compile(
            r'ISO\s+\d+(?:[-:]\d+)*(?::\d{4})?', 
            re.IGNORECASE
        )
        
        # Titel-Heuristiken
        self.title_indicators = [
            r'^[A-ZÄÖÜ][^.!?]*$',  # Großbuchstabe am Anfang, keine Satzzeichen
            r'^\d+\.?\s+[A-ZÄÖÜ]',  # Nummerierung + Großbuchstabe
        ]
    
    def extract(self, docling_doc: Any, pdf_path: Path) -> Dict[str, Any]:
        """
        Hauptmethode: Extrahiert alle verfügbaren Metadaten.
        
        Args:
            docling_doc: Docling Document-Objekt
            pdf_path: Pfad zur PDF-Datei
            
        Returns:
            Dictionary mit allen extrahierten Metadaten
        """
        extraction_sources = []
        
        try:
            # Basis-Metadaten aus Datei
            file_metadata = self._extract_file_metadata(pdf_path)
            extraction_sources.append("file_system")
            
            # Strukturierte Metadaten-Objekt erstellen
            doc_metadata = DocumentMetadata(
                filename=file_metadata["filename"],
                file_path=file_metadata["file_path"],
                file_size_bytes=file_metadata["file_size_bytes"],
                file_size_mb=file_metadata["file_size_mb"],
                extraction_timestamp=datetime.now().isoformat(),
                extraction_sources=extraction_sources
            )
            
            # 1. PDF-Metadaten (robuste Extraktion)
            pdf_metadata = self._extract_pdf_metadata(pdf_path)
            if pdf_metadata:
                extraction_sources.append("pdf_properties")
                self._merge_pdf_metadata(doc_metadata, pdf_metadata)
            
            # 2. Docling-Metadaten (echte API-Integration)
            docling_metadata = self._extract_docling_metadata(docling_doc)
            if docling_metadata:
                extraction_sources.append("docling_analysis")
                self._merge_docling_metadata(doc_metadata, docling_metadata)
            
            # 3. Heuristische Extraktion aus Inhalt
            content_metadata = self._extract_content_metadata(docling_doc)
            if content_metadata:
                extraction_sources.append("content_heuristics")
                self._merge_content_metadata(doc_metadata, content_metadata)
            
            # 4. Dateiname-basierte Heuristiken
            filename_metadata = self._extract_filename_metadata(pdf_path)
            if filename_metadata:
                extraction_sources.append("filename_heuristics")
                self._merge_filename_metadata(doc_metadata, filename_metadata)
            
            # 5. Qualitätsbewertung und Finalisierung
            doc_metadata.extraction_sources = extraction_sources
            doc_metadata.extraction_confidence = self._calculate_confidence(doc_metadata)
            
            # In Dictionary umwandeln für Rückgabe
            result = asdict(doc_metadata)
            
            # Zusätzliche Kompatibilitäts-Felder für Legacy-Code
            result.update(self._create_legacy_fields(doc_metadata))
            
            self.logger.debug(f"Metadaten extrahiert: {len(result)} Felder aus {len(extraction_sources)} Quellen")
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei Metadaten-Extraktion: {str(e)}")
            # Fallback: Minimale Metadaten
            return {
                "filename": pdf_path.name,
                "file_path": str(pdf_path),
                "extraction_error": str(e),
                "extraction_timestamp": datetime.now().isoformat(),
                "extraction_confidence": 0.1,
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
        Robuste Extraktion von PDF-Metadaten mit Fallback-Strategien.
        
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
            
            # Mehrere Versuche mit verschiedenen Strict-Modi
            reader = None
            for strict_mode in [True, False]:
                try:
                    reader = PdfReader(str(pdf_path), strict=strict_mode)
                    break
                except Exception as e:
                    if strict_mode:
                        self.logger.debug(f"Strict-Modus fehlgeschlagen: {e}")
                        continue
                    else:
                        raise
            
            if not reader:
                return {}
            
            pdf_info = reader.metadata or {}
            
            # PDF-Metadaten normalisieren und bereinigen
            metadata = {}
            
            # Standard PDF-Felder mit Bereinigung
            for pdf_key, meta_key in [
                ('/Title', 'title'),
                ('/Author', 'author'), 
                ('/Subject', 'subject'),
                ('/Creator', 'creator'),
                ('/Producer', 'producer'),
            ]:
                if pdf_key in pdf_info:
                    value = str(pdf_info[pdf_key]).strip()
                    # Leere oder nur Whitespace-Werte ignorieren
                    if value and not value.isspace():
                        # Häufige PDF-Artefakte entfernen
                        value = self._clean_pdf_text(value)
                        if value:
                            metadata[meta_key] = value
            
            # Datum-Felder
            for pdf_key, meta_key in [
                ('/CreationDate', 'creation_date'),
                ('/ModDate', 'modification_date'),
            ]:
                if pdf_key in pdf_info:
                    date_str = str(pdf_info[pdf_key])
                    parsed_date = self._parse_pdf_date(date_str)
                    if parsed_date:
                        metadata[meta_key] = parsed_date
            
            # Seitenzahl
            try:
                metadata['page_count'] = len(reader.pages)
            except Exception as e:
                self.logger.warning(f"Fehler beim Zählen der Seiten: {e}")
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Fehler bei PDF-Metadaten-Extraktion: {str(e)}")
            return {}
    
    def _extract_docling_metadata(self, docling_doc: Any) -> Dict[str, Any]:
        """
        Extrahiert Metadaten aus Docling-Document mit echter API-Integration.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            Dictionary mit Docling-Metadaten
        """
        try:
            metadata = {}
            
            # Echte Docling-Metadaten-API (verschiedene Attribute versuchen)
            for meta_attr in ['metadata', 'meta', 'document_meta']:
                if hasattr(docling_doc, meta_attr):
                    docling_meta = getattr(docling_doc, meta_attr)
                    if docling_meta:
                        # Standard-Metadaten extrahieren
                        for attr, key in [
                            ('title', 'title'),
                            ('authors', 'authors'),
                            ('author', 'author'),
                            ('language', 'language'),
                            ('subject', 'subject'),
                            ('keywords', 'keywords'),
                        ]:
                            if hasattr(docling_meta, attr):
                                value = getattr(docling_meta, attr)
                                if value:
                                    metadata[key] = value
                        break
            
            # Strukturelle Informationen aus Docling-Analyse
            if hasattr(docling_doc, 'pages'):
                metadata['page_count'] = len(docling_doc.pages)
                
                # Detaillierte Element-Analyse
                element_counts = {}
                total_elements = 0
                
                for page in docling_doc.pages:
                    if hasattr(page, 'elements'):
                        for element in page.elements:
                            element_type = getattr(element, 'type', 'unknown')
                            element_counts[element_type] = element_counts.get(element_type, 0) + 1
                            total_elements += 1
                
                # Strukturierte Element-Counts (nicht als String)
                metadata['element_counts'] = element_counts
                metadata['total_elements'] = total_elements
                
                # Boolean-Flags für wichtige Element-Typen
                metadata['has_tables'] = element_counts.get('table', 0) > 0
                metadata['has_images'] = any(
                    element_counts.get(t, 0) > 0 
                    for t in ['image', 'figure', 'picture']
                )
                metadata['has_formulas'] = any(
                    element_counts.get(t, 0) > 0 
                    for t in ['formula', 'equation', 'math']
                )
                metadata['has_headings'] = any(
                    element_counts.get(t, 0) > 0 
                    for t in ['heading', 'title', 'h1', 'h2', 'h3']
                )
                
                # Strukturqualität bewerten
                structure_score = self._calculate_structure_score(element_counts)
                metadata['structure_score'] = structure_score
            
            # Layout-Informationen (falls verfügbar)
            if hasattr(docling_doc, 'layout'):
                layout = docling_doc.layout
                if hasattr(layout, 'reading_order'):
                    metadata['has_reading_order'] = True
                if hasattr(layout, 'columns'):
                    metadata['column_count'] = len(layout.columns)
            
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
                
                # Kontextuelle Jahr-Erkennung
                year = self._extract_year_contextual(first_page_text)
                if year:
                    metadata['year'] = year
                
                # Norm-Erkennung (DIN, ISO, etc.)
                norms = self._extract_norms(first_page_text)
                if norms:
                    metadata.update(norms)
                
                # Verbesserte Sprach-Erkennung
                language = self._detect_language_advanced(first_page_text)
                if language:
                    metadata['language'] = language
            
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
    
    def _detect_language_advanced(self, text: str) -> Optional[str]:
        """
        Verbesserte Sprach-Erkennung mit mehreren Indikatoren.
        
        Args:
            text: Zu analysierender Text
            
        Returns:
            Sprach-Code oder None
        """
        if len(text) < 50:  # Zu wenig Text für zuverlässige Erkennung
            return None
        
        text_lower = text.lower()
        
        # Erweiterte Wortlisten mit Gewichtung
        language_indicators = {
            'de': {
                'common': ['der', 'die', 'das', 'und', 'oder', 'mit', 'von', 'zu', 'auf', 'für', 'ist', 'sind', 'wird', 'werden'],
                'specific': ['dass', 'durch', 'nach', 'über', 'unter', 'zwischen', 'während', 'wegen', 'trotz'],
                'technical': ['norm', 'standard', 'verfahren', 'anforderung', 'prüfung', 'bestimmung'],
            },
            'en': {
                'common': ['the', 'and', 'or', 'with', 'from', 'to', 'on', 'for', 'is', 'are', 'will', 'be', 'of', 'in'],
                'specific': ['that', 'through', 'after', 'over', 'under', 'between', 'during', 'because', 'despite'],
                'technical': ['standard', 'specification', 'procedure', 'requirement', 'test', 'determination'],
            }
        }
        
        scores = {}
        
        for lang, categories in language_indicators.items():
            score = 0
            for category, words in categories.items():
                weight = {'common': 1, 'specific': 2, 'technical': 3}[category]
                for word in words:
                    # Wort-Grenzen beachten
                    if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                        score += weight
            scores[lang] = score
        
        # Zusätzliche Heuristiken
        # Deutsche Umlaute
        if re.search(r'[äöüß]', text_lower):
            scores['de'] = scores.get('de', 0) + 5
        
        # Englische Artikel-Häufigkeit
        the_count = len(re.findall(r'\bthe\b', text_lower))
        if the_count > 3:
            scores['en'] = scores.get('en', 0) + the_count
        
        # Beste Sprache ermitteln
        if not scores:
            return None
        
        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]
        
        # Mindest-Confidence erforderlich
        if best_score < 3:
            return None
        
        # Deutlicher Gewinner erforderlich
        other_scores = [s for lang, s in scores.items() if lang != best_lang]
        if other_scores and best_score < max(other_scores) * 1.5:
            return None
        
        return best_lang
    
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
    
    # Neue Hilfsmethoden für verbesserte Metadaten-Extraktion
    
    def _clean_pdf_text(self, text: str) -> str:
        """Bereinigt PDF-Text von häufigen Artefakten."""
        if not text:
            return ""
        
        # Häufige PDF-Artefakte entfernen
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Mehrfache Leerzeichen reduzieren
        text = re.sub(r'\s+', ' ', text)
        
        # Führende/nachfolgende Leerzeichen
        text = text.strip()
        
        return text
    
    def _parse_pdf_date(self, date_str: str) -> Optional[str]:
        """Parst PDF-Datum in ISO-Format."""
        try:
            # PDF-Datum-Format: D:YYYYMMDDHHmmSSOHH'mm'
            if date_str.startswith('D:'):
                date_str = date_str[2:]
            
            # Nur Jahr-Monat-Tag extrahieren
            if len(date_str) >= 8:
                year = date_str[:4]
                month = date_str[4:6]
                day = date_str[6:8]
                
                # Validierung
                if year.isdigit() and month.isdigit() and day.isdigit():
                    return f"{year}-{month}-{day}"
        except Exception:
            pass
        
        return None
    
    def _extract_year_contextual(self, text: str) -> Optional[int]:
        """Extrahiert Jahr mit Kontext-Prüfung."""
        # Zuerst kontextuelle Pattern versuchen
        contextual_match = self.year_pattern.search(text)
        if contextual_match:
            year_str = re.search(r'(19|20)\d{2}', contextual_match.group())
            if year_str:
                year = int(year_str.group())
                if 1900 <= year <= 2030:  # Plausibilitätsprüfung
                    return year
        
        # Fallback: Alle Jahre finden und das wahrscheinlichste wählen
        all_years = self.simple_year_pattern.findall(text)
        if all_years:
            years = [int(y) for y in all_years if 1900 <= int(y) <= 2030]
            if years:
                # Bevorzuge neuere Jahre (wahrscheinlicher Publikationsjahr)
                return max(years)
        
        return None
    
    def _extract_norms(self, text: str) -> Dict[str, str]:
        """Extrahiert verschiedene Norm-Identifikatoren."""
        norms = {}
        
        # DIN-Normen
        din_match = self.din_pattern.search(text)
        if din_match:
            norms['din_norm'] = din_match.group().strip()
        
        # ISO-Normen
        iso_match = self.iso_pattern.search(text)
        if iso_match:
            norms['iso_norm'] = iso_match.group().strip()
        
        return norms
    
    def _extract_filename_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extrahiert Metadaten aus dem Dateinamen."""
        filename = pdf_path.stem  # Ohne Erweiterung
        metadata = {}
        
        # Jahr aus Dateiname
        year_match = self.simple_year_pattern.search(filename)
        if year_match:
            year = int(year_match.group())
            if 1900 <= year <= 2030:
                metadata['filename_year'] = year
        
        # Norm-Nummern aus Dateiname
        din_match = self.din_pattern.search(filename)
        if din_match:
            metadata['filename_din'] = din_match.group()
        
        iso_match = self.iso_pattern.search(filename)
        if iso_match:
            metadata['filename_iso'] = iso_match.group()
        
        # Titel aus Dateiname (bereinigt)
        title_candidate = filename
        # Entferne häufige Präfixe/Suffixe
        title_candidate = re.sub(r'^(DIN|ISO|EN)[-_\s]*', '', title_candidate, flags=re.IGNORECASE)
        title_candidate = re.sub(r'[-_\s]*(draft|entwurf|final|v\d+)$', '', title_candidate, flags=re.IGNORECASE)
        title_candidate = re.sub(r'[-_]', ' ', title_candidate)
        title_candidate = title_candidate.strip()
        
        if len(title_candidate) > 5:
            metadata['filename_title'] = title_candidate
        
        return metadata
    
    def _calculate_structure_score(self, element_counts: Dict[str, int]) -> float:
        """Berechnet einen Struktur-Qualitätsscore."""
        if not element_counts:
            return 0.0
        
        total_elements = sum(element_counts.values())
        if total_elements == 0:
            return 0.0
        
        score = 0.0
        
        # Überschriften (wichtig für Struktur)
        heading_types = ['heading', 'title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        heading_count = sum(element_counts.get(t, 0) for t in heading_types)
        if heading_count > 0:
            score += min(0.4, heading_count / total_elements * 2)
        
        # Tabellen (strukturierte Daten)
        table_count = element_counts.get('table', 0)
        if table_count > 0:
            score += min(0.3, table_count / total_elements * 5)
        
        # Listen (strukturierte Aufzählungen)
        list_count = element_counts.get('list', 0) + element_counts.get('list_item', 0)
        if list_count > 0:
            score += min(0.2, list_count / total_elements * 3)
        
        # Vielfalt der Element-Typen
        type_diversity = len(element_counts)
        if type_diversity > 3:
            score += min(0.1, (type_diversity - 3) * 0.02)
        
        return min(1.0, score)
    
    def _calculate_confidence(self, metadata: DocumentMetadata) -> float:
        """Berechnet Confidence-Score für die Metadaten-Extraktion."""
        confidence = 0.0
        
        # Basis-Confidence für erfolgreiche Extraktion
        confidence += 0.2
        
        # Titel vorhanden
        if metadata.title:
            confidence += 0.3
        
        # Autor vorhanden
        if metadata.author:
            confidence += 0.2
        
        # Jahr vorhanden
        if metadata.year:
            confidence += 0.1
        
        # Strukturelle Informationen
        if metadata.element_counts:
            confidence += 0.1
        
        # Mehrere Quellen
        if metadata.extraction_sources and len(metadata.extraction_sources) > 2:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _merge_pdf_metadata(self, doc_metadata: DocumentMetadata, pdf_data: Dict[str, Any]):
        """Fügt PDF-Metadaten in DocumentMetadata ein."""
        if 'title' in pdf_data and not doc_metadata.title:
            doc_metadata.title = pdf_data['title']
        if 'author' in pdf_data and not doc_metadata.author:
            doc_metadata.author = pdf_data['author']
        if 'subject' in pdf_data and not doc_metadata.subject:
            doc_metadata.subject = pdf_data['subject']
        if 'creator' in pdf_data:
            doc_metadata.creator = pdf_data['creator']
        if 'producer' in pdf_data:
            doc_metadata.producer = pdf_data['producer']
        if 'page_count' in pdf_data and not doc_metadata.page_count:
            doc_metadata.page_count = pdf_data['page_count']
    
    def _merge_docling_metadata(self, doc_metadata: DocumentMetadata, docling_data: Dict[str, Any]):
        """Fügt Docling-Metadaten in DocumentMetadata ein."""
        if 'title' in docling_data and not doc_metadata.title:
            doc_metadata.title = docling_data['title']
        if 'author' in docling_data and not doc_metadata.author:
            doc_metadata.author = docling_data['author']
        if 'language' in docling_data and not doc_metadata.language:
            doc_metadata.language = docling_data['language']
        if 'page_count' in docling_data and not doc_metadata.page_count:
            doc_metadata.page_count = docling_data['page_count']
        if 'element_counts' in docling_data:
            doc_metadata.element_counts = docling_data['element_counts']
        if 'has_tables' in docling_data:
            doc_metadata.has_tables = docling_data['has_tables']
        if 'has_images' in docling_data:
            doc_metadata.has_images = docling_data['has_images']
        if 'has_formulas' in docling_data:
            doc_metadata.has_formulas = docling_data['has_formulas']
    
    def _merge_content_metadata(self, doc_metadata: DocumentMetadata, content_data: Dict[str, Any]):
        """Fügt Content-Heuristik-Metadaten in DocumentMetadata ein."""
        if 'title' in content_data and not doc_metadata.title:
            doc_metadata.title = content_data['title']
        if 'year' in content_data and not doc_metadata.year:
            doc_metadata.year = content_data['year']
        if 'language' in content_data and not doc_metadata.language:
            doc_metadata.language = content_data['language']
        if 'doi' in content_data:
            doc_metadata.doi = content_data['doi']
        if 'isbn' in content_data:
            doc_metadata.isbn = content_data['isbn']
        if 'din_norm' in content_data:
            doc_metadata.din_norm = content_data['din_norm']
        if 'iso_norm' in content_data:
            doc_metadata.iso_norm = content_data['iso_norm']
    
    def _merge_filename_metadata(self, doc_metadata: DocumentMetadata, filename_data: Dict[str, Any]):
        """Fügt Dateiname-Metadaten in DocumentMetadata ein (niedrigste Priorität)."""
        if 'filename_title' in filename_data and not doc_metadata.title:
            doc_metadata.title = filename_data['filename_title']
        if 'filename_year' in filename_data and not doc_metadata.year:
            doc_metadata.year = filename_data['filename_year']
        if 'filename_din' in filename_data and not doc_metadata.din_norm:
            doc_metadata.din_norm = filename_data['filename_din']
        if 'filename_iso' in filename_data and not doc_metadata.iso_norm:
            doc_metadata.iso_norm = filename_data['filename_iso']
    
    def _create_legacy_fields(self, doc_metadata: DocumentMetadata) -> Dict[str, Any]:
        """Erstellt Legacy-Felder für Rückwärtskompatibilität."""
        legacy = {}
        
        # Alte Feldnamen für Kompatibilität
        if doc_metadata.title:
            legacy['pdf_title'] = doc_metadata.title
            legacy['heuristic_title'] = doc_metadata.title
        if doc_metadata.author:
            legacy['pdf_author'] = doc_metadata.author
        if doc_metadata.year:
            legacy['heuristic_year'] = doc_metadata.year
        if doc_metadata.language:
            legacy['heuristic_language'] = doc_metadata.language
        if doc_metadata.page_count:
            legacy['pdf_page_count'] = doc_metadata.page_count
            legacy['docling_page_count'] = doc_metadata.page_count
        
        # Element-Counts als String (für ChromaDB-Kompatibilität)
        if doc_metadata.element_counts:
            legacy['docling_element_counts'] = json.dumps(doc_metadata.element_counts)
            # Einzelne Counts
            for element_type, count in doc_metadata.element_counts.items():
                legacy[f'docling_{element_type}_count'] = count
        
        return legacy


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