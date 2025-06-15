"""
DoclingMetadataExtractor
------------------------
Schlanker Wrapper um Doclings native Metadaten-Funktionalität.

Nutzt primär docling_doc.meta anstatt eigene Heuristiken.
Minimale Ergänzungen nur für spezielle Fälle (DIN-Normen, Dateisystem).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import re
from datetime import datetime


class DoclingMetadataExtractor:
    """
    Schlanker Metadaten-Extraktor, der primär auf Docling-Metadaten setzt.
    
    Nutzt docling_doc.meta als Hauptquelle und ergänzt nur minimal.
    """
    
    def __init__(self):
        """Initialisiert den DoclingMetadataExtractor."""
        self.logger = logging.getLogger(__name__)
        
        # Minimale Regex-Pattern nur für spezielle Fälle
        self.din_pattern = re.compile(r'DIN\s+(?:EN\s+)?(?:ISO\s+)?\d+(?:[-:]\d+)*', re.IGNORECASE)
        self.iso_pattern = re.compile(r'ISO\s+\d+(?:[-:]\d+)*', re.IGNORECASE)
    
    def extract(self, docling_doc: Any, pdf_path: Path) -> Dict[str, Any]:
        """
        Hauptmethode: Extrahiert Metadaten primär aus docling_doc.meta.
        
        Args:
            docling_doc: Docling Document-Objekt
            pdf_path: Pfad zur PDF-Datei
            
        Returns:
            Dictionary mit Metadaten
        """
        try:
            # 1. Basis-Metadaten aus Dateisystem
            metadata = self._extract_file_metadata(pdf_path)
            
            # 2. Docling-Metadaten (Hauptquelle)
            docling_meta = self._extract_docling_metadata(docling_doc)
            metadata.update(docling_meta)
            
            # 3. Minimale Ergänzungen für spezielle Fälle
            special_meta = self._extract_special_metadata(docling_doc)
            metadata.update(special_meta)
            
            # 4. Finalisierung
            metadata.update({
                "extraction_timestamp": datetime.now().isoformat(),
                "extraction_source": "docling_native",
                "docling_available": True,
            })
            
            self.logger.debug(f"Metadaten extrahiert: {len(metadata)} Felder")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Fehler bei Metadaten-Extraktion: {str(e)}")
            return self._create_fallback_metadata(pdf_path, str(e))
    
    def _extract_file_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extrahiert Basis-Metadaten aus dem Dateisystem."""
        try:
            stat = pdf_path.stat()
            return {
                "filename": pdf_path.name,
                "file_path": str(pdf_path),
                "file_size_bytes": stat.st_size,
                "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                "file_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        except Exception as e:
            self.logger.warning(f"Fehler bei Datei-Metadaten: {e}")
            return {
                "filename": pdf_path.name,
                "file_path": str(pdf_path),
            }
    
    def _extract_docling_metadata(self, docling_doc: Any) -> Dict[str, Any]:
        """
        Extrahiert Metadaten direkt aus docling_doc.meta.
        
        Dies ist die Hauptquelle - nutzt Doclings native Metadaten-Extraktion.
        """
        metadata = {}
        
        try:
            # Versuche verschiedene Metadaten-Attribute
            docling_meta = None
            for attr in ['meta', 'metadata', 'document_meta']:
                if hasattr(docling_doc, attr):
                    docling_meta = getattr(docling_doc, attr)
                    if docling_meta:
                        self.logger.debug(f"Docling-Metadaten gefunden in: {attr}")
                        break
            
            if docling_meta:
                # Standard-Metadaten aus Docling übernehmen
                for docling_key, meta_key in [
                    ('title', 'title'),
                    ('authors', 'author'),
                    ('author', 'author'),
                    ('subject', 'subject'),
                    ('language', 'language'),
                    ('keywords', 'keywords'),
                    ('creator', 'creator'),
                    ('producer', 'producer'),
                    ('creation_date', 'creation_date'),
                    ('modification_date', 'modification_date'),
                ]:
                    if hasattr(docling_meta, docling_key):
                        value = getattr(docling_meta, docling_key)
                        if value:
                            metadata[meta_key] = value
                    elif isinstance(docling_meta, dict) and docling_key in docling_meta:
                        value = docling_meta[docling_key]
                        if value:
                            metadata[meta_key] = value
            
            # Strukturelle Informationen aus Docling-Analyse
            if hasattr(docling_doc, 'pages'):
                metadata['page_count'] = len(docling_doc.pages)
                
                # Element-Analyse (vereinfacht)
                element_counts = {}
                for page in docling_doc.pages:
                    if hasattr(page, 'elements'):
                        for element in page.elements:
                            element_type = getattr(element, 'type', 'text')
                            element_counts[element_type] = element_counts.get(element_type, 0) + 1
                
                # Boolean-Flags für wichtige Element-Typen
                metadata['has_tables'] = any('table' in t.lower() for t in element_counts.keys())
                metadata['has_images'] = any('image' in t.lower() or 'figure' in t.lower() for t in element_counts.keys())
                metadata['has_formulas'] = any('formula' in t.lower() or 'equation' in t.lower() for t in element_counts.keys())
                metadata['element_types'] = list(element_counts.keys())
                metadata['total_elements'] = sum(element_counts.values())
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Fehler bei Docling-Metadaten: {e}")
            return {}
    
    def _extract_special_metadata(self, docling_doc: Any) -> Dict[str, Any]:
        """
        Minimale Ergänzungen für spezielle Fälle (DIN-Normen, etc.).
        
        Nur das, was Docling nicht automatisch erkennt.
        """
        metadata = {}
        
        try:
            # Text der ersten Seite für spezielle Pattern
            first_page_text = self._get_first_page_text(docling_doc)
            
            if first_page_text:
                # DIN-Normen (spezifisch für unser Projekt)
                din_match = self.din_pattern.search(first_page_text)
                if din_match:
                    metadata['din_norm'] = din_match.group().strip()
                
                # ISO-Normen
                iso_match = self.iso_pattern.search(first_page_text)
                if iso_match:
                    metadata['iso_norm'] = iso_match.group().strip()
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Fehler bei speziellen Metadaten: {e}")
            return {}
    
    def _get_first_page_text(self, docling_doc: Any) -> str:
        """Extrahiert Text der ersten Seite für spezielle Pattern."""
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
            return ""
        except Exception:
            return ""
    
    def _create_fallback_metadata(self, pdf_path: Path, error_msg: str) -> Dict[str, Any]:
        """Erstellt minimale Fallback-Metadaten bei Fehlern."""
        return {
            "filename": pdf_path.name,
            "file_path": str(pdf_path),
            "extraction_error": error_msg,
            "extraction_timestamp": datetime.now().isoformat(),
            "extraction_source": "fallback",
            "docling_available": False,
        }


# Backward-Kompatibilität: Alias für alte MetadataExtractor-Klasse
MetadataExtractor = DoclingMetadataExtractor


# Hilfsfunktionen für externe Nutzung
def extract_metadata_simple(pdf_path: Path) -> Dict[str, Any]:
    """
    Einfache Metadaten-Extraktion ohne Docling-Document.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        
    Returns:
        Dictionary mit minimalen Metadaten
    """
    extractor = DoclingMetadataExtractor()
    return extractor._extract_file_metadata(pdf_path)


def has_metadata(pdf_path: Path) -> bool:
    """
    Prüft, ob eine PDF-Datei Metadaten enthält.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        
    Returns:
        True wenn Metadaten vorhanden
    """
    metadata = extract_metadata_simple(pdf_path)
    return bool(metadata.get('title') or metadata.get('author')) 