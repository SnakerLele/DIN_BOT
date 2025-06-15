"""
DoclingAdapter
--------------
Vereinfachte Schnittstelle: PDF -> DoclingReader -> LlamaIndex Documents

Diese Klasse nutzt LlamaIndex's native Docling-Integration:
1. DoclingReader für PDF-Parsing mit JSON-Export
2. DoclingNodeParser für optimale Node-Erstellung
3. Minimaler eigener Code, maximale Docling-Integration
"""

from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging
import time
import json

from llama_index.core import Document

# LlamaIndex Docling Integration
try:
    from llama_index.readers.docling import DoclingReader
    from llama_index.node_parser.docling import DoclingNodeParser
    LLAMAINDEX_DOCLING_AVAILABLE = True
except ImportError:
    DoclingReader = None
    DoclingNodeParser = None
    LLAMAINDEX_DOCLING_AVAILABLE = False

# Fallback: Docling direkt
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DocumentConverter = None
    DOCLING_AVAILABLE = False

from .quality import QualityAnalyzer
from .metadata import DoclingMetadataExtractor
from .utils import file_lock

# Globaler Logger
def _setup_logger() -> logging.Logger:
    """Konfiguriert Logger nur einmal global."""
    logger = logging.getLogger("docling_adapter")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class DoclingAdapter:
    """
    Vereinfachte PDF-Verarbeitung mit LlamaIndex's nativer Docling-Integration.
    
    Nutzt DoclingReader + DoclingNodeParser für optimale Ergebnisse.
    """
    
    # Standard-Konfiguration
    DEFAULT_CONFIG = {
        "export_type": "JSON",  # JSON für beste Metadaten, Markdown für Einfachheit
        "ocr_enabled": True,
        "table_extraction": True,
        "image_extraction": True,
        "formula_extraction": True,
        "layout_analysis": True,
        "timeout_seconds": 300,
        "preserve_table_html": True,  # Tabellen als HTML für besseres Retrieval
        "preserve_bounding_boxes": True,  # Bounding-Box-Metadaten für UI-Highlighting
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert den DoclingAdapter.
        
        Args:
            config: Konfigurationsdictionary
        """
        self.logger = _setup_logger()
        
        # Konfiguration zusammenführen
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Prüfe verfügbare Integrationen
        if not LLAMAINDEX_DOCLING_AVAILABLE:
            self.logger.warning(
                "LlamaIndex Docling-Integration nicht verfügbar. "
                "Installiere mit: pip install llama-index-readers-docling llama-index-node-parser-docling"
            )
            if not DOCLING_AVAILABLE:
                raise ValueError(
                    "Weder LlamaIndex-Docling noch Docling direkt verfügbar. "
                    "Installiere mit: pip install docling"
                )
        
        # Initialisiere Komponenten
        self.reader = self._create_docling_reader()
        self.node_parser = self._create_docling_node_parser()
        self.quality_analyzer = QualityAnalyzer()
        self.metadata_extractor = DoclingMetadataExtractor()
        
        # Statistiken
        self._last_stats: Dict[str, Any] = {}
        
        self.logger.info(f"DoclingAdapter initialisiert - LlamaIndex-Integration: {LLAMAINDEX_DOCLING_AVAILABLE}")

    def _create_docling_reader(self):
        """Erstellt konfigurierten DoclingReader."""
        if not LLAMAINDEX_DOCLING_AVAILABLE:
            return None
        
        # Export-Typ bestimmen
        export_type = DoclingReader.ExportType.JSON if self.config.get("export_type") == "JSON" else DoclingReader.ExportType.MARKDOWN
        
        # DoclingReader mit Konfiguration erstellen
        reader = DoclingReader(export_type=export_type)
        
        self.logger.info(f"DoclingReader erstellt - Export: {self.config.get('export_type')}")
        return reader

    def _create_docling_node_parser(self):
        """Erstellt konfigurierten DoclingNodeParser."""
        if not LLAMAINDEX_DOCLING_AVAILABLE:
            return None
        
        # Nur bei JSON-Export verwenden
        if self.config.get("export_type") == "JSON":
            node_parser = DoclingNodeParser()
            self.logger.info("DoclingNodeParser erstellt für JSON-Format")
            return node_parser
        else:
            # Bei Markdown-Export: Standard MarkdownNodeParser
            from llama_index.core.node_parser import MarkdownNodeParser
            node_parser = MarkdownNodeParser()
            self.logger.info("MarkdownNodeParser erstellt für Markdown-Format")
            return node_parser

    def parse_pdf(
        self, 
        pdf_path: Union[str, Path], 
        strict_mode: bool = True
    ) -> List[Document]:
        """
        Hauptmethode: konvertiert ein PDF in LlamaIndex-Documents.
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            strict_mode: Strenge Fehlerbehandlung
            
        Returns:
            Liste von LlamaIndex Document-Objekten
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF-Datei nicht gefunden: {pdf_path}")
        
        # File-Locking für Concurrency-Sicherheit
        with file_lock(pdf_path, timeout=60) as locked:
            if not locked:
                raise RuntimeError(f"Konnte File-Lock für {pdf_path.name} nicht akquirieren")
            
            self.logger.info(f"Verarbeite PDF: {pdf_path.name} ({pdf_path.stat().st_size / 1024 / 1024:.1f} MB)")
            
            start_time = time.time()
            
            try:
                if LLAMAINDEX_DOCLING_AVAILABLE:
                    # Nutze LlamaIndex native Integration
                    documents = self._parse_with_llamaindex(pdf_path)
                else:
                    # Fallback: Direkte Docling-Nutzung
                    documents = self._parse_with_docling_fallback(pdf_path)
                
                # Statistiken sammeln
                processing_time = time.time() - start_time
                self._last_stats = {
                    "processing_time_seconds": round(processing_time, 2),
                    "file_size_mb": round(pdf_path.stat().st_size / 1024 / 1024, 2),
                    "documents_created": len(documents),
                    "llamaindex_integration": LLAMAINDEX_DOCLING_AVAILABLE,
                    "export_type": self.config.get("export_type"),
                }
                
                self.logger.info(f"Erstellt {len(documents)} Documents in {processing_time:.1f}s")
                return documents
                
            except Exception as e:
                self.logger.error(f"Fehler beim Verarbeiten von {pdf_path.name}: {str(e)}")
                if strict_mode:
                    raise
                else:
                    return self._create_fallback_documents(pdf_path, str(e))

    def _parse_with_llamaindex(self, pdf_path: Path) -> List[Document]:
        """
        Nutzt LlamaIndex's native DoclingReader + DoclingNodeParser.
        
        Dies ist der optimale Pfad mit minimaler eigener Logik.
        """
        # 1. DoclingReader lädt PDF und erstellt Documents
        documents = self.reader.load_data(file_path=str(pdf_path))
        
        self.logger.info(f"DoclingReader: {len(documents)} Documents geladen")
        
        # 2. Erweitere Metadaten mit eigenen Analysen
        for doc in documents:
            # Zusätzliche Metadaten hinzufügen
            doc.metadata.update({
                "adapter_version": "2.0.0",
                "processing_timestamp": time.time(),
                "source_file": pdf_path.name,
            })
            
            # Tabellen-HTML extrahieren (falls JSON-Export)
            if self.config.get("export_type") == "JSON" and self.config.get("preserve_table_html"):
                self._extract_table_html_metadata(doc)
            
            # Bounding-Box-Informationen bewahren
            if self.config.get("preserve_bounding_boxes"):
                self._preserve_bounding_box_metadata(doc)
            
            # WICHTIG: Komplexe Metadaten in extra_info verschieben für ChromaDB-Kompatibilität
            doc = self._separate_complex_metadata(doc)
        
        return documents

    def _separate_complex_metadata(self, document: Document) -> Document:
        """
        Trennt komplexe Metadaten von einfachen für ChromaDB-Kompatibilität.
        
        Komplexe Strukturen (Listen, Dicts) werden in extra_info verschoben,
        während metadata nur primitive Typen (str, int, float, None) behält.
        """
        # Definiere welche Keys komplex sind und in extra_info gehören
        complex_keys = [
            "doc_items",           # Docling-Strukturdaten
            "bounding_boxes",      # Layout-Koordinaten
            "table_html_content",  # HTML-Tabellen
        ]
        
        # Initialisiere extra_info falls nicht vorhanden
        if not hasattr(document, 'extra_info') or document.extra_info is None:
            document.extra_info = {}
        
        # Verschiebe komplexe Metadaten
        for key in complex_keys:
            if key in document.metadata:
                value = document.metadata[key]
                # Nur verschieben wenn es wirklich komplex ist
                if isinstance(value, (list, dict)):
                    document.extra_info[key] = value
                    del document.metadata[key]
                    self.logger.debug(f"Verschoben {key} zu extra_info ({type(value).__name__})")
        
        # Bereinige verbleibende Metadaten für ChromaDB
        cleaned_metadata = {}
        for key, value in document.metadata.items():
            if value is None:
                cleaned_metadata[key] = None
            elif isinstance(value, (str, int, float)):
                cleaned_metadata[key] = value
            elif isinstance(value, bool):
                cleaned_metadata[key] = str(value)  # bool -> str für ChromaDB
            else:
                # Fallback: zu String konvertieren
                cleaned_metadata[key] = str(value)
                self.logger.debug(f"Konvertiert {key} ({type(value)}) zu String")
        
        document.metadata = cleaned_metadata
        return document

    def _extract_table_html_metadata(self, document: Document):
        """
        Extrahiert Tabellen-HTML aus Docling-JSON für besseres Retrieval.
        
        TableFormer liefert strukturierte Tabellen - diese als HTML bewahren.
        """
        try:
            # Prüfe ob Document Tabellen-Informationen enthält
            if 'doc_items' in document.metadata:
                doc_items = document.metadata['doc_items']
                
                tables_found = []
                for item in doc_items:
                    if item.get('label') == 'table':
                        # Tabellen-HTML extrahieren (falls verfügbar)
                        if 'table_html' in item:
                            tables_found.append(item['table_html'])
                        elif 'table_data' in item:
                            # Fallback: Strukturierte Daten zu HTML konvertieren
                            html = self._convert_table_data_to_html(item['table_data'])
                            if html:
                                tables_found.append(html)
                
                if tables_found:
                    document.metadata['table_html_content'] = tables_found
                    document.metadata['has_tables'] = True
                    document.metadata['table_count'] = len(tables_found)
                    self.logger.debug(f"Extrahiert {len(tables_found)} Tabellen als HTML")
                    
        except Exception as e:
            self.logger.warning(f"Fehler bei Tabellen-HTML-Extraktion: {e}")

    def _preserve_bounding_box_metadata(self, document: Document):
        """
        Bewahrt Bounding-Box-Informationen für UI-Highlighting.
        
        Docling liefert präzise Koordinaten - diese für spätere Nutzung bewahren.
        """
        try:
            if 'doc_items' in document.metadata:
                doc_items = document.metadata['doc_items']
                
                bounding_boxes = []
                for item in doc_items:
                    if 'prov' in item:
                        for prov in item['prov']:
                            if 'bbox' in prov and 'page_no' in prov:
                                bbox_info = {
                                    'page': prov['page_no'],
                                    'bbox': prov['bbox'],
                                    'element_type': item.get('label', 'text')
                                }
                                bounding_boxes.append(bbox_info)
                
                if bounding_boxes:
                    document.metadata['bounding_boxes'] = bounding_boxes
                    document.metadata['has_layout_info'] = True
                    self.logger.debug(f"Bewahrt {len(bounding_boxes)} Bounding-Boxes")
                    
        except Exception as e:
            self.logger.warning(f"Fehler bei Bounding-Box-Bewahrung: {e}")

    def _convert_table_data_to_html(self, table_data: Any) -> Optional[str]:
        """Konvertiert strukturierte Tabellendaten zu HTML."""
        try:
            # Einfache HTML-Tabellen-Generierung
            if isinstance(table_data, list) and table_data:
                html = "<table>\n"
                
                # Header (erste Zeile)
                if table_data[0]:
                    html += "  <thead>\n    <tr>\n"
                    for cell in table_data[0]:
                        html += f"      <th>{cell}</th>\n"
                    html += "    </tr>\n  </thead>\n"
                
                # Body (restliche Zeilen)
                if len(table_data) > 1:
                    html += "  <tbody>\n"
                    for row in table_data[1:]:
                        html += "    <tr>\n"
                        for cell in row:
                            html += f"      <td>{cell}</td>\n"
                        html += "    </tr>\n"
                    html += "  </tbody>\n"
                
                html += "</table>"
                return html
                
        except Exception as e:
            self.logger.warning(f"Fehler bei HTML-Konvertierung: {e}")
        
        return None

    def _clean_metadata_for_chromadb(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        DEPRECATED: Ersetzt durch _separate_complex_metadata()
        
        Diese Methode wird nicht mehr verwendet, da wir jetzt
        extra_info für komplexe Daten nutzen.
        """
        # Diese Methode bleibt für Rückwärtskompatibilität, wird aber nicht mehr aufgerufen
        return metadata

    def _parse_with_docling_fallback(self, pdf_path: Path) -> List[Document]:
        """Fallback: Direkte Docling-Nutzung ohne LlamaIndex-Integration."""
        self.logger.warning("Verwende Docling-Fallback (LlamaIndex-Integration nicht verfügbar)")
        
        if not DOCLING_AVAILABLE:
            raise ValueError("Docling nicht verfügbar")
        
        # Direkte Docling-Konvertierung
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        docling_doc = result.document
        
        # Einfache Document-Erstellung
        if hasattr(docling_doc, 'export_to_markdown'):
            text = docling_doc.export_to_markdown()
        else:
            text = str(docling_doc)
        
        # Basis-Metadaten
        metadata = {
            "filename": pdf_path.name,
            "file_path": str(pdf_path),
            "fallback_mode": True,
            "export_type": "markdown_fallback",
        }
        
        return [Document(text=text, metadata=metadata)]

    def _create_fallback_documents(self, pdf_path: Path, error_msg: str) -> List[Document]:
        """Erstellt minimale Documents bei Fehlern."""
        fallback_metadata = {
            "filename": pdf_path.name,
            "file_path": str(pdf_path),
            "processing_error": error_msg,
            "fallback_mode": True,
        }
        
        return [Document(
            text=f"Fallback-Verarbeitung für {pdf_path.name}. Fehler: {error_msg}",
            metadata=fallback_metadata
        )]

    def get_last_processing_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken des letzten parse_pdf-Aufrufs zurück."""
        return self._last_stats.copy()

    def is_llamaindex_docling_available(self) -> bool:
        """Prüft, ob LlamaIndex Docling-Integration verfügbar ist."""
        return LLAMAINDEX_DOCLING_AVAILABLE

    def get_config(self) -> Dict[str, Any]:
        """Gibt die aktuelle Konfiguration zurück."""
        return self.config.copy()


# Convenience-Funktion für einfache Nutzung
def parse_pdf_simple(pdf_path: Union[str, Path], **kwargs) -> List[Document]:
    """
    Einfache PDF-zu-Documents Konvertierung.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        **kwargs: Zusätzliche Konfigurationsparameter
        
    Returns:
        Liste von LlamaIndex Document-Objekten
    """
    adapter = DoclingAdapter(config=kwargs)
    return adapter.parse_pdf(pdf_path)


# CLI-Interface für Tests
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vereinfachter Docling PDF-Parser v2.0")
    parser.add_argument("--input", required=True, help="PDF-Datei")
    parser.add_argument("--export-type", choices=["JSON", "MARKDOWN"], default="JSON",
                       help="Export-Format")
    parser.add_argument("--stats", action="store_true", help="Zeige Statistiken")
    
    args = parser.parse_args()
    
    try:
        config = {"export_type": args.export_type}
        adapter = DoclingAdapter(config=config)
        
        print(f"🔧 LlamaIndex-Docling verfügbar: {'✅' if adapter.is_llamaindex_docling_available() else '❌'}")
        print(f"📄 Export-Format: {args.export_type}")
        print()
        
        documents = adapter.parse_pdf(args.input)
        print(f"✅ Erfolgreich {len(documents)} Documents erstellt")
        
        if args.stats:
            stats = adapter.get_last_processing_stats()
            print("\n📊 Statistiken:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
                
    except Exception as e:
        print(f"❌ Fehler: {str(e)}")
        exit(1) 