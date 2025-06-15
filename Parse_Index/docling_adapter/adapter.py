"""
DoclingAdapter
--------------
Zentrale Schnittstelle: PDF -> Docling -> LlamaIndex Document-Liste

Diese Klasse orchestriert die komplette Pipeline:
1. PDF mit Docling parsen
2. Qualitätsanalyse durchführen
3. Semantische Abschnitte extrahieren
4. Metadaten anreichern
5. LlamaIndex Documents erstellen
"""

from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging
import time
import signal
from contextlib import contextmanager

from llama_index.core import Document

# Docling-Import mit mehreren Fallback-Pfaden
DOCLING_AVAILABLE = False
DOCLING_VERSION = None
DOCLING_FEATURES = {
    "pipeline_options": False,
    "format_options": False,
    "chunker": False,
    "metadata": False
}

try:
    # Neuere Docling-Versionen (v2+)
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
    DOCLING_VERSION = "v2+"
    
    # Teste verfügbare Features
    try:
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption
        DOCLING_FEATURES["pipeline_options"] = True
        DOCLING_FEATURES["format_options"] = True
    except ImportError:
        pass
    
    try:
        from docling.chunking import HierarchicalChunker
        DOCLING_FEATURES["chunker"] = True
    except ImportError:
        pass
        
except ImportError:
    try:
        # Ältere Docling-Versionen
        from docling.convert import DocumentConverter
        DOCLING_AVAILABLE = True
        DOCLING_VERSION = "v1"
    except ImportError:
        try:
            # Alternative Import-Pfade
            from docling import DocumentConverter
            DOCLING_AVAILABLE = True
            DOCLING_VERSION = "legacy"
        except ImportError:
            DocumentConverter = None

from .quality import QualityAnalyzer
from .sectionizer import DoclingChunker
from .metadata import MetadataExtractor
from .utils import validate_docling_config

# Globaler Logger - wird nur einmal konfiguriert
_logger_configured = False

def _setup_logger() -> logging.Logger:
    """
    Konfiguriert Logger nur einmal global.
    Verhindert Handler-Duplikate bei mehreren Instanzen.
    """
    global _logger_configured
    logger = logging.getLogger("docling_adapter")
    
    if not _logger_configured:
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        _logger_configured = True
    
    return logger


@contextmanager
def timeout_context(seconds: int):
    """
    Context Manager für Timeout-Behandlung.
    
    Args:
        seconds: Timeout in Sekunden
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation dauerte länger als {seconds} Sekunden")
    
    # Nur auf Unix-Systemen verfügbar
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows-Fallback ohne Timeout
        yield


class DoclingAdapter:
    """
    Hauptklasse für die PDF-Verarbeitung mit Docling.
    
    Orchestriert die komplette Pipeline von PDF zu LlamaIndex Documents.
    """
    
    # Standard-Konfiguration
    DEFAULT_CONFIG = {
        "ocr_enabled": True,
        "table_extraction": True,
        "image_extraction": True,
        "formula_extraction": True,
        "layout_analysis": True,
        "reading_order": True,
        "export_format": "markdown",
        "chunk_by_page": False,
        "preserve_formatting": True,
        "extract_metadata": True,
        "timeout_seconds": 300,  # 5 Minuten Timeout
        "chunk_strategy": "hybrid",
        "max_chunk_size": 2000,
        "chunk_overlap": 200,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert den DoclingAdapter.
        
        Args:
            config: Konfigurationsdictionary, falls None wird Standard + config.py geladen
        """
        self.logger = _setup_logger()
        
        # Konfiguration zusammenführen
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Versuche config.py zu laden
        try:
            from ..config import DOCLING_CONFIG
            self.config.update(DOCLING_CONFIG)
            self.logger.debug("Konfiguration aus config.py geladen")
        except ImportError:
            self.logger.debug("Keine config.py gefunden, verwende Standard-Konfiguration")
        
        # Benutzer-Konfiguration überschreibt alles
        if config:
            self.config.update(config)
            
        # Validiere finale Konfiguration
        try:
            validate_docling_config(self.config)
        except ValueError as e:
            self.logger.error(f"Ungültige Konfiguration: {e}")
            raise
        
        # Initialisiere Komponenten
        self.chunker = DoclingChunker(
            strategy=self.config.get("chunk_strategy", "hybrid")
        )
        self.quality_analyzer = QualityAnalyzer()
        self.metadata_extractor = MetadataExtractor()
        
        # Docling Converter mit Konfiguration
        if DOCLING_AVAILABLE:
            try:
                self.converter = self._create_docling_converter()
                features_info = ", ".join([k for k, v in DOCLING_FEATURES.items() if v])
                self.logger.info(f"Docling verfügbar (Version: {DOCLING_VERSION}) - Features: {features_info or 'basic'}")
            except Exception as e:
                self.logger.error(f"Fehler beim Initialisieren von Docling: {e}")
                self.converter = None
                # Setze lokale Variable statt globale zu überschreiben
                self._docling_available = False
        else:
            self.converter = None
            self.logger.warning(
                "Docling nicht verfügbar. Installiere mit: pip install docling"
            )
            
        # Statistiken des letzten Parsing-Vorgangs
        self._last_stats: Dict[str, Any] = {}

    def _create_docling_converter(self) -> Any:
        """
        Erstellt einen konfigurierten DocumentConverter basierend auf der Config.
        
        Reicht Parameter wie OCR, Tabellen-Extraktion etc. an Docling weiter,
        anstatt sie nur intern zu verwenden.
        
        Returns:
            Konfigurierter DocumentConverter
        """
        # Prüfe verfügbare Features und wähle beste Konfigurationsmethode
        if DOCLING_FEATURES["pipeline_options"] and DOCLING_FEATURES["format_options"]:
            return self._create_advanced_docling_converter()
        else:
            return self._create_basic_docling_converter()
    
    def _create_advanced_docling_converter(self) -> Any:
        """
        Erstellt DocumentConverter mit erweiterten Pipeline-Optionen.
        
        Nutzt die neueste Docling-API für maximale Kontrolle.
        """
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
            
            # Pipeline-Optionen basierend auf unserer Config erstellen
            pipeline_options = PdfPipelineOptions()
            configured_options = []
            
            # OCR-Konfiguration
            if hasattr(pipeline_options, 'do_ocr'):
                pipeline_options.do_ocr = self.config.get("ocr_enabled", True)
                configured_options.append(f"OCR: {pipeline_options.do_ocr}")
            
            # Tabellen-Extraktion (TableFormer)
            if hasattr(pipeline_options, 'do_table_structure'):
                pipeline_options.do_table_structure = self.config.get("table_extraction", True)
                configured_options.append(f"Tables: {pipeline_options.do_table_structure}")
            
            # Layout-Analyse
            if hasattr(pipeline_options, 'do_layout'):
                pipeline_options.do_layout = self.config.get("layout_analysis", True)
                configured_options.append(f"Layout: {pipeline_options.do_layout}")
            
            # Bilder-Extraktion
            if hasattr(pipeline_options, 'images_scale'):
                if self.config.get("image_extraction", True):
                    pipeline_options.images_scale = 2.0  # Höhere Auflösung
                    configured_options.append("Images: enabled (2x scale)")
                else:
                    pipeline_options.images_scale = 0.0  # Deaktiviert
                    configured_options.append("Images: disabled")
            
            # Formeln-Extraktion (falls verfügbar)
            if hasattr(pipeline_options, 'do_formula') and self.config.get("formula_extraction", True):
                pipeline_options.do_formula = True
                configured_options.append("Formulas: enabled")
            
            # Format-Optionen mit Pipeline konfigurieren
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
            
            converter = DocumentConverter(format_options=format_options)
            self.logger.info(f"Docling erweitert konfiguriert: {', '.join(configured_options)}")
            return converter
            
        except Exception as e:
            self.logger.warning(f"Erweiterte Konfiguration fehlgeschlagen: {e}")
            return self._create_basic_docling_converter()
    
    def _create_basic_docling_converter(self) -> Any:
        """
        Fallback: Erstellt Standard-DocumentConverter.
        
        Für ältere Docling-Versionen oder wenn erweiterte Konfiguration fehlschlägt.
        """
        try:
            # Versuche wenigstens grundlegende Parameter zu setzen
            converter_kwargs = {}
            
            # Manche Docling-Versionen unterstützen diese Parameter direkt
            if self.config.get("ocr_enabled") is False:
                converter_kwargs['disable_ocr'] = True
            
            converter = DocumentConverter(**converter_kwargs)
            self.logger.info("Docling mit Standard-Konfiguration erstellt")
            return converter
            
        except Exception as e:
            self.logger.warning(f"Auch Standard-Konfiguration fehlgeschlagen: {e}")
            # Letzter Fallback: Komplett ohne Parameter
            return DocumentConverter()

    def parse_pdf(
        self, 
        pdf_path: Union[str, Path], 
        strict_mode: bool = True
    ) -> List[Document]:
        """
        Hauptmethode: konvertiert ein PDF in LlamaIndex-Documents.
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            strict_mode: Wenn False, wird bei Docling-Fehlern Fallback verwendet
            
        Returns:
            Liste von LlamaIndex Document-Objekten
            
        Raises:
            ValueError: Wenn Docling nicht verfügbar ist (nur bei strict_mode=True)
            FileNotFoundError: Wenn PDF-Datei nicht existiert
            TimeoutError: Wenn Verarbeitung zu lange dauert
        """
        if not DOCLING_AVAILABLE and strict_mode:
            raise ValueError(
                "Docling ist nicht installiert. Installiere mit: pip install docling "
                "oder verwende strict_mode=False für Fallback-Modus"
            )
            
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF-Datei nicht gefunden: {pdf_path}")
            
        # File-Locking für Concurrency-Sicherheit
        from .utils import file_lock
        
        with file_lock(
            pdf_path, 
            timeout=self.config.get("lock_timeout_seconds", 60),
            retry_attempts=self.config.get("lock_retry_attempts", 3),
            retry_delay=self.config.get("lock_retry_delay", 1.0)
        ) as locked:
            
            if not locked:
                raise RuntimeError(f"Konnte File-Lock für {pdf_path.name} nicht akquirieren. "
                                 f"Datei wird möglicherweise bereits verarbeitet.")
            
            self.logger.info(f"Verarbeite PDF: {pdf_path.name} ({pdf_path.stat().st_size / 1024 / 1024:.1f} MB)")
            
            start_time = time.time()
            
            try:
                # 1. PDF mit Docling parsen (mit Timeout)
                docling_doc = self._parse_with_docling(pdf_path, strict_mode)
            
                # 2. Semantische Blöcke extrahieren
                blocks = self.chunker.split(docling_doc)
                self.logger.info(f"Extrahierte {len(blocks)} semantische Blöcke")
                
                # 3. Qualitätsanalyse
                quality_stats = self.quality_analyzer.analyze(docling_doc)
                
                # 4. Metadaten extrahieren
                file_metadata = self.metadata_extractor.extract(docling_doc, pdf_path)
                
                # 5. Statistiken speichern
                processing_time = time.time() - start_time
                self._last_stats = {
                    **quality_stats,
                    "pages": len(docling_doc.pages) if hasattr(docling_doc, 'pages') else 0,
                    "blocks": len(blocks),
                    "tables": sum(1 for b in blocks if b["metadata"].get("docling_type") == "table"),
                    "images": sum(1 for b in blocks if b["metadata"].get("docling_type") == "image"),
                    "formulas": sum(1 for b in blocks if b["metadata"].get("docling_type") == "formula"),
                    "processing_time_seconds": round(processing_time, 2),
                    "file_size_mb": round(pdf_path.stat().st_size / 1024 / 1024, 2),
                    "docling_available": DOCLING_AVAILABLE,
                    "docling_version": DOCLING_VERSION,
                }
                
                # 6. LlamaIndex Documents erstellen
                llama_docs = self._create_documents(blocks, file_metadata, quality_stats)
                
                self.logger.info(
                    f"Erstellt {len(llama_docs)} LlamaIndex Documents "
                    f"in {processing_time:.1f}s"
                )
                return llama_docs
                
            except TimeoutError:
                self.logger.error(f"Timeout bei Verarbeitung von {pdf_path.name}")
                raise
            except Exception as e:
                self.logger.error(f"Fehler beim Verarbeiten von {pdf_path.name}: {str(e)}")
                if strict_mode:
                    raise
                else:
                    # Fallback: Minimale Verarbeitung
                    return self._create_fallback_documents(pdf_path, str(e))

    def _parse_with_docling(self, pdf_path: Path, strict_mode: bool = True):
        """
        Interne Methode für Docling-Parsing mit Timeout.
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            strict_mode: Strenge Fehlerbehandlung
            
        Returns:
            Docling Document-Objekt
        """
        if not DOCLING_AVAILABLE:
            if strict_mode:
                raise ValueError("Docling nicht verfügbar")
            else:
                return self._create_dummy_doc(pdf_path)
        
        try:
            self.logger.info(f"Starte Docling-Parsing für: {pdf_path.name}")
            
            timeout_seconds = self.config.get("timeout_seconds", 300)
            
            # Konvertierung mit Timeout durchführen
            with timeout_context(timeout_seconds):
                result = self.converter.convert(str(pdf_path))
                
            # Das erste (und einzige) Dokument aus dem Result extrahieren
            docling_doc = result.document
            
            page_count = len(docling_doc.pages) if hasattr(docling_doc, 'pages') else 0
            self.logger.info(f"Docling-Parsing erfolgreich: {page_count} Seiten")
            return docling_doc
            
        except TimeoutError:
            self.logger.error(f"Docling-Parsing Timeout nach {timeout_seconds}s")
            raise
        except Exception as e:
            self.logger.error(f"Docling-Parsing fehlgeschlagen: {str(e)}")
            if strict_mode:
                raise
            else:
                self.logger.warning("Verwende Fallback-Dummy-Implementierung")
                return self._create_dummy_doc(pdf_path)

    def _create_dummy_doc(self, pdf_path: Path):
        """Erstellt ein Dummy-Docling-Document für Fallback-Fälle."""
        class DummyPage:
            def __init__(self, page_num: int):
                self.page_number = page_num
                self.elements = []
                
        class DummyDoc:
            def __init__(self, path: Path):
                self.source_path = str(path)
                self.pages = [DummyPage(1)]
                
        return DummyDoc(pdf_path)

    def _create_documents(
        self, 
        blocks: List[Dict[str, Any]], 
        file_metadata: Dict[str, Any], 
        quality_stats: Dict[str, Any]
    ) -> List[Document]:
        """
        Erstellt LlamaIndex Documents aus Blöcken und Metadaten.
        
        Metadaten-Merge-Priorität:
        1. Block-spezifische Metadaten (höchste Priorität)
        2. Qualitäts-Statistiken
        3. Datei-Metadaten (niedrigste Priorität)
        """
        llama_docs: List[Document] = []
        
        for i, blk in enumerate(blocks):
            # Metadaten in definierter Reihenfolge zusammenführen
            combined_metadata = {}
            
            # 1. Basis: Datei-Metadaten
            combined_metadata.update(file_metadata)
            
            # 2. Qualitäts-Informationen hinzufügen
            combined_metadata.update({
                "quality_score": quality_stats.get("quality_score", 0.5),
                "ocr_ratio": quality_stats.get("ocr_ratio", 0.0),
                "has_structure": quality_stats.get("has_structure", False),
            })
            
            # 3. Block-spezifische Metadaten (überschreiben bei Konflikten)
            combined_metadata.update(blk["metadata"])
            
            # 4. Zusätzliche Verarbeitungs-Metadaten
            combined_metadata.update({
                "block_index": i,
                "total_blocks": len(blocks),
                "text_length": len(blk["text"]),
                "adapter_version": "1.0.0",
            })
            
            # Document erstellen
            doc = Document(
                text=blk["text"],
                metadata=combined_metadata
            )
            llama_docs.append(doc)
        
        return llama_docs

    def _create_fallback_documents(self, pdf_path: Path, error_msg: str) -> List[Document]:
        """Erstellt minimale Documents bei Fallback-Verarbeitung."""
        fallback_metadata = {
            "filename": pdf_path.name,
            "file_path": str(pdf_path),
            "processing_error": error_msg,
            "fallback_mode": True,
            "quality_score": 0.1,  # Niedrige Qualität signalisieren
        }
        
        return [Document(
            text=f"Fallback-Verarbeitung für {pdf_path.name}. Fehler: {error_msg}",
            metadata=fallback_metadata
        )]

    def get_last_processing_stats(self) -> Dict[str, Any]:
        """
        Gibt Statistiken des letzten parse_pdf-Aufrufs zurück.
        
        Returns:
            Dictionary mit Statistiken (pages, blocks, tables, images, etc.)
        """
        return self._last_stats.copy()

    def parse_multiple_pdfs(
        self, 
        pdf_paths: List[Union[str, Path]], 
        strict_mode: bool = False,
        show_progress: bool = True
    ) -> List[Document]:
        """
        Verarbeitet mehrere PDFs in einem Durchgang.
        
        Args:
            pdf_paths: Liste von PDF-Pfaden
            strict_mode: Strenge Fehlerbehandlung
            show_progress: Fortschrittsanzeige aktivieren
            
        Returns:
            Kombinierte Liste aller LlamaIndex Documents
        """
        all_documents = []
        failed_files = []
        
        total_files = len(pdf_paths)
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            if show_progress:
                print(f"Verarbeite {i}/{total_files}: {Path(pdf_path).name}")
            
            try:
                docs = self.parse_pdf(pdf_path, strict_mode=strict_mode)
                all_documents.extend(docs)
                
                if show_progress:
                    print(f"  ✅ {len(docs)} Documents erstellt")
                    
            except Exception as e:
                failed_files.append((pdf_path, str(e)))
                self.logger.error(f"Fehler bei {pdf_path}: {str(e)}")
                
                if show_progress:
                    print(f"  ❌ Fehler: {str(e)}")
                
                if strict_mode:
                    raise
                continue
        
        # Zusammenfassung
        success_count = total_files - len(failed_files)
        self.logger.info(
            f"Batch-Verarbeitung abgeschlossen: "
            f"{success_count}/{total_files} erfolgreich, "
            f"{len(all_documents)} Documents erstellt"
        )
        
        if failed_files and show_progress:
            print(f"\n⚠️  {len(failed_files)} Dateien fehlgeschlagen:")
            for path, error in failed_files:
                print(f"  - {Path(path).name}: {error}")
                
        return all_documents

    def is_docling_available(self) -> bool:
        """Prüft, ob Docling verfügbar und funktionsfähig ist."""
        return DOCLING_AVAILABLE and self.converter is not None

    def get_config(self) -> Dict[str, Any]:
        """Gibt die aktuelle Konfiguration zurück."""
        return self.config.copy()


# CLI-Interface für schnelle Tests
if __name__ == "__main__":
    """Verbessertes CLI für Tests und Debugging."""
    import argparse
    import json
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(
        description="Docling PDF-Parser v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python adapter.py --input document.pdf --stats
  python adapter.py --input ./pdfs/ --output results.json --parallel
  python adapter.py --input document.pdf --no-strict --timeout 60
        """
    )
    
    parser.add_argument("--input", required=True, 
                       help="PDF-Datei oder Ordner")
    parser.add_argument("--output", 
                       help="Ausgabe-Datei (JSON)")
    parser.add_argument("--stats", action="store_true", 
                       help="Zeige detaillierte Statistiken")
    parser.add_argument("--no-strict", action="store_true",
                       help="Verwende Fallback bei Fehlern")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout in Sekunden (Standard: 300)")
    parser.add_argument("--chunk-strategy", choices=["hybrid", "by_page", "by_heading", "by_element"],
                       default="hybrid", help="Chunking-Strategie")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduzierte Ausgabe")
    
    args = parser.parse_args()
    
    # Konfiguration aus CLI-Argumenten
    config = {
        "timeout_seconds": args.timeout,
        "chunk_strategy": args.chunk_strategy,
    }
    
    # Adapter erstellen
    adapter = DoclingAdapter(config=config)
    
    if not args.quiet:
        print(f"🔧 Docling verfügbar: {'✅' if adapter.is_docling_available() else '❌'}")
        print(f"⚙️  Chunk-Strategie: {args.chunk_strategy}")
        print(f"⏱️  Timeout: {args.timeout}s")
        print()
    
    try:
        # PDF verarbeiten
        input_path = Path(args.input)
        
        if input_path.is_file():
            documents = adapter.parse_pdf(input_path, strict_mode=not args.no_strict)
        elif input_path.is_dir():
            pdf_files = list(input_path.glob("*.pdf"))
            if not pdf_files:
                print("❌ Keine PDF-Dateien im Ordner gefunden")
                exit(1)
            
            documents = adapter.parse_multiple_pdfs(
                pdf_files, 
                strict_mode=not args.no_strict,
                show_progress=not args.quiet
            )
        else:
            raise FileNotFoundError(f"Pfad nicht gefunden: {input_path}")
        
        if not args.quiet:
            print(f"\n✅ Erfolgreich {len(documents)} Documents extrahiert")
        
        # Statistiken anzeigen
        if args.stats:
            stats = adapter.get_last_processing_stats()
            print("\n📊 Verarbeitungsstatistiken:")
            print("=" * 40)
            for key, value in stats.items():
                if isinstance(value, float):
                    if 0 < value < 1:
                        print(f"   {key}: {value:.1%}")
                    else:
                        print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
        
        # Optional: Ausgabe speichern
        if args.output:
            output_data = [
                {"text": doc.text, "metadata": doc.metadata} 
                for doc in documents
            ]
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
            
            if not args.quiet:
                print(f"💾 Ausgabe gespeichert: {args.output}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Verarbeitung abgebrochen")
        exit(1)
    except Exception as e:
        print(f"❌ Fehler: {str(e)}")
        if args.stats:
            import traceback
            traceback.print_exc()
        exit(1) 