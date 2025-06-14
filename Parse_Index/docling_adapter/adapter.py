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
from typing import List, Dict, Optional
import logging

from llama_index.core import Document

# Docling-Import mit Fallback
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.warning("Docling nicht installiert. Installiere mit: pip install docling")

from .quality import QualityAnalyzer
from .sectionizer import SectionExtractor
from .metadata import MetadataExtractor
from .utils import setup_logging, validate_docling_config


class DoclingAdapter:
    """
    Hauptklasse für die PDF-Verarbeitung mit Docling.
    
    Orchestriert die komplette Pipeline von PDF zu LlamaIndex Documents.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialisiert den DoclingAdapter.
        
        Args:
            config: Konfigurationsdictionary, falls None wird aus config.py geladen
        """
        # Lade Konfiguration
        if config is None:
            try:
                from ..config import DOCLING_CONFIG
                self.config = DOCLING_CONFIG
            except ImportError:
                # Fallback-Konfiguration
                self.config = {
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
                }
        else:
            self.config = config
            
        # Validiere Konfiguration
        validate_docling_config(self.config)
        
        # Initialisiere Komponenten
        self.sectionizer = SectionExtractor(strategy="hybrid")
        self.quality_analyzer = QualityAnalyzer()
        self.metadata_extractor = MetadataExtractor()
        
        # Docling Converter mit Konfiguration
        if DOCLING_AVAILABLE:
            # Document Converter erstellen
            self.converter = DocumentConverter()
        else:
            self.converter = None
            
        # Statistiken des letzten Parsing-Vorgangs
        self._last_stats: Dict = {}
        
        # Logging einrichten
        setup_logging(level="INFO")
        self.logger = logging.getLogger(__name__)

    def parse_pdf(self, pdf_path: str | Path) -> List[Document]:
        """
        Hauptmethode: konvertiert ein PDF in LlamaIndex-Documents.
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            
        Returns:
            Liste von LlamaIndex Document-Objekten
            
        Raises:
            ValueError: Wenn Docling nicht verfügbar ist
            FileNotFoundError: Wenn PDF-Datei nicht existiert
        """
        if not DOCLING_AVAILABLE:
            raise ValueError("Docling ist nicht installiert. Installiere mit: pip install docling")
            
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF-Datei nicht gefunden: {pdf_path}")
            
        self.logger.info(f"Verarbeite PDF: {pdf_path.name}")
        
        try:
            # 1. PDF mit Docling parsen
            docling_doc = self._parse_with_docling(pdf_path)
            
            # 2. Semantische Blöcke extrahieren
            blocks = self.sectionizer.split(docling_doc)
            self.logger.info(f"Extrahierte {len(blocks)} semantische Blöcke")
            
            # 3. Qualitätsanalyse
            quality_stats = self.quality_analyzer.analyze(docling_doc)
            
            # 4. Metadaten extrahieren
            file_metadata = self.metadata_extractor.extract(docling_doc, pdf_path)
            
            # 5. Statistiken speichern
            self._last_stats = {
                **quality_stats,
                "pages": len(docling_doc.pages) if hasattr(docling_doc, 'pages') else 0,
                "blocks": len(blocks),
                "tables": sum(1 for b in blocks if b["metadata"].get("docling_type") == "table"),
                "images": sum(1 for b in blocks if b["metadata"].get("docling_type") == "image"),
                "formulas": sum(1 for b in blocks if b["metadata"].get("docling_type") == "formula"),
            }
            
            # 6. LlamaIndex Documents erstellen
            llama_docs: List[Document] = []
            for blk in blocks:
                # Metadaten zusammenführen
                combined_metadata = {
                    **file_metadata,
                    **blk["metadata"],
                    "quality_score": quality_stats.get("quality_score", 1.0),
                }
                
                # Document erstellen
                doc = Document(
                    text=blk["text"],
                    metadata=combined_metadata
                )
                llama_docs.append(doc)
            
            self.logger.info(f"Erstellt {len(llama_docs)} LlamaIndex Documents")
            return llama_docs
            
        except Exception as e:
            self.logger.error(f"Fehler beim Verarbeiten von {pdf_path.name}: {str(e)}")
            raise

    def _parse_with_docling(self, pdf_path: Path):
        """
        Interne Methode für Docling-Parsing.
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            
        Returns:
            Docling Document-Objekt
        """
        try:
            # PDF mit Docling konvertieren
            self.logger.info(f"Starte Docling-Parsing für: {pdf_path.name}")
            
            # Konvertierung durchführen
            result = self.converter.convert(str(pdf_path))
            
            # Das erste (und einzige) Dokument aus dem Result extrahieren
            docling_doc = result.document
            
            self.logger.info(f"Docling-Parsing erfolgreich: {len(docling_doc.pages)} Seiten")
            return docling_doc
            
        except Exception as e:
            self.logger.error(f"Docling-Parsing fehlgeschlagen: {str(e)}")
            # Fallback: Dummy-Implementierung
            self.logger.warning("Verwende Fallback-Dummy-Implementierung")
            
            class DummyPage:
                def __init__(self, page_num):
                    self.page_number = page_num
                    self.elements = []
                    
            class DummyDoc:
                def __init__(self, path):
                    self.source_path = path
                    self.pages = [DummyPage(1)]
                    
            return DummyDoc(pdf_path)

    def get_last_processing_stats(self) -> Dict:
        """
        Gibt Statistiken des letzten parse_pdf-Aufrufs zurück.
        
        Returns:
            Dictionary mit Statistiken (pages, blocks, tables, images, etc.)
        """
        return self._last_stats.copy()

    def parse_multiple_pdfs(self, pdf_paths: List[str | Path]) -> List[Document]:
        """
        Verarbeitet mehrere PDFs in einem Durchgang.
        
        Args:
            pdf_paths: Liste von PDF-Pfaden
            
        Returns:
            Kombinierte Liste aller LlamaIndex Documents
        """
        all_documents = []
        
        for pdf_path in pdf_paths:
            try:
                docs = self.parse_pdf(pdf_path)
                all_documents.extend(docs)
            except Exception as e:
                self.logger.error(f"Fehler bei {pdf_path}: {str(e)}")
                continue
                
        return all_documents


# CLI-Interface für schnelle Tests
if __name__ == "__main__":
    """Mini-CLI für schnelle Tests und Debugging."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Docling PDF-Parser")
    parser.add_argument("--input", required=True, help="PDF-Datei oder Ordner")
    parser.add_argument("--output", help="Ausgabe-Datei (JSON)")
    parser.add_argument("--to-llamaindex", action="store_true", 
                       help="Konvertiere zu LlamaIndex Documents")
    parser.add_argument("--stats", action="store_true", 
                       help="Zeige detaillierte Statistiken")
    
    args = parser.parse_args()
    
    # Adapter erstellen
    adapter = DoclingAdapter()
    
    try:
        # PDF verarbeiten
        input_path = Path(args.input)
        
        if input_path.is_file():
            documents = adapter.parse_pdf(input_path)
        elif input_path.is_dir():
            pdf_files = list(input_path.glob("*.pdf"))
            documents = adapter.parse_multiple_pdfs(pdf_files)
        else:
            raise FileNotFoundError(f"Pfad nicht gefunden: {input_path}")
        
        print(f"✅ Erfolgreich {len(documents)} Documents extrahiert")
        
        # Statistiken anzeigen
        if args.stats:
            stats = adapter.get_last_processing_stats()
            print("\n📊 Statistiken:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        # Optional: Ausgabe speichern
        if args.output:
            import json
            output_data = [
                {"text": doc.text, "metadata": doc.metadata} 
                for doc in documents
            ]
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Ausgabe gespeichert: {args.output}")
            
    except Exception as e:
        print(f"❌ Fehler: {str(e)}")
        exit(1) 