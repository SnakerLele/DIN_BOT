"""
img2table Integration für verbesserte Tabellen-Extraktion

Dieses Modul nutzt img2table für robuste Tabellen-Erkennung:
- Konvertiert PDF-Seiten zu Bildern (höhere Erkennungsrate)
- Extrahiert Tabellen mit OpenCV-basierter Bildverarbeitung  
- Liefert direkt Markdown-formatierte Tabellen für RAG
- Unterstützt gemergte Zellen und borderless tables
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import tempfile
import logging
from PIL import Image as PILImage

try:
    from pdf2image import convert_from_path
    from img2table.document import Image as Img2TableImage
    from img2table.ocr import TesseractOCR
    PDF2IMAGE_AVAILABLE = True
except ImportError as e:
    PDF2IMAGE_AVAILABLE = False
    logging.warning(f"img2table dependencies nicht verfügbar: {e}")

logger = logging.getLogger(__name__)

class Img2TableExtractor:
    """
    img2table basierte Tabellen-Extraktion für PDFs
    """
    
    def __init__(self, 
                 lang: str = "deu+eng",  # Deutsch + Englisch für beste Ergebnisse
                 dpi: int = 300,
                 n_threads: int = 2,
                 borderless_tables: bool = True,
                 implicit_rows: bool = False,
                 implicit_columns: bool = False):
        """
        Initialisiert den img2table Extraktor
        
        Args:
            lang: OCR Sprachen (Tesseract Format)
            dpi: Auflösung für PDF-zu-Bild Konvertierung
            n_threads: Anzahl OCR Threads
            borderless_tables: Erkenne Tabellen ohne Rahmen
            implicit_rows: Erkenne implizite Tabellenzeilen
            implicit_columns: Erkenne implizite Tabellenspalten
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError(
                "img2table dependencies fehlen. Installiere mit: "
                "pip install img2table pdf2image pillow"
            )
        
        self.lang = lang
        self.dpi = dpi
        self.n_threads = n_threads
        self.borderless_tables = borderless_tables
        self.implicit_rows = implicit_rows
        self.implicit_columns = implicit_columns
        
        # OCR Engine initialisieren
        try:
            self.ocr = TesseractOCR(
                lang=self.lang, 
                n_threads=self.n_threads
            )
            logger.info(f"✅ img2table OCR initialisiert (Sprache: {self.lang})")
        except Exception as e:
            logger.error(f"❌ OCR Initialisierung fehlgeschlagen: {e}")
            raise
    
    def extract_tables_from_pdf(self, pdf_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extrahiert alle Tabellen aus einem PDF als Markdown
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            
        Returns:
            Tuple aus (Markdown-Tabellen-Liste, Metadaten-Liste)
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF nicht gefunden: {pdf_path}")
        
        logger.info(f"🔍 Starte img2table Extraktion für: {pdf_path.name}")
        
        try:
            # PDF zu Bildern konvertieren
            pages = convert_from_path(
                str(pdf_path), 
                dpi=self.dpi,
                fmt='PNG'  # PNG für bessere Qualität
            )
            logger.info(f"📄 {len(pages)} Seiten konvertiert (DPI: {self.dpi})")
            
            markdown_tables = []
            table_metadata = []
            total_tables = 0
            
            # Jede Seite nach Tabellen durchsuchen
            for page_num, pil_image in enumerate(pages, 1):
                logger.info(f"🔍 Verarbeite Seite {page_num}/{len(pages)}")
                
                # Temporäre Datei für das Bild erstellen
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    pil_image.save(temp_file.name, 'PNG')
                    temp_path = temp_file.name
                
                try:
                    # img2table Dokument erstellen (mit Dateipfad)
                    img_doc = Img2TableImage(temp_path)
                    
                    # Tabellen extrahieren
                    extracted_tables = img_doc.extract_tables(
                        ocr=self.ocr,
                        implicit_rows=self.implicit_rows,
                        implicit_columns=self.implicit_columns,
                        borderless_tables=self.borderless_tables,
                        min_confidence=50  # Mindestqualität für OCR
                    )
                    
                    page_tables = len(extracted_tables)
                    total_tables += page_tables
                    
                    if page_tables > 0:
                        logger.info(f"  📊 {page_tables} Tabelle(n) auf Seite {page_num} gefunden")
                    
                    # Jede Tabelle zu Markdown konvertieren
                    for table_idx, table in enumerate(extracted_tables):
                        try:
                            # DataFrame zu Markdown
                            markdown = table.df.to_markdown(index=False)
                            
                            # Metadaten sammeln (numpy-Typen zu Python-Typen konvertieren)
                            metadata = {
                                'extraction_method': 'img2table',
                                'page_number': int(page_num),
                                'table_index': int(table_idx),
                                'table_id': f"page_{page_num}_table_{table_idx}",
                                'rows': int(table.df.shape[0]),
                                'columns': int(table.df.shape[1]),
                                'bbox': {
                                    'x1': float(table.bbox.x1),
                                    'y1': float(table.bbox.y1), 
                                    'x2': float(table.bbox.x2),
                                    'y2': float(table.bbox.y2)
                                },
                                'title': table.title if hasattr(table, 'title') else None,
                                'content_type': 'table',
                                'is_table': True,
                                'table_quality': self._assess_table_quality(table.df, markdown),
                                'character_count': int(len(markdown))
                            }
                            
                            markdown_tables.append(markdown)
                            table_metadata.append(metadata)
                            
                            logger.info(f"  ✅ Tabelle {table_idx+1}: {metadata['rows']}x{metadata['columns']} -> {len(markdown)} Zeichen")
                            
                        except Exception as e:
                            logger.error(f"  ❌ Konvertierungsfehler Tabelle {table_idx}: {e}")
                            continue
                finally:
                    # Temporäre Datei löschen
                    try:
                        Path(temp_path).unlink()
                    except:
                        pass
            
            logger.info(f"🎯 img2table Extraktion abgeschlossen: {total_tables} Tabellen gefunden")
            return markdown_tables, table_metadata
            
        except Exception as e:
            logger.error(f"❌ img2table Extraktion fehlgeschlagen: {e}")
            return [], []
    
    def extract_tables_from_image(self, image_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extrahiert Tabellen direkt aus einem Bild
        
        Args:
            image_path: Pfad zur Bilddatei
            
        Returns:
            Tuple aus (Markdown-Tabellen-Liste, Metadaten-Liste)
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
        
        logger.info(f"🖼️ Starte img2table Extraktion für Bild: {image_path.name}")
        
        try:
            # Bild laden
            pil_image = PILImage.open(image_path)
            img_doc = Img2TableImage(pil_image)
            
            # Tabellen extrahieren
            extracted_tables = img_doc.extract_tables(
                ocr=self.ocr,
                implicit_rows=self.implicit_rows,
                implicit_columns=self.implicit_columns,
                borderless_tables=self.borderless_tables,
                min_confidence=50
            )
            
            markdown_tables = []
            table_metadata = []
            
            logger.info(f"📊 {len(extracted_tables)} Tabelle(n) im Bild gefunden")
            
            for table_idx, table in enumerate(extracted_tables):
                try:
                    markdown = table.df.to_markdown(index=False)
                    
                    metadata = {
                        'extraction_method': 'img2table',
                        'source_image': str(image_path.name),
                        'table_index': int(table_idx),
                        'table_id': f"img_{image_path.stem}_table_{table_idx}",
                        'rows': int(table.df.shape[0]),
                        'columns': int(table.df.shape[1]),
                        'bbox': {
                            'x1': float(table.bbox.x1),
                            'y1': float(table.bbox.y1),
                            'x2': float(table.bbox.x2),
                            'y2': float(table.bbox.y2)
                        },
                        'title': table.title if hasattr(table, 'title') else None,
                        'content_type': 'table',
                        'is_table': True,
                        'table_quality': self._assess_table_quality(table.df, markdown),
                        'character_count': int(len(markdown))
                    }
                    
                    markdown_tables.append(markdown)
                    table_metadata.append(metadata)
                    
                    logger.info(f"✅ Tabelle {table_idx+1}: {metadata['rows']}x{metadata['columns']}")
                    
                except Exception as e:
                    logger.error(f"❌ Konvertierungsfehler Tabelle {table_idx}: {e}")
                    continue
            
            return markdown_tables, table_metadata
            
        except Exception as e:
            logger.error(f"❌ Bild-Extraktion fehlgeschlagen: {e}")
            return [], []
    
    def _assess_table_quality(self, df, markdown: str) -> str:
        """
        Bewertet die Qualität einer extrahierten Tabelle
        
        Args:
            df: pandas DataFrame
            markdown: Markdown-String der Tabelle
            
        Returns:
            Qualitätsbewertung als String
        """
        # Einfache Heuristiken für Tabellenqualität
        if df.empty:
            return 'poor'
        
        # Mindestgröße prüfen
        if df.shape[0] < 2 or df.shape[1] < 2:
            return 'poor'
        
        # Leere Zellen prüfen
        empty_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        
        # Text-Länge prüfen
        char_count = len(markdown)
        
        if empty_ratio > 0.5:  # Mehr als 50% leere Zellen
            return 'poor'
        elif empty_ratio > 0.2 or char_count < 100:  # 20-50% leer oder sehr kurz
            return 'medium'
        else:
            return 'good'


def create_img2table_extractor(**kwargs) -> Optional[Img2TableExtractor]:
    """
    Factory-Funktion für img2table Extraktor
    
    Args:
        **kwargs: Parameter für Img2TableExtractor
        
    Returns:
        Img2TableExtractor Instanz oder None bei Fehlern
    """
    try:
        return Img2TableExtractor(**kwargs)
    except Exception as e:
        logger.error(f"❌ img2table Extraktor konnte nicht erstellt werden: {e}")
        return None