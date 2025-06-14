"""
Utilities für Docling Adapter
-----------------------------
Hilfsfunktionen für Logging, Konfiguration, Validierung und allgemeine Tasks.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import re


def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> logging.Logger:
    """
    Richtet Logging für den Docling Adapter ein.
    
    Args:
        level: Log-Level ("DEBUG", "INFO", "WARNING", "ERROR")
        format_string: Optionales Custom-Format
        
    Returns:
        Konfigurierter Logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Root-Logger konfigurieren
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Docling-spezifischen Logger erstellen
    logger = logging.getLogger("docling_adapter")
    logger.setLevel(getattr(logging, level.upper()))
    
    return logger


def validate_docling_config(config: Dict[str, Any]) -> bool:
    """
    Validiert eine Docling-Konfiguration.
    
    Args:
        config: Konfigurationsdictionary
        
    Returns:
        True wenn Konfiguration gültig
        
    Raises:
        ValueError: Bei ungültiger Konfiguration
    """
    required_fields = [
        "ocr_enabled",
        "table_extraction", 
        "image_extraction",
        "reading_order"
    ]
    
    # Prüfe erforderliche Felder
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Docling-Config fehlen Felder: {', '.join(missing_fields)}")
    
    # Prüfe Datentypen
    boolean_fields = [
        "ocr_enabled", "table_extraction", "image_extraction", 
        "formula_extraction", "layout_analysis", "reading_order",
        "chunk_by_page", "preserve_formatting", "extract_metadata"
    ]
    
    for field in boolean_fields:
        if field in config and not isinstance(config[field], bool):
            raise ValueError(f"Feld '{field}' muss boolean sein, ist {type(config[field])}")
    
    # Prüfe Export-Format
    if "export_format" in config:
        valid_formats = ["markdown", "html", "json", "text"]
        if config["export_format"] not in valid_formats:
            raise ValueError(f"Ungültiges export_format: {config['export_format']}. Gültig: {valid_formats}")
    
    return True


def slugify(text: str, max_length: int = 50) -> str:
    """
    Wandelt Text in einen URL/Dateiname-sicheren Slug um.
    
    Args:
        text: Zu konvertierender Text
        max_length: Maximale Länge des Slugs
        
    Returns:
        Bereinigter Slug
    """
    if not text:
        return "untitled"
    
    # Zu Kleinbuchstaben, Umlaute ersetzen
    text = text.lower()
    text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
    
    # Nur alphanumerische Zeichen und Bindestriche behalten
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    
    # Leerzeichen durch Bindestriche ersetzen
    text = re.sub(r'\s+', '-', text)
    
    # Mehrfache Bindestriche reduzieren
    text = re.sub(r'-+', '-', text)
    
    # Bindestriche am Anfang/Ende entfernen
    text = text.strip('-')
    
    # Auf maximale Länge kürzen
    if len(text) > max_length:
        text = text[:max_length].rstrip('-')
    
    return text or "untitled"


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Teilt Text in überlappende Chunks auf.
    
    Args:
        text: Zu teilender Text
        chunk_size: Größe der Chunks in Zeichen
        overlap: Überlappung zwischen Chunks in Zeichen
        
    Returns:
        Liste von Text-Chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Versuche an Satzende zu trennen
        if end < len(text):
            # Suche nach Satzende in den letzten 100 Zeichen
            search_start = max(start, end - 100)
            sentence_end = text.rfind('.', search_start, end)
            
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Nächster Start mit Überlappung
        start = end - overlap
        
        # Verhindere Endlosschleife
        if start >= end:
            break
    
    return chunks


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Macht einen Dateinamen sicher für das Dateisystem.
    
    Args:
        filename: Original-Dateiname
        max_length: Maximale Länge
        
    Returns:
        Sicherer Dateiname
    """
    # Gefährliche Zeichen entfernen
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Kontrollzeichen entfernen
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Auf maximale Länge kürzen
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = max_length - len(ext) - 1 if ext else max_length
        filename = name[:max_name_length] + ('.' + ext if ext else '')
    
    return filename.strip()


def format_file_size(size_bytes: int) -> str:
    """
    Formatiert Dateigröße in menschenlesbarer Form.
    
    Args:
        size_bytes: Größe in Bytes
        
    Returns:
        Formatierte Größe (z.B. "1.5 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def validate_pdf_file(pdf_path: str | Path) -> bool:
    """
    Validiert, ob eine Datei eine gültige PDF ist.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        
    Returns:
        True wenn gültige PDF
    """
    pdf_path = Path(pdf_path)
    
    # Existenz prüfen
    if not pdf_path.exists():
        return False
    
    # Dateierweiterung prüfen
    if pdf_path.suffix.lower() != '.pdf':
        return False
    
    # PDF-Header prüfen
    try:
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False


def get_pdf_info(pdf_path: str | Path) -> Dict[str, Any]:
    """
    Sammelt grundlegende Informationen über eine PDF-Datei.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        
    Returns:
        Dictionary mit PDF-Informationen
    """
    pdf_path = Path(pdf_path)
    
    info = {
        "path": str(pdf_path),
        "filename": pdf_path.name,
        "exists": pdf_path.exists(),
        "is_valid_pdf": False,
        "size_bytes": 0,
        "size_formatted": "0 B",
    }
    
    if pdf_path.exists():
        stat = pdf_path.stat()
        info["size_bytes"] = stat.st_size
        info["size_formatted"] = format_file_size(stat.st_size)
        info["modified"] = stat.st_mtime
        info["is_valid_pdf"] = validate_pdf_file(pdf_path)
    
    return info


def create_output_filename(input_path: str | Path, suffix: str = "", extension: str = ".json") -> str:
    """
    Erstellt einen Ausgabe-Dateinamen basierend auf dem Eingabe-Pfad.
    
    Args:
        input_path: Eingabe-Dateipfad
        suffix: Optionaler Suffix (z.B. "_processed")
        extension: Dateierweiterung für Ausgabe
        
    Returns:
        Ausgabe-Dateiname
    """
    input_path = Path(input_path)
    base_name = input_path.stem  # Dateiname ohne Erweiterung
    
    output_name = f"{base_name}{suffix}{extension}"
    return safe_filename(output_name)


def save_json(data: Any, output_path: str | Path, indent: int = 2, ensure_ascii: bool = False) -> bool:
    """
    Speichert Daten als JSON-Datei.
    
    Args:
        data: Zu speichernde Daten
        output_path: Ausgabe-Pfad
        indent: JSON-Einrückung
        ensure_ascii: ASCII-Encoding erzwingen
        
    Returns:
        True wenn erfolgreich gespeichert
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        
        return True
    except Exception as e:
        logging.error(f"Fehler beim Speichern von JSON: {str(e)}")
        return False


def load_json(input_path: str | Path) -> Optional[Any]:
    """
    Lädt Daten aus JSON-Datei.
    
    Args:
        input_path: Pfad zur JSON-Datei
        
    Returns:
        Geladene Daten oder None bei Fehler
    """
    try:
        input_path = Path(input_path)
        if not input_path.exists():
            return None
        
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Fehler beim Laden von JSON: {str(e)}")
        return None


def print_processing_stats(stats: Dict[str, Any], title: str = "Verarbeitungsstatistiken") -> None:
    """
    Gibt Verarbeitungsstatistiken formatiert aus.
    
    Args:
        stats: Statistiken-Dictionary
        title: Titel für die Ausgabe
    """
    print(f"\n📊 {title}")
    print("=" * (len(title) + 4))
    
    for key, value in stats.items():
        if isinstance(value, float):
            if 0 < value < 1:
                print(f"   {key}: {value:.1%}")
            else:
                print(f"   {key}: {value:.3f}")
        elif isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"      {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")


def estimate_processing_time(file_size_mb: float, pages: int = None) -> str:
    """
    Schätzt die Verarbeitungszeit für eine PDF-Datei.
    
    Args:
        file_size_mb: Dateigröße in MB
        pages: Anzahl Seiten (optional)
        
    Returns:
        Geschätzte Zeit als String
    """
    # Grobe Schätzung: 1MB = ~30 Sekunden, 1 Seite = ~5 Sekunden
    time_by_size = file_size_mb * 30
    time_by_pages = (pages or file_size_mb * 2) * 5  # Schätze 2 Seiten pro MB
    
    estimated_seconds = max(time_by_size, time_by_pages)
    
    if estimated_seconds < 60:
        return f"~{int(estimated_seconds)} Sekunden"
    elif estimated_seconds < 3600:
        return f"~{int(estimated_seconds / 60)} Minuten"
    else:
        return f"~{int(estimated_seconds / 3600)} Stunden"


# Konstanten für häufig verwendete Werte
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
MAX_FILENAME_LENGTH = 255
SUPPORTED_EXTENSIONS = ['.pdf']

# Logging-Setup beim Import
logger = setup_logging() 