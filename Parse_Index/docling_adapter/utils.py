"""
Utilities für Docling Adapter
-----------------------------
Robuste Hilfsfunktionen für Logging, Konfiguration, Validierung und Text-Verarbeitung.
Optimiert für Produktions-RAG-Systeme.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import re
import hashlib
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import Counter


# Globale Logger-Konfiguration (verhindert Handler-Duplikate)
_logger_configured = False
_global_logger = None


@dataclass
class ProcessingStats:
    """Strukturierte Verarbeitungsstatistiken."""
    files_processed: int = 0
    files_failed: int = 0
    total_documents: int = 0
    total_pages: int = 0
    total_processing_time: float = 0.0
    avg_time_per_file: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def setup_logging(
    level: str = "INFO", 
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    force_reconfigure: bool = False
) -> logging.Logger:
    """
    Richtet robustes Logging für den Docling Adapter ein.
    Verhindert Handler-Duplikate bei mehrfachen Aufrufen.
    
    Args:
        level: Log-Level ("DEBUG", "INFO", "WARNING", "ERROR")
        format_string: Optionales Custom-Format
        log_file: Optional: Zusätzlich in Datei loggen
        force_reconfigure: Logger neu konfigurieren erzwingen
        
    Returns:
        Konfigurierter Logger
    """
    global _logger_configured, _global_logger
    
    if _logger_configured and not force_reconfigure and _global_logger:
        return _global_logger
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Logger erstellen/holen
    logger = logging.getLogger("docling_adapter")
    
    # Bestehende Handler entfernen bei Rekonfiguration
    if force_reconfigure:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Nur konfigurieren wenn noch keine Handler vorhanden
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper()))
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(format_string)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Optional: File Handler
        if log_file:
            try:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_path, encoding='utf-8')
                file_handler.setLevel(getattr(logging, level.upper()))
                file_formatter = logging.Formatter(format_string)
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
                
                logger.info(f"Logging in Datei aktiviert: {log_path}")
            except Exception as e:
                logger.warning(f"Konnte Log-Datei nicht erstellen: {e}")
        
        # Propagation verhindern (verhindert doppelte Ausgaben)
        logger.propagate = False
    
    _logger_configured = True
    _global_logger = logger
    return logger


def validate_docling_config(config: Dict[str, Any]) -> bool:
    """
    Erweiterte Validierung einer Docling-Konfiguration.
    
    Args:
        config: Konfigurationsdictionary
        
    Returns:
        True wenn Konfiguration gültig
        
    Raises:
        ValueError: Bei ungültiger Konfiguration
    """
    # Basis-Validierung
    required_fields = [
        "ocr_enabled",
        "table_extraction", 
        "image_extraction",
        "reading_order"
    ]
    
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Docling-Config fehlen Felder: {', '.join(missing_fields)}")
    
    # Erweiterte Typ-Validierung
    field_types = {
        # Boolean-Felder
        "ocr_enabled": bool,
        "table_extraction": bool,
        "image_extraction": bool,
        "formula_extraction": bool,
        "layout_analysis": bool,
        "reading_order": bool,
        "chunk_by_page": bool,
        "preserve_formatting": bool,
        "extract_metadata": bool,
        
        # String-Felder
        "export_format": str,
        "chunk_strategy": str,
        
        # Numerische Felder
        "timeout_seconds": (int, float),
        "max_chunk_size": int,
        "chunk_overlap": int,
    }
    
    for field, expected_type in field_types.items():
        if field in config:
            if not isinstance(config[field], expected_type):
                raise ValueError(
                    f"Feld '{field}' muss {expected_type} sein, ist {type(config[field])}"
                )
    
    # Werte-Validierung
    if "export_format" in config:
        valid_formats = ["markdown", "html", "json", "text"]
        if config["export_format"] not in valid_formats:
            raise ValueError(
                f"Ungültiges export_format: {config['export_format']}. "
                f"Gültig: {valid_formats}"
            )
    
    if "chunk_strategy" in config:
        valid_strategies = ["by_heading", "by_page", "hybrid", "by_element"]
        if config["chunk_strategy"] not in valid_strategies:
            raise ValueError(
                f"Ungültige chunk_strategy: {config['chunk_strategy']}. "
                f"Gültig: {valid_strategies}"
            )
    
    # Numerische Bereiche
    if "timeout_seconds" in config:
        timeout = config["timeout_seconds"]
        if not (10 <= timeout <= 3600):  # 10 Sekunden bis 1 Stunde
            raise ValueError(f"timeout_seconds muss zwischen 10 und 3600 sein, ist {timeout}")
    
    if "max_chunk_size" in config:
        chunk_size = config["max_chunk_size"]
        if not (100 <= chunk_size <= 10000):  # 100 bis 10k Zeichen
            raise ValueError(f"max_chunk_size muss zwischen 100 und 10000 sein, ist {chunk_size}")
    
    return True


def slugify(text: str, max_length: int = 50, allow_unicode: bool = False) -> str:
    """
    Erweiterte Slug-Erstellung mit Unicode-Unterstützung.
    
    Args:
        text: Zu konvertierender Text
        max_length: Maximale Länge des Slugs
        allow_unicode: Unicode-Zeichen beibehalten
        
    Returns:
        Bereinigter Slug
    """
    if not text:
        return "untitled"
    
    # Zu Kleinbuchstaben
    text = text.lower().strip()
    
    if not allow_unicode:
        # Umlaute und Sonderzeichen ersetzen
        replacements = {
            'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
            'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'å': 'a',
            'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
            'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
            'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o',
            'ù': 'u', 'ú': 'u', 'û': 'u',
            'ñ': 'n', 'ç': 'c'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        # Nur alphanumerische Zeichen und Bindestriche behalten
        text = re.sub(r'[^a-z0-9\s-]', '', text)
    else:
        # Unicode beibehalten, aber gefährliche Zeichen entfernen
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', '', text)
    
    # Leerzeichen durch Bindestriche ersetzen
    text = re.sub(r'\s+', '-', text)
    
    # Mehrfache Bindestriche reduzieren
    text = re.sub(r'-+', '-', text)
    
    # Bindestriche am Anfang/Ende entfernen
    text = text.strip('-')
    
    # Auf maximale Länge kürzen (Unicode-aware)
    if len(text) > max_length:
        text = text[:max_length].rstrip('-')
    
    return text or "untitled"


def chunk_text_advanced(
    text: str, 
    chunk_size: int = 1024, 
    overlap: int = 100,
    min_chunk_size: int = 50,
    sentence_splitters: Optional[List[str]] = None,
    preserve_paragraphs: bool = True
) -> List[str]:
    """
    Erweiterte Text-Aufteilung mit intelligenter Satz- und Absatzerkennung.
    
    Args:
        text: Zu teilender Text
        chunk_size: Größe der Chunks in Zeichen
        overlap: Überlappung zwischen Chunks in Zeichen
        min_chunk_size: Minimale Chunk-Größe
        sentence_splitters: Custom Satz-Trenner
        preserve_paragraphs: Absätze wenn möglich nicht trennen
        
    Returns:
        Liste von Text-Chunks
    """
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_chunk_size else []
    
    if sentence_splitters is None:
        sentence_splitters = ['. ', '! ', '? ', '.\n', '!\n', '?\n', '.\r\n', '!\r\n', '?\r\n']
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            # Letzter Chunk
            chunk = text[start:].strip()
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
            break
        
        # Beste Trennstelle finden
        split_pos = _find_best_split_position(
            text, start, end, sentence_splitters, preserve_paragraphs
        )
        
        if split_pos == -1:
            # Notfall: Hart bei chunk_size trennen
            split_pos = end
        
        chunk = text[start:split_pos].strip()
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        
        # Nächster Start mit Überlappung
        start = max(split_pos - overlap, start + 1)  # Verhindere Endlosschleife
    
    return chunks


def _find_best_split_position(
    text: str, 
    start: int, 
    max_end: int, 
    sentence_splitters: List[str],
    preserve_paragraphs: bool
) -> int:
    """
    Findet die optimale Position zum Trennen des Textes.
    
    Returns:
        Position zum Trennen oder -1 wenn keine gute Position gefunden
    """
    search_text = text[start:max_end]
    
    # 1. Versuche Absatzgrenze (höchste Priorität)
    if preserve_paragraphs:
        paragraph_patterns = ['\n\n', '\r\n\r\n', '\n\r\n\r']
        for pattern in paragraph_patterns:
            pos = search_text.rfind(pattern)
            if pos > len(search_text) * 0.5:  # Mindestens 50% der gewünschten Länge
                return start + pos + len(pattern)
    
    # 2. Versuche Satzgrenze
    for splitter in sentence_splitters:
        pos = search_text.rfind(splitter)
        if pos > len(search_text) * 0.7:  # Mindestens 70% der gewünschten Länge
            return start + pos + len(splitter)
    
    # 3. Versuche Wortgrenze
    word_pos = search_text.rfind(' ')
    if word_pos > len(search_text) * 0.8:  # Mindestens 80% der gewünschten Länge
        return start + word_pos + 1
    
    # 4. Versuche Zeilenumbruch
    line_pos = search_text.rfind('\n')
    if line_pos > len(search_text) * 0.6:
        return start + line_pos + 1
    
    return -1  # Keine gute Trennstelle gefunden


def safe_filename(filename: str, max_length: int = 255, replacement_char: str = '_') -> str:
    """
    Erweiterte Dateiname-Bereinigung mit konfigurierbaren Optionen.
    
    Args:
        filename: Original-Dateiname
        max_length: Maximale Länge
        replacement_char: Zeichen für Ersetzung gefährlicher Zeichen
        
    Returns:
        Sicherer Dateiname
    """
    if not filename:
        return "unnamed_file"
    
    # Gefährliche Zeichen für verschiedene Dateisysteme
    unsafe_chars = '<>:"/\\|?*'
    
    # Windows-spezifische reservierte Namen
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    # Gefährliche Zeichen ersetzen
    for char in unsafe_chars:
        filename = filename.replace(char, replacement_char)
    
    # Kontrollzeichen entfernen (ASCII 0-31 und 127)
    filename = ''.join(char for char in filename if ord(char) >= 32 and ord(char) != 127)
    
    # Führende/nachfolgende Punkte und Leerzeichen entfernen
    filename = filename.strip('. ')
    
    # Auf maximale Länge kürzen (Extension berücksichtigen)
    if len(filename) > max_length:
        if '.' in filename:
            name, ext = filename.rsplit('.', 1)
            max_name_length = max_length - len(ext) - 1
            if max_name_length > 0:
                filename = name[:max_name_length] + '.' + ext
            else:
                filename = filename[:max_length]
        else:
            filename = filename[:max_length]
    
    # Reservierte Namen prüfen
    name_without_ext = filename.split('.')[0].upper()
    if name_without_ext in reserved_names:
        filename = f"file_{filename}"
    
    return filename.strip() or "unnamed_file"


def format_file_size(size_bytes: int, decimal_places: int = 1) -> str:
    """
    Erweiterte Dateigröße-Formatierung mit konfigurierbarer Präzision.
    
    Args:
        size_bytes: Größe in Bytes
        decimal_places: Anzahl Dezimalstellen
        
    Returns:
        Formatierte Größe (z.B. "1.5 MB")
    """
    if size_bytes < 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:  # Bytes - keine Dezimalstellen
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.{decimal_places}f} {units[unit_index]}"


def validate_pdf_file(pdf_path: Union[str, Path], check_content: bool = True) -> Dict[str, Any]:
    """
    Erweiterte PDF-Validierung mit detaillierter Diagnose.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        check_content: Auch Inhalt validieren (langsamer)
        
    Returns:
        Dictionary mit Validierungs-Ergebnissen
    """
    pdf_path = Path(pdf_path)
    
    result = {
        "is_valid": False,
        "exists": False,
        "is_pdf_extension": False,
        "has_pdf_header": False,
        "has_pdf_footer": False,
        "is_readable": False,
        "file_size": 0,
        "errors": [],
        "warnings": []
    }
    
    # Existenz prüfen
    if not pdf_path.exists():
        result["errors"].append("Datei existiert nicht")
        return result
    
    result["exists"] = True
    result["file_size"] = pdf_path.stat().st_size
    
    # Dateierweiterung prüfen
    if pdf_path.suffix.lower() != '.pdf':
        result["warnings"].append(f"Ungewöhnliche Dateierweiterung: {pdf_path.suffix}")
    else:
        result["is_pdf_extension"] = True
    
    # Datei-Header und -Footer prüfen
    try:
        with open(pdf_path, 'rb') as f:
            # Header prüfen (erste 4 Bytes)
            header = f.read(4)
            if header == b'%PDF':
                result["has_pdf_header"] = True
            else:
                result["errors"].append(f"Ungültiger PDF-Header: {header}")
            
            # Footer prüfen (letzte 1024 Bytes nach %%EOF)
            if result["file_size"] > 1024:
                f.seek(-1024, 2)  # 1024 Bytes vom Ende
                footer_content = f.read()
                if b'%%EOF' in footer_content:
                    result["has_pdf_footer"] = True
                else:
                    result["warnings"].append("Kein %%EOF-Marker gefunden")
            
    except Exception as e:
        result["errors"].append(f"Fehler beim Lesen der Datei: {str(e)}")
        return result
    
    # Basis-Validierung
    result["is_readable"] = True
    
    # Erweiterte Inhalts-Validierung
    if check_content and result["has_pdf_header"]:
        try:
            # Versuche PDF mit PyPDF2/pypdf zu öffnen
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                try:
                    from pypdf import PdfReader
                except ImportError:
                    result["warnings"].append("PyPDF2/pypdf nicht verfügbar für Inhalts-Validierung")
                    PdfReader = None
            
            if PdfReader:
                try:
                    reader = PdfReader(str(pdf_path), strict=False)
                    page_count = len(reader.pages)
                    
                    if page_count == 0:
                        result["errors"].append("PDF enthält keine Seiten")
                    elif page_count > 10000:
                        result["warnings"].append(f"Sehr viele Seiten: {page_count}")
                    
                    # Versuche erste Seite zu lesen
                    if page_count > 0:
                        first_page = reader.pages[0]
                        text = first_page.extract_text()
                        if not text.strip():
                            result["warnings"].append("Erste Seite enthält keinen extrahierbaren Text")
                    
                except Exception as e:
                    result["errors"].append(f"PDF-Inhalts-Validierung fehlgeschlagen: {str(e)}")
        
        except Exception as e:
            result["warnings"].append(f"Erweiterte Validierung fehlgeschlagen: {str(e)}")
    
    # Gesamtbewertung
    result["is_valid"] = (
        result["exists"] and 
        result["has_pdf_header"] and 
        result["is_readable"] and 
        len(result["errors"]) == 0
    )
    
    return result


def get_pdf_info(pdf_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Erweiterte PDF-Informations-Sammlung.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        
    Returns:
        Dictionary mit detaillierten PDF-Informationen
    """
    pdf_path = Path(pdf_path)
    
    info = {
        "path": str(pdf_path),
        "filename": pdf_path.name,
        "stem": pdf_path.stem,
        "suffix": pdf_path.suffix,
        "exists": pdf_path.exists(),
        "validation": {},
        "file_stats": {},
        "pdf_metadata": {},
        "estimated_processing_time": "unbekannt"
    }
    
    if pdf_path.exists():
        # Datei-Statistiken
        stat = pdf_path.stat()
        info["file_stats"] = {
            "size_bytes": stat.st_size,
            "size_formatted": format_file_size(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "is_readable": os.access(pdf_path, os.R_OK),
            "is_writable": os.access(pdf_path, os.W_OK)
        }
        
        # PDF-Validierung
        info["validation"] = validate_pdf_file(pdf_path, check_content=True)
        
        # Geschätzte Verarbeitungszeit
        size_mb = stat.st_size / (1024 * 1024)
        info["estimated_processing_time"] = estimate_processing_time(size_mb)
        
        # Basis-PDF-Metadaten (falls möglich)
        if info["validation"]["is_valid"]:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(pdf_path), strict=False)
                
                info["pdf_metadata"] = {
                    "page_count": len(reader.pages),
                    "is_encrypted": reader.is_encrypted,
                    "metadata_available": reader.metadata is not None
                }
                
                if reader.metadata:
                    metadata = reader.metadata
                    info["pdf_metadata"]["title"] = str(metadata.get('/Title', '')).strip()
                    info["pdf_metadata"]["author"] = str(metadata.get('/Author', '')).strip()
                    info["pdf_metadata"]["creator"] = str(metadata.get('/Creator', '')).strip()
                    
            except Exception as e:
                info["pdf_metadata"]["extraction_error"] = str(e)
    
    return info


def create_output_filename(
    input_path: Union[str, Path], 
    suffix: str = "", 
    extension: str = ".json",
    timestamp: bool = False,
    hash_input: bool = False
) -> str:
    """
    Erweiterte Ausgabe-Dateiname-Erstellung.
    
    Args:
        input_path: Eingabe-Dateipfad
        suffix: Optionaler Suffix (z.B. "_processed")
        extension: Dateierweiterung für Ausgabe
        timestamp: Zeitstempel hinzufügen
        hash_input: Hash des Eingabe-Pfads hinzufügen
        
    Returns:
        Ausgabe-Dateiname
    """
    input_path = Path(input_path)
    base_name = input_path.stem
    
    # Suffix-Komponenten sammeln
    suffix_parts = []
    
    if suffix:
        suffix_parts.append(suffix)
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_parts.append(timestamp_str)
    
    if hash_input:
        path_hash = hashlib.md5(str(input_path).encode()).hexdigest()[:8]
        suffix_parts.append(path_hash)
    
    # Finalen Namen zusammenbauen
    if suffix_parts:
        final_suffix = "_" + "_".join(suffix_parts)
    else:
        final_suffix = ""
    
    output_name = f"{base_name}{final_suffix}{extension}"
    return safe_filename(output_name)


def save_json(
    data: Any, 
    output_path: Union[str, Path], 
    indent: int = 2, 
    ensure_ascii: bool = False,
    backup_existing: bool = True
) -> bool:
    """
    Robuste JSON-Speicherung mit Backup-Option.
    
    Args:
        data: Zu speichernde Daten
        output_path: Ausgabe-Pfad
        indent: JSON-Einrückung
        ensure_ascii: ASCII-Encoding erzwingen
        backup_existing: Bestehende Datei sichern
        
    Returns:
        True wenn erfolgreich gespeichert
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup erstellen falls Datei existiert
        if backup_existing and output_path.exists():
            backup_path = output_path.with_suffix(f".backup_{int(time.time())}{output_path.suffix}")
            output_path.rename(backup_path)
        
        # JSON speichern
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        
        return True
        
    except Exception as e:
        logger = logging.getLogger("docling_adapter")
        logger.error(f"Fehler beim Speichern von JSON: {str(e)}")
        return False


def load_json(input_path: Union[str, Path], default: Any = None) -> Any:
    """
    Robuste JSON-Ladung mit Default-Wert.
    
    Args:
        input_path: Pfad zur JSON-Datei
        default: Rückgabe-Wert bei Fehler
        
    Returns:
        Geladene Daten oder default-Wert
    """
    try:
        input_path = Path(input_path)
        if not input_path.exists():
            return default
        
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except Exception as e:
        logger = logging.getLogger("docling_adapter")
        logger.error(f"Fehler beim Laden von JSON: {str(e)}")
        return default


def print_processing_stats(
    stats: Union[Dict[str, Any], ProcessingStats], 
    title: str = "Verarbeitungsstatistiken",
    show_details: bool = True
) -> None:
    """
    Erweiterte Statistik-Ausgabe mit strukturierter Formatierung.
    
    Args:
        stats: Statistiken-Dictionary oder ProcessingStats
        title: Titel für die Ausgabe
        show_details: Detaillierte Ausgabe aktivieren
    """
    print(f"\n📊 {title}")
    print("=" * (len(title) + 4))
    
    # ProcessingStats zu Dict konvertieren
    if isinstance(stats, ProcessingStats):
        stats_dict = asdict(stats)
    else:
        stats_dict = stats
    
    # Kategorisierte Ausgabe
    categories = {
        "Dateien": ["files_processed", "files_failed", "total_documents"],
        "Inhalt": ["total_pages", "blocks", "tables", "images", "formulas"],
        "Qualität": ["quality_score", "ocr_ratio", "structure_quality_score"],
        "Performance": ["processing_time_seconds", "total_processing_time", "avg_time_per_file"],
        "Technisch": ["docling_available", "docling_version", "file_size_mb"]
    }
    
    for category, keys in categories.items():
        category_stats = {k: v for k, v in stats_dict.items() if k in keys and v is not None}
        
        if category_stats:
            print(f"\n{category}:")
            for key, value in category_stats.items():
                formatted_value = _format_stat_value(key, value)
                print(f"   {_format_stat_key(key)}: {formatted_value}")
    
    # Restliche Statistiken
    if show_details:
        remaining_stats = {
            k: v for k, v in stats_dict.items() 
            if k not in sum(categories.values(), []) and not k.startswith('_')
        }
        
        if remaining_stats:
            print(f"\nWeitere Details:")
            for key, value in remaining_stats.items():
                if isinstance(value, (list, dict)) and len(str(value)) > 100:
                    print(f"   {_format_stat_key(key)}: {type(value).__name__} ({len(value)} Einträge)")
                else:
                    formatted_value = _format_stat_value(key, value)
                    print(f"   {_format_stat_key(key)}: {formatted_value}")


def _format_stat_key(key: str) -> str:
    """Formatiert Statistik-Schlüssel für Ausgabe."""
    # Unterstriche durch Leerzeichen ersetzen und Titel-Case
    return key.replace('_', ' ').title()


def _format_stat_value(key: str, value: Any) -> str:
    """Formatiert Statistik-Werte für Ausgabe."""
    if value is None:
        return "N/A"
    
    # Prozent-Werte
    if 'ratio' in key or 'score' in key:
        if isinstance(value, (int, float)) and 0 <= value <= 1:
            return f"{value:.1%}"
    
    # Zeit-Werte
    if 'time' in key and isinstance(value, (int, float)):
        if value < 60:
            return f"{value:.1f}s"
        elif value < 3600:
            return f"{value/60:.1f}min"
        else:
            return f"{value/3600:.1f}h"
    
    # Dateigrößen
    if 'size' in key and 'mb' in key and isinstance(value, (int, float)):
        return f"{value:.1f} MB"
    
    # Boolean-Werte
    if isinstance(value, bool):
        return "✅" if value else "❌"
    
    # Listen
    if isinstance(value, list):
        return f"{len(value)} Einträge"
    
    # Dictionaries
    if isinstance(value, dict):
        return f"{len(value)} Felder"
    
    return str(value)


def estimate_processing_time(
    file_size_mb: float, 
    pages: Optional[int] = None,
    complexity_factor: float = 1.0
) -> str:
    """
    Verbesserte Verarbeitungszeit-Schätzung.
    
    Args:
        file_size_mb: Dateigröße in MB
        pages: Anzahl Seiten (optional)
        complexity_factor: Komplexitäts-Faktor (1.0 = normal, >1.0 = komplexer)
        
    Returns:
        Geschätzte Zeit als String
    """
    # Basis-Schätzung: 20-40 Sekunden pro MB je nach Komplexität
    base_time_per_mb = 20 + (complexity_factor - 1.0) * 20
    time_by_size = file_size_mb * base_time_per_mb
    
    # Seiten-basierte Schätzung: 3-8 Sekunden pro Seite
    if pages:
        base_time_per_page = 3 + (complexity_factor - 1.0) * 5
        time_by_pages = pages * base_time_per_page
    else:
        # Schätze 2-3 Seiten pro MB
        estimated_pages = file_size_mb * 2.5
        time_by_pages = estimated_pages * (3 + (complexity_factor - 1.0) * 5)
    
    # Nehme das Maximum der beiden Schätzungen
    estimated_seconds = max(time_by_size, time_by_pages)
    
    # Minimum: 5 Sekunden
    estimated_seconds = max(5, estimated_seconds)
    
    if estimated_seconds < 60:
        return f"~{int(estimated_seconds)} Sekunden"
    elif estimated_seconds < 3600:
        return f"~{int(estimated_seconds / 60)} Minuten"
    else:
        hours = int(estimated_seconds / 3600)
        minutes = int((estimated_seconds % 3600) / 60)
        if minutes > 0:
            return f"~{hours}h {minutes}min"
        else:
            return f"~{hours} Stunden"


def calculate_text_statistics(text: str) -> Dict[str, Any]:
    """
    Berechnet detaillierte Text-Statistiken.
    
    Args:
        text: Zu analysierender Text
        
    Returns:
        Dictionary mit Text-Statistiken
    """
    if not text:
        return {"error": "Leerer Text"}
    
    # Basis-Statistiken
    char_count = len(text)
    char_count_no_spaces = len(text.replace(' ', ''))
    
    # Wörter und Sätze
    words = text.split()
    word_count = len(words)
    
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Zeilen und Absätze
    lines = text.split('\n')
    line_count = len(lines)
    
    paragraphs = re.split(r'\n\s*\n', text)
    paragraph_count = len([p for p in paragraphs if p.strip()])
    
    # Durchschnittswerte
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    avg_sentence_length = word_count / max(1, sentence_count)
    avg_paragraph_length = word_count / max(1, paragraph_count)
    
    # Zeichen-Analyse
    char_counter = Counter(text.lower())
    most_common_chars = char_counter.most_common(10)
    
    # Sprach-Indikatoren
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, char_count)
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(1, char_count)
    punctuation_ratio = sum(1 for c in text if c in '.,!?;:') / max(1, char_count)
    
    return {
        "char_count": char_count,
        "char_count_no_spaces": char_count_no_spaces,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "line_count": line_count,
        "paragraph_count": paragraph_count,
        "avg_word_length": round(avg_word_length, 2),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_paragraph_length": round(avg_paragraph_length, 2),
        "uppercase_ratio": round(uppercase_ratio, 3),
        "digit_ratio": round(digit_ratio, 3),
        "punctuation_ratio": round(punctuation_ratio, 3),
        "most_common_chars": most_common_chars,
        "readability_estimate": _estimate_readability(avg_sentence_length, avg_word_length)
    }


def _estimate_readability(avg_sentence_length: float, avg_word_length: float) -> str:
    """Einfache Lesbarkeits-Schätzung."""
    # Vereinfachte Flesch-Kincaid-ähnliche Heuristik
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
    
    if score >= 90:
        return "Sehr einfach"
    elif score >= 80:
        return "Einfach"
    elif score >= 70:
        return "Ziemlich einfach"
    elif score >= 60:
        return "Standard"
    elif score >= 50:
        return "Ziemlich schwer"
    elif score >= 30:
        return "Schwer"
    else:
        return "Sehr schwer"


# Erweiterte Konstanten und Konfiguration
class Config:
    """Zentrale Konfigurationskonstanten."""
    
    # Text-Verarbeitung
    DEFAULT_CHUNK_SIZE = 1024
    DEFAULT_CHUNK_OVERLAP = 100
    MIN_CHUNK_SIZE = 50
    MAX_CHUNK_SIZE = 8192
    
    # Datei-Verarbeitung
    MAX_FILENAME_LENGTH = 255
    SUPPORTED_EXTENSIONS = ['.pdf']
    MAX_FILE_SIZE_MB = 500
    
    # Qualitäts-Schwellwerte
    HIGH_QUALITY_THRESHOLD = 0.8
    LOW_QUALITY_THRESHOLD = 0.3
    HIGH_OCR_RATIO_THRESHOLD = 0.7
    
    # Performance
    DEFAULT_TIMEOUT_SECONDS = 300
    MAX_TIMEOUT_SECONDS = 3600
    CACHE_SIZE_LIMIT = 1000
    
    # Logging
    DEFAULT_LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# Globale Instanzen
logger = setup_logging()

# Export für externe Nutzung
__all__ = [
    "setup_logging",
    "validate_docling_config", 
    "slugify",
    "chunk_text_advanced",
    "safe_filename",
    "format_file_size",
    "validate_pdf_file",
    "get_pdf_info",
    "create_output_filename",
    "save_json",
    "load_json",
    "print_processing_stats",
    "estimate_processing_time",
    "calculate_text_statistics",
    "ProcessingStats",
    "Config"
] 