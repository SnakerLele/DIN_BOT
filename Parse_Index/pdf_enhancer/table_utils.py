"""
Tabellen-Utilities für PDF-Verarbeitung

Dieses Modul enthält Funktionen zur:
- Tabellen-Erkennung und -Konvertierung mit Unstructured (bisherig)
- Tabellen-Extraktion mit img2table (neu, robuster)
- Tabellen-Metadaten-Extraktion
- Qualitätsbewertung von Tabellen
"""

from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def convert_table_to_text_and_meta(table_element) -> Tuple[str, Dict[str, Any]]:
    """
    Konvertiert ein Tabellen-Element zu Text und Metadaten.
    
    Args:
        table_element: Unstructured Table-Element
        
    Returns:
        Tuple aus (Markdown-Text, Metadaten-Dict)
    """
    try:
        # Versuche verschiedene Konvertierungsmethoden
        table_text = ""
        table_meta = {
            'is_table': True,
            'table_extraction_method': 'unknown',
            'table_rows': 0,
            'table_cols': 0,
            'table_quality': 'unknown'
        }
        
        # Methode 1: Markdown-Konvertierung (bevorzugt für Embeddings)
        try:
            if hasattr(table_element, 'to_markdown'):
                table_text = table_element.to_markdown(index=False)
                table_meta['table_extraction_method'] = 'markdown'
                print(f"  [TABLE] Markdown-Konvertierung erfolgreich ({len(table_text)} Zeichen)")
            else:
                # Fallback: Text-Repräsentation
                table_text = str(table_element)
                table_meta['table_extraction_method'] = 'text_fallback'
                print(f"  [TABLE] Fallback zu Text-Repräsentation ({len(table_text)} Zeichen)")
        except Exception as e:
            # Fallback: Einfache String-Konvertierung
            table_text = str(table_element)
            table_meta['table_extraction_method'] = 'string_fallback'
            print(f"  [TABLE] String-Fallback verwendet: {str(e)}")
        
        # Schätze Tabellen-Dimensionen aus dem Text
        if '|' in table_text:
            lines = [line.strip() for line in table_text.split('\n') if line.strip() and '|' in line]
            if lines:
                table_meta['table_rows'] = int(len(lines) - 1)  # Header abziehen
                # Spalten aus der ersten Zeile schätzen
                first_line_cols = lines[0].count('|') - 1
                table_meta['table_cols'] = int(max(1, first_line_cols))
        
        # Qualitätsbewertung
        if len(table_text) > 50 and table_meta['table_rows'] > 0:
            table_meta['table_quality'] = 'good'
        elif len(table_text) > 20:
            table_meta['table_quality'] = 'medium'
        else:
            table_meta['table_quality'] = 'poor'
        
        # Zusätzliche Metadaten
        table_meta['table_text_length'] = int(len(table_text))
        table_meta['content_type'] = 'table'
        
        return table_text, table_meta
        
    except Exception as e:
        print(f"❌ Fehler bei Tabellen-Konvertierung: {str(e)}")
        # Notfall-Fallback
        fallback_text = str(table_element)
        fallback_meta = {
            'is_table': True,
            'table_extraction_method': 'error_fallback',
            'table_quality': 'poor',
            'content_type': 'table',
            'extraction_error': str(e)
        }
        return fallback_text, fallback_meta


def extract_tables_with_img2table(pdf_path: Path, 
                                  use_img2table: bool = True,
                                  **img2table_kwargs) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Extrahiert Tabellen aus einem PDF mit img2table als primäre Methode
    
    Diese Funktion nutzt img2table für robuste Tabellen-Extraktion:
    - Konvertiert PDF-Seiten zu hochauflösenden Bildern
    - Nutzt OpenCV für zuverlässige Tabellen-Erkennung
    - Liefert direkt Markdown-formatierte Tabellen
    - Funktioniert besser bei gescannten PDFs und komplexen Layouts
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        use_img2table: Ob img2table verwendet werden soll
        **img2table_kwargs: Zusätzliche Parameter für img2table
        
    Returns:
        Tuple aus (Markdown-Tabellen-Liste, Metadaten-Liste)
    """
    if not use_img2table:
        logger.info("img2table deaktiviert, verwende Unstructured Fallback")
        return [], []
    
    try:
        # img2table Import (lazy loading)
        from .img2table_utils import create_img2table_extractor
        
        # Standard-Parameter für img2table
        default_params = {
            'lang': 'deu+eng',  # Deutsch + Englisch für beste Ergebnisse
            'dpi': 300,
            'borderless_tables': True,
            'implicit_rows': False,
            'implicit_columns': False
        }
        default_params.update(img2table_kwargs)
        
        # img2table Extraktor erstellen
        extractor = create_img2table_extractor(**default_params)
        if extractor is None:
            logger.warning("img2table Extraktor konnte nicht erstellt werden")
            return [], []
        
        # Tabellen extrahieren
        markdown_tables, metadata_list = extractor.extract_tables_from_pdf(pdf_path)
        
        if markdown_tables:
            logger.info(f"🎯 img2table erfolgreich: {len(markdown_tables)} Tabellen extrahiert")
            return markdown_tables, metadata_list
        else:
            logger.info("ℹ️ img2table: Keine Tabellen gefunden")
            return [], []
            
    except ImportError as e:
        logger.warning(f"img2table nicht verfügbar: {e}")
        return [], []
    except Exception as e:
        logger.error(f"❌ img2table Extraktion fehlgeschlagen: {e}")
        return [], []


def create_table_nodes_from_img2table(markdown_tables: List[str], 
                                     metadata_list: List[Dict[str, Any]],
                                     source_file: str) -> List[Dict[str, Any]]:
    """
    Erstellt LlamaIndex-kompatible Nodes aus img2table Ergebnissen
    
    Args:
        markdown_tables: Liste von Markdown-Tabellen
        metadata_list: Liste der Tabellen-Metadaten
        source_file: Name der Quelldatei
        
    Returns:
        Liste von Node-Dictionaries für LlamaIndex
    """
    nodes = []
    
    for i, (markdown, metadata) in enumerate(zip(markdown_tables, metadata_list)):
        # Node-Text: Markdown-Tabelle mit Kontext
        node_text = f"**Tabelle {i+1}** (Seite {metadata.get('page_number', '?')}):\n\n{markdown}"
        
        # Erweiterte Metadaten für den Node (alle Werte zu nativen Python-Typen konvertieren)
        # bbox als String für ChromaDB-Kompatibilität
        bbox_str = ""
        if metadata.get('bbox'):
            bbox = metadata.get('bbox')
            bbox_str = f"x1:{bbox.get('x1', 0)},y1:{bbox.get('y1', 0)},x2:{bbox.get('x2', 0)},y2:{bbox.get('y2', 0)}"
        
        node_metadata = {
            'source_file': str(source_file),
            'node_type': 'table',
            'extraction_method': 'img2table',
            'page_number': int(metadata.get('page_number', 0)),
            'table_id': str(metadata.get('table_id', '')),
            'table_index': int(metadata.get('table_index', 0)),
            'table_rows': int(metadata.get('rows', 0)),
            'table_columns': int(metadata.get('columns', 0)),
            'table_quality': str(metadata.get('table_quality', 'unknown')),
            'character_count': int(len(node_text)),
            'bbox_coordinates': bbox_str,
            'is_table': True,
            'content_type': 'table'
        }
        
        # Node-Dictionary erstellen
        node_dict = {
            'text': node_text,
            'metadata': node_metadata,
            'id': f"{source_file}_{metadata.get('table_id', f'table_{i}')}"
        }
        
        nodes.append(node_dict)
        logger.info(f"✅ Table-Node erstellt: {metadata.get('rows')}x{metadata.get('columns')} Tabelle")
    
    return nodes


def enhanced_table_extraction(pdf_path: Path, 
                             prefer_img2table: bool = True,
                             fallback_to_unstructured: bool = True) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Verbesserte Tabellen-Extraktion mit img2table als primäre Methode
    
    Diese Funktion kombiniert img2table und Unstructured für optimale Ergebnisse:
    1. Versucht zuerst img2table (robuster, bessere Markdown-Ausgabe)
    2. Falls img2table keine Tabellen findet, Fallback zu Unstructured
    3. Kombiniert ggf. Ergebnisse beider Methoden
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        prefer_img2table: Ob img2table bevorzugt werden soll
        fallback_to_unstructured: Ob Unstructured als Fallback genutzt werden soll
        
    Returns:
        Tuple aus (Markdown-Tabellen-Liste, Metadaten-Liste)
    """
    all_tables = []
    all_metadata = []
    
    # Schritt 1: img2table Extraktion
    if prefer_img2table:
        logger.info("🚀 Starte primäre Tabellen-Extraktion mit img2table")
        img2table_tables, img2table_meta = extract_tables_with_img2table(pdf_path)
        
        if img2table_tables:
            all_tables.extend(img2table_tables)
            all_metadata.extend(img2table_meta)
            logger.info(f"✅ img2table: {len(img2table_tables)} Tabellen extrahiert")
        else:
            logger.info("ℹ️ img2table: Keine Tabellen gefunden")
    
    # Schritt 2: Unstructured Fallback (falls gewünscht und nötig)
    if fallback_to_unstructured and not all_tables:
        logger.info("🔄 Fallback: Versuche Unstructured Tabellen-Extraktion")
        
        try:
            # TODO: Hier könnte Unstructured-basierte Extraktion eingefügt werden
            # Für jetzt nur ein Platzhalter
            logger.info("⚠️ Unstructured Fallback noch nicht implementiert")
        except Exception as e:
            logger.error(f"❌ Unstructured Fallback fehlgeschlagen: {e}")
    
    # Zusammenfassung
    total_tables = len(all_tables)
    if total_tables > 0:
        logger.info(f"🎯 Tabellen-Extraktion abgeschlossen: {total_tables} Tabellen insgesamt")
    else:
        logger.warning("⚠️ Keine Tabellen in der PDF gefunden")
    
    return all_tables, all_metadata 