"""
Tabellen-Utilities für PDF-Verarbeitung

Dieses Modul enthält Funktionen zur:
- Tabellen-Erkennung und -Konvertierung
- Tabellen-Metadaten-Extraktion
- Qualitätsbewertung von Tabellen
"""

from typing import Tuple, Dict, Any

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
                table_meta['table_rows'] = len(lines) - 1  # Header abziehen
                # Spalten aus der ersten Zeile schätzen
                first_line_cols = lines[0].count('|') - 1
                table_meta['table_cols'] = max(1, first_line_cols)
        
        # Qualitätsbewertung
        if len(table_text) > 50 and table_meta['table_rows'] > 0:
            table_meta['table_quality'] = 'good'
        elif len(table_text) > 20:
            table_meta['table_quality'] = 'medium'
        else:
            table_meta['table_quality'] = 'poor'
        
        # Zusätzliche Metadaten
        table_meta['table_text_length'] = len(table_text)
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