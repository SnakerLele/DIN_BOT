"""
Header-Utilities für PDF-Verarbeitung

Dieses Modul enthält Funktionen zur:
- Hierarchischen Header-Analyse
- Header-Level-Klassifikation  
- Metadaten-Extraktion für Chunks
- Header-Statistiken und Zusammenfassungen
"""

from typing import List, Dict, Any

def classify_header_level(element_text: str, element_type: str) -> str:
    """
    Klassifiziert Header-Ebene basierend auf allgemeinen Text-Eigenschaften.
    
    Args:
        element_text: Text des Elements
        element_type: Typ des Elements
        
    Returns:
        str: Header-Level ('h1', 'h2', 'h3', oder 'content')
    """
    import re

    text_stripped: str = element_text.strip()
    text_length: int = len(text_stripped)

    # ---------- Frühzeitige Filter ------------------------------------------------
    # 1) Unplausible Längen
    if text_length < 3 or text_length > 200:
        return 'content'

    # 2) OCR-Rauschen: hoher Anteil Sonderzeichen
    non_alpha_ratio = sum(1 for c in text_stripped if not c.isalnum() and c not in {'.', '-', ' ', ':'}) / text_length
    if non_alpha_ratio > 0.4:
        return 'content'

    # 3) Typische Autorenzeile (viele Kommas oder Sonderzeichen † ‡)
    if (text_stripped.count(',') >= 2 and '@' not in text_stripped) or any(sym in text_stripped for sym in '†‡'):
        return 'content'

    # 4) Abkürzungszeilen in Großbuchstaben (≥3 Wörter, Ø Wortlänge ≤4) → wahrscheinlich Tabelle/Legende
    if text_stripped.isupper():
        tokens = text_stripped.split()
        if len(tokens) >= 3 and (sum(len(t) for t in tokens) / len(tokens)) <= 4:
            return 'content'

    # ---------- Numerische/Römische Gliederungen ----------------------------------
    numeric_match = re.match(r'^(?P<num_seq>\d+(?:\.\d+)*)(?:\.)?\s+.+', text_stripped)
    if numeric_match:
        seq = numeric_match.group('num_seq')
        depth_segments = seq.count('.') + 1  # "2" =>1, "2.1"=>2, "2.1.3"=>3
        if depth_segments == 1:
            return 'h1'
        elif depth_segments == 2:
            return 'h2'
        else:
            return 'h3'

    roman_match = re.match(r'^([IVXLC]+)(?:\.|\))?\s+.+', text_stripped, flags=re.IGNORECASE)
    if roman_match:
        return 'h1'

    # ---------- Groß/Klein-Schreibung & Element-Typ --------------------------------
    if element_type == 'Title':
        # Typische Kapitelüberschriften ohne Numerierung explizit als H1 zulassen
        top_level_keywords = {
            'abstract', 'introduction', 'methods', 'method', 'materials', 'results',
            'discussion', 'conclusion', 'conclusions', 'related work', 'acknowledgments',
            'acknowledgements', 'references', 'background'
        }
        if text_stripped.lower() in top_level_keywords:
            return 'h1'
        # Andernfalls konservativ H2
        return 'h2'
    if element_type == 'Header':
        return 'h2'
    if element_type == 'SubHeader':
        return 'h3'

    # ---------- Weitere Heuristiken ------------------------------------------------
    # Einzelnes Großwort (<=4 Zeichen) → wahrscheinlich Content (z. B. Tabellenkennung)
    if text_stripped.isupper() and len(text_stripped.split()) == 1 and text_length <= 4:
        return 'content'

    # Buchstabenpräfix ("A. Grundlagen")
    if re.match(r'^[A-Z]\.\s+.+', text_stripped):
        return 'h2'

    # Fallback → Content
    return 'content'

def analyze_hierarchical_headers(elements: list) -> Dict[int, Dict[str, Any]]:
    """
    Analysiert Unstructured-Elemente und erstellt eine Header-Zuordnung.
    
    Da ElementMetadata read-only ist, geben wir eine Mapping-Tabelle zurück.
    
    Args:
        elements: Liste von Unstructured-Elementen
        
    Returns:
        dict: Mapping von Element-Index zu Header-Metadaten
    """
    print("[HEADER-ANALYSE] Starte hierarchische Header-Analyse...")
    
    current_headers = {
        'h1': None,
        'h2': None, 
        'h3': None,
        'current_section': None
    }
    
    header_stats = {
        'titles_found': 0,
        'headers_found': 0,
        'elements_with_headers': 0
    }
    
    # Mapping von Element-Index zu Header-Metadaten
    header_mapping = {}
    
    for i, element in enumerate(elements):
        element_text = str(element).strip()
        element_type = type(element).__name__
        
        # Kandidat für Header? (Nur Title / Header Elemente berücksichtigen)
        header_level = classify_header_level(element_text, element_type)
        
        # Ein Element gilt nur dann als Header, wenn die Klassifikation h1/h2/h3 zurückgibt
        is_header = header_level in ['h1', 'h2', 'h3']
        
        if is_header:
            if header_level == 'h1':
                # Neue Hauptüberschrift - Reset aller Sub-Header
                current_headers['h1'] = element_text
                current_headers['h2'] = None
                current_headers['h3'] = None
                current_headers['current_section'] = element_text
                header_stats['titles_found'] += 1
                print(f"  [H1] Neue Hauptüberschrift: '{element_text}'")
                
            elif header_level == 'h2':
                # Neue Unterüberschrift - Reset nur H3
                current_headers['h2'] = element_text
                current_headers['h3'] = None
                current_headers['current_section'] = element_text
                header_stats['headers_found'] += 1
                print(f"  [H2] Neue Unterüberschrift: '{element_text}'")
                
            elif header_level == 'h3':
                # Neue Detail-Überschrift
                current_headers['h3'] = element_text
                current_headers['current_section'] = element_text
                header_stats['headers_found'] += 1
                print(f"  [H3] Neue Detail-Überschrift: '{element_text}'")
        
        # Erstelle Header-Metadaten für dieses Element
        element_header_metadata = {
            'element_type': element_type,
            'is_header': is_header
        }
        
        # Füge aktuelle Header hinzu
        for key in ['h1', 'h2', 'h3', 'current_section']:
            if current_headers[key]:
                element_header_metadata[f'section_{key}'] = current_headers[key]
        
        # Speichere Metadaten für diesen Element-Index
        header_mapping[i] = element_header_metadata
        
        if any(current_headers.values()):
            header_stats['elements_with_headers'] += 1
    
    print_header_analysis_summary(header_stats, len(elements))
    
    return header_mapping

def extract_header_info_for_chunk(
    element_indices: List[int], 
    header_mapping: Dict[int, Dict[str, Any]], 
    elements: list
) -> Dict[str, Any]:
    """
    Extrahiert Header-Informationen für einen spezifischen Chunk.
    
    Args:
        element_indices: Liste der Element-Indices im Chunk
        header_mapping: Mapping von Element-Index zu Header-Metadaten
        elements: Ursprüngliche Unstructured-Elemente
        
    Returns:
        Dictionary mit Header-Metadaten für den Chunk
    """
    chunk_header_info = {
        'section_h1': None,
        'section_h2': None, 
        'section_h3': None,
        'current_section': None,
        'header_elements': []
    }
    
    for element_idx in element_indices:
        if element_idx in header_mapping:
            element_headers = header_mapping[element_idx]
            
            # Sammle Header-Informationen (verwende die letzten gefundenen)
            for key in ['section_h1', 'section_h2', 'section_h3', 'current_section']:
                if element_headers.get(key):
                    chunk_header_info[key] = element_headers[key]
            
            # Sammle Header-Elemente
            if element_headers.get('is_header'):
                chunk_header_info['header_elements'].append({
                    'text': str(elements[element_idx]).strip(),
                    'type': element_headers.get('element_type'),
                    'index': element_idx
                })
    
    # Zusätzliche Header-Statistiken
    header_count = len(chunk_header_info['header_elements'])
    metadata_additions = {
        'header_count': header_count,
        'contains_headers': header_count > 0
    }
    
    if chunk_header_info['header_elements']:
        metadata_additions['first_header'] = chunk_header_info['header_elements'][0]['text']
    
    # Entferne komplexe Datenstrukturen und None-Werte (ChromaDB kompatibel)
    clean_header_info = {}
    for k, v in chunk_header_info.items():
        if k == 'header_elements':
            # Überspringe komplexe Datenstrukturen
            continue
        elif v is not None and isinstance(v, (str, int, float, bool)):
            clean_header_info[k] = v
    
    clean_header_info.update(metadata_additions)
    
    return clean_header_info

def print_header_analysis_summary(header_stats: Dict[str, int], total_elements: int):
    """
    Gibt eine Zusammenfassung der Header-Analyse aus.
    
    Args:
        header_stats: Statistiken der Header-Analyse
        total_elements: Gesamtanzahl der verarbeiteten Elemente
    """
    print(f"[HEADER-ANALYSE] Analyse abgeschlossen:")
    print(f"  - H1-Überschriften erkannt: {header_stats['titles_found']}")
    print(f"  - H2/H3-Überschriften erkannt: {header_stats['headers_found']}")
    print(f"  - Elemente mit Header-Zuordnung: {header_stats['elements_with_headers']}")
    print(f"  - Gesamt-Elemente verarbeitet: {total_elements}") 