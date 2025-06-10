"""
Document Enhancer für das Parse_Index System

Dieses Modul enthält Funktionen zur semantischen Anreicherung und Verbesserung
von PDF-Dokumenten durch:
- Hierarchische Header-Analyse
- Semantische Abschnitts-Gruppierung  
- Tabellen-Erkennung und -Strukturierung
- Metadaten-Anreicherung
- Strukturelle Dokumentenverbesserung
"""

import os
from typing import List, Dict, Any, Tuple
from pathlib import Path

from llama_index.core import Document

def semantic_enhanced_pdf(pdf_path: str) -> List[Document]:
    """
    Erstellt semantisch sinnvolle Dokument-Chunks durch intelligente Gruppierung.
    
    Kombiniert zusammengehörige PDF-Elemente zu kohärenten Abschnitten,
    integriert hierarchische Header-Analyse und erkennt Tabellen für
    reichere Metadaten und bessere Retrieval-Qualität.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        
    Returns:
        Liste von Document-Objekten mit semantischen Abschnitten und Tabellen
    """
    print(f"[SEMANTIC] Verarbeite {pdf_path} mit semantischer Abschnitts-Gruppierung und Tabellen-Erkennung")
    
    # 1. Lade PDF-Elemente mit Unstructured (hi_res für bessere Header- und Tabellen-Erkennung)
    try:
        from unstructured.partition.pdf import partition_pdf
        
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",  # Geändert von "auto" zu "hi_res" für bessere Layout-Erkennung
            include_page_breaks=True,
            include_metadata=True,  # Zusätzliche Metadaten für bessere Header-Erkennung
            combine_text_under_n_chars=0,
            infer_table_structure=True  # Aktiviert Tabellen-Struktur-Erkennung
        )
        
        print(f"[SEMANTIC] {len(elements)} PDF-Elemente geladen (hi_res Strategie mit Tabellen-Erkennung)")
        
    except Exception as e:
        print(f"❌ Fehler beim Laden der PDF-Elemente: {str(e)}")
        return []
    
    # 2. Analysiere hierarchische Headers für alle Elemente
    print(f"[SEMANTIC] Starte hierarchische Header-Analyse für bessere Metadaten...")
    header_mapping = analyze_hierarchical_headers(elements)
    
    # 3. Gruppiere Elemente zu semantischen Abschnitten (inkl. Tabellen-Behandlung)
    semantic_chunks = group_elements_into_semantic_sections(elements)
    
    # 4. Konvertiere zu LlamaIndex Documents mit erweiterten Header-Metadaten
    documents = convert_semantic_chunks_to_documents(
        semantic_chunks, 
        header_mapping, 
        elements, 
        pdf_path
    )
    
    print_semantic_analysis_summary(semantic_chunks, documents)
    
    return documents

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

def group_elements_into_semantic_sections(elements: list) -> List[Dict[str, Any]]:
    """
    Gruppiert PDF-Elemente zu semantisch zusammengehörigen Abschnitten.
    Behandelt Tabellen als separate, eigenständige Chunks.
    
    Args:
        elements: Liste von Unstructured-PDF-Elementen
        
    Returns:
        Liste von semantischen Chunk-Dictionaries
    """
    semantic_chunks = []
    current_chunk = ""
    current_section = None
    current_page = None
    current_element_indices = []
    
    tables_found = 0
    
    for i, element in enumerate(elements):
        element_text = str(element).strip()
        element_type = type(element).__name__
        
        # Extrahiere Seitennummer
        page_number = None
        if hasattr(element, 'metadata') and element.metadata and hasattr(element.metadata, 'page_number'):
            page_number = element.metadata.page_number
        
        # **TABELLEN-BEHANDLUNG**: Erkenne Tabellen-Elemente
        if element_type in ['Table', 'TableChunk']:
            # Speichere den vorherigen Chunk wenn er Inhalt hat
            if current_chunk.strip() and len(current_chunk.strip()) > 50:
                semantic_chunks.append({
                    'text': current_chunk.strip(),
                    'section': current_section or "Unnamed Section",
                    'page_number': current_page,
                    'length': len(current_chunk.strip()),
                    'element_indices': current_element_indices.copy(),
                    'is_table': False
                })
            
            # Verarbeite die Tabelle als eigenen Chunk
            table_text, table_metadata = convert_table_to_text_and_meta(element)
            
            tables_found += 1
            table_section = f"Tabelle {tables_found}" + (f" (Seite {page_number})" if page_number else "")
            
            # Erstelle Tabellen-Chunk
            table_chunk = {
                'text': table_text,
                'section': table_section,
                'page_number': page_number,
                'length': len(table_text),
                'element_indices': [i],
                'is_table': True
            }
            # Füge alle Tabellen-Metadaten hinzu
            table_chunk.update(table_metadata)
            
            semantic_chunks.append(table_chunk)
            print(f"  [TABLE] Tabelle {tables_found} erkannt: {element_type} → {table_metadata['table_quality']} Qualität")
            
            # Reset für nächsten Chunk
            current_chunk = ""
            current_section = None
            current_page = page_number
            current_element_indices = []
            continue
        
        # Erkenne Abschnittswechsel basierend auf Header-Klassifikation
        header_level = classify_header_level(element_text, element_type)
        is_new_section = header_level in ['h1', 'h2']
        
        if is_new_section:
            # Speichere den vorherigen Chunk wenn er Inhalt hat
            if current_chunk.strip() and len(current_chunk.strip()) > 50:
                semantic_chunks.append({
                    'text': current_chunk.strip(),
                    'section': current_section or "Unnamed Section",
                    'page_number': current_page,
                    'length': len(current_chunk.strip()),
                    'element_indices': current_element_indices.copy(),
                    'is_table': False
                })
            
            # Starte neuen Chunk
            current_section = element_text
            current_page = page_number
            current_chunk = element_text + "\n\n"
            current_element_indices = [i]
        else:
            # Füge nur saubere Texte zum aktuellen Chunk hinzu
            import re
            cleaned = element_text.strip()
            # Ausschluss: sehr kurzer Text, hoher Sonderzeichenanteil, typische Fußnoten- oder Kopfzeilensignaturen
            non_alpha_ratio = sum(1 for c in cleaned if not c.isalnum() and c not in {'.', ',', ';', ':', '-', ' '}) / max(len(cleaned), 1)

            is_noise = (
                len(cleaned) < 5 or
                non_alpha_ratio > 0.5 or
                re.match(r'^\d+$', cleaned) is not None or   # reine Zahlen
                re.match(r'^Copyright', cleaned, flags=re.IGNORECASE) is not None
            )

            if cleaned and not is_noise:
                current_chunk += cleaned + " "
                current_element_indices.append(i)
                # Update page number if available
                if page_number and not current_page:
                    current_page = page_number
    
    # Letzten Chunk speichern
    if current_chunk.strip() and len(current_chunk.strip()) > 50:
        semantic_chunks.append({
            'text': current_chunk.strip(),
            'section': current_section or "Final Section",
            'page_number': current_page,
            'length': len(current_chunk.strip()),
            'element_indices': current_element_indices.copy(),
            'is_table': False
        })
    
    print(f"[SEMANTIC] {tables_found} Tabellen als separate Chunks erkannt")
    return semantic_chunks

def convert_semantic_chunks_to_documents(
    semantic_chunks: List[Dict[str, Any]], 
    header_mapping: Dict[int, Dict[str, Any]], 
    elements: list, 
    pdf_path: str
) -> List[Document]:
    """
    Konvertiert semantische Chunks zu LlamaIndex Documents mit Header-Metadaten.
    Behandelt Tabellen-Chunks mit speziellen Metadaten.
    
    Args:
        semantic_chunks: Liste von semantischen Chunk-Dictionaries
        header_mapping: Mapping von Element-Index zu Header-Metadaten
        elements: Ursprüngliche Unstructured-Elemente
        pdf_path: Pfad zur PDF-Datei
        
    Returns:
        Liste von Document-Objekten mit angereicherten Metadaten
    """
    documents = []
    filename = os.path.basename(pdf_path)
    
    for i, chunk in enumerate(semantic_chunks):
        # Basis-Metadaten
        metadata = {
            "file_path": pdf_path,
            "file_directory": os.path.dirname(pdf_path),
            "filename": filename,
            "section_title": chunk['section'],
            "chunk_index": i,
            "semantic_chunk": True
        }
        
        # Füge Seitennummer hinzu wenn verfügbar
        if chunk['page_number'] is not None:
            metadata["page_number"] = chunk['page_number']
        
        # **TABELLEN-SPEZIFISCHE METADATEN**
        if chunk.get('is_table', False):
            # Füge alle Tabellen-Metadaten hinzu
            for key, value in chunk.items():
                if key.startswith('table_') or key in ['is_table', 'content_type', 'extraction_error']:
                    # Nur einfache Datentypen für ChromaDB
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
            
            print(f"  [TABLE-DOC] Tabellen-Document erstellt: {chunk.get('table_rows', 0)}x{chunk.get('table_cols', 0)} - {chunk.get('table_quality', 'unknown')} Qualität")
        else:
            # Normale Chunks: Integriere hierarchische Header-Metadaten
            chunk_header_info = extract_header_info_for_chunk(
                chunk['element_indices'], 
                header_mapping, 
                elements
            )
            
            # Füge Header-Metadaten zu den Dokument-Metadaten hinzu
            metadata.update(chunk_header_info)
            metadata['content_type'] = 'text'
        
        document = Document(
            text=chunk['text'],
            metadata=metadata
        )
        documents.append(document)
    
    return documents

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

def print_semantic_analysis_summary(semantic_chunks: List[Dict[str, Any]], documents: List[Document]):
    """
    Gibt eine Zusammenfassung der semantischen Analyse aus.
    Inkludiert Tabellen-Statistiken.
    
    Args:
        semantic_chunks: Liste der semantischen Chunks
        documents: Liste der erstellten Documents
    """
    # Basis-Statistiken
    total_chunks = len(semantic_chunks)
    table_chunks = [chunk for chunk in semantic_chunks if chunk.get('is_table', False)]
    text_chunks = [chunk for chunk in semantic_chunks if not chunk.get('is_table', False)]
    
    print(f"[SEMANTIC] {total_chunks} semantische Abschnitte erstellt:")
    print(f"  - Text-Abschnitte: {len(text_chunks)}")
    print(f"  - Tabellen-Abschnitte: {len(table_chunks)}")
    
    # Text-Chunks
    for i, chunk in enumerate(text_chunks[:3]):  # Nur erste 3 anzeigen
        print(f"    Text {i+1}: '{chunk['section']}' ({chunk['length']} Zeichen)")
    if len(text_chunks) > 3:
        print(f"    ... und {len(text_chunks)-3} weitere Text-Abschnitte")
    
    # Tabellen-Chunks
    if table_chunks:
        print(f"  [TABELLEN] Erkannte Tabellen:")
        for i, table in enumerate(table_chunks):
            quality = table.get('table_quality', 'unknown')
            rows = table.get('table_rows', '?')
            cols = table.get('table_cols', '?')
            method = table.get('table_extraction_method', 'unknown')
            print(f"    Tabelle {i+1}: {rows}x{cols} Zellen - {quality} Qualität ({method})")
    
    # Header-Integration Statistiken
    header_enriched_chunks = [d for d in documents if d.metadata.get('contains_headers')]
    h1_headers = set(d.metadata.get('section_h1') for d in documents if d.metadata.get('section_h1'))
    h2_headers = set(d.metadata.get('section_h2') for d in documents if d.metadata.get('section_h2'))
    h3_headers = set(d.metadata.get('section_h3') for d in documents if d.metadata.get('section_h3'))
    
    print(f"[HEADER] Header-Integration erfolgreich:")
    print(f"  - Chunks mit Header-Metadaten: {len(header_enriched_chunks)}/{len(documents)}")
    print(f"  - H1-Überschriften gefunden: {len(h1_headers)}")
    print(f"  - H2-Überschriften gefunden: {len(h2_headers)}")
    print(f"  - H3-Überschriften gefunden: {len(h3_headers)}")

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

def enrich_document_metadata(document: Document, additional_metadata: Dict[str, Any]) -> Document:
    """
    Reichert ein Document mit zusätzlichen Metadaten an.
    
    Args:
        document: Ursprüngliches Document
        additional_metadata: Zusätzliche Metadaten
        
    Returns:
        Document mit angereicherten Metadaten
    """
    enhanced_metadata = document.metadata.copy() if document.metadata else {}
    enhanced_metadata.update(additional_metadata)
    
    return Document(
        text=document.text,
        metadata=enhanced_metadata
    )

def validate_document_quality(documents: List[Document]) -> Dict[str, Any]:
    """
    Validiert die Qualität der erstellten Dokumente.
    Inkludiert Tabellen-spezifische Qualitäts-Checks.
    
    Args:
        documents: Liste der zu validierenden Dokumente
        
    Returns:
        Dictionary mit Qualitäts-Metriken
    """
    if not documents:
        return {'valid': False, 'error': 'Keine Dokumente vorhanden'}
    
    # Basis-Metriken
    table_docs = [d for d in documents if d.metadata.get('is_table', False)]
    text_docs = [d for d in documents if not d.metadata.get('is_table', False)]
    
    quality_metrics = {
        'total_documents': len(documents),
        'text_documents': len(text_docs),
        'table_documents': len(table_docs),
        'avg_text_length': sum(len(doc.text) for doc in documents) / len(documents),
        'documents_with_pages': len([d for d in documents if d.metadata.get('page_number')]),
        'documents_with_headers': len([d for d in documents if d.metadata.get('contains_headers')]),
        'unique_sections': len(set(d.metadata.get('section_title') for d in documents if d.metadata.get('section_title'))),
        'empty_documents': len([d for d in documents if not d.text.strip()]),
        'valid': True
    }
    
    # **TABELLEN-SPEZIFISCHE METRIKEN**
    if table_docs:
        table_qualities = [d.metadata.get('table_quality', 'unknown') for d in table_docs]
        good_tables = len([q for q in table_qualities if q == 'good'])
        medium_tables = len([q for q in table_qualities if q == 'medium'])
        poor_tables = len([q for q in table_qualities if q == 'poor'])
        
        total_table_rows = sum(d.metadata.get('table_rows', 0) for d in table_docs)
        total_table_cols = sum(d.metadata.get('table_cols', 0) for d in table_docs)
        
        quality_metrics.update({
            'tables_good_quality': good_tables,
            'tables_medium_quality': medium_tables,
            'tables_poor_quality': poor_tables,
            'avg_table_rows': total_table_rows / len(table_docs) if table_docs else 0,
            'avg_table_cols': total_table_cols / len(table_docs) if table_docs else 0,
            'table_extraction_success_rate': (good_tables + medium_tables) / len(table_docs) if table_docs else 0
        })
    else:
        quality_metrics.update({
            'tables_good_quality': 0,
            'tables_medium_quality': 0,
            'tables_poor_quality': 0,
            'avg_table_rows': 0,
            'avg_table_cols': 0,
            'table_extraction_success_rate': 0
        })
    
    # Qualitätswarnungen
    warnings = []
    if quality_metrics['empty_documents'] > 0:
        warnings.append(f"{quality_metrics['empty_documents']} leere Dokumente gefunden")
    
    if quality_metrics['documents_with_pages'] / len(documents) < 0.5:
        warnings.append("Weniger als 50% der Dokumente haben Seitennummern")
    
    if quality_metrics['avg_text_length'] < 100:
        warnings.append("Durchschnittliche Textlänge sehr kurz (< 100 Zeichen)")
    
    # Tabellen-Warnungen
    if table_docs:
        if quality_metrics['table_extraction_success_rate'] < 0.7:
            warnings.append(f"Nur {quality_metrics['table_extraction_success_rate']:.1%} der Tabellen haben gute/mittlere Qualität")
        
        if quality_metrics['tables_poor_quality'] > 0:
            warnings.append(f"{quality_metrics['tables_poor_quality']} Tabellen mit schlechter Qualität gefunden")
    
    quality_metrics['warnings'] = warnings
    quality_metrics['quality_score'] = calculate_quality_score(quality_metrics)
    
    return quality_metrics

def calculate_quality_score(metrics: Dict[str, Any]) -> float:
    """
    Berechnet einen Qualitäts-Score für die Dokumente (0-100).
    Berücksichtigt auch Tabellen-Qualität.
    
    Args:
        metrics: Qualitäts-Metriken
        
    Returns:
        Qualitäts-Score zwischen 0 und 100
    """
    score = 100.0
    
    # Abzüge für Probleme
    if metrics['empty_documents'] > 0:
        score -= (metrics['empty_documents'] / metrics['total_documents']) * 20
    
    if metrics['documents_with_pages'] / metrics['total_documents'] < 0.5:
        score -= 15
    
    if metrics['avg_text_length'] < 100:
        score -= 10
    
    if metrics['unique_sections'] < 2:
        score -= 10
    
    # **TABELLEN-QUALITÄTS-BEWERTUNG**
    if metrics['table_documents'] > 0:
        table_success_rate = metrics['table_extraction_success_rate']
        if table_success_rate < 0.5:
            score -= 15  # Starker Abzug für schlechte Tabellen-Extraktion
        elif table_success_rate < 0.8:
            score -= 8   # Mittlerer Abzug
        # Bonus für gute Tabellen-Extraktion
        elif table_success_rate > 0.9:
            score += 5
    
    return max(0.0, score) 