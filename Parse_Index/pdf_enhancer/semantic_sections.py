"""
Semantische Abschnitts-Verarbeitung für PDF-Dokumente

Dieses Modul enthält Funktionen zur:
- Gruppierung von PDF-Elementen zu semantischen Abschnitten
- Konvertierung zu LlamaIndex Documents
- Integration von Header- und Tabellen-Metadaten
"""

import os
from typing import List, Dict, Any
from llama_index.core import Document

from .header_utils import classify_header_level, extract_header_info_for_chunk
from .table_utils import convert_table_to_text_and_meta

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