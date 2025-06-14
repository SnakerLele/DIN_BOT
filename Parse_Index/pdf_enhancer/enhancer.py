"""
Hauptmodul für semantische PDF-Verbesserung

Dieses Modul orchestriert die semantische Verarbeitung von PDF-Dokumenten
durch Koordination aller Teilmodule:
- Priorisierte img2table Tabellen-Extraktion (Seiten als Bilder -> OCR -> Markdown)
- Header-Analyse
- Unstructured-Verarbeitung als Fallback für Text und missed Tables
- Semantische Abschnitts-Gruppierung
- Qualitäts-Validierung
"""

import os
from typing import List
from pathlib import Path
from llama_index.core import Document

from .header_utils import analyze_hierarchical_headers
from .semantic_sections import group_elements_into_semantic_sections, convert_semantic_chunks_to_documents
from .print_utils import print_semantic_analysis_summary
from .table_utils import extract_tables_with_img2table, create_table_nodes_from_img2table

def semantic_enhanced_pdf(pdf_path: str, use_img2table: bool = True) -> List[Document]:
    """
    Erstellt semantisch sinnvolle Dokument-Chunks durch intelligente Gruppierung.
    
    **NEUE PIPELINE MIT img2table-PRIORISIERUNG:**
    
    1. **img2table-Extraktion (PRIORITÄT):**
       - Konvertiert PDF-Seiten zu hochauflösenden Bildern
       - Nutzt OCR + OpenCV für robuste Tabellen-Erkennung
       - Extrahiert Tabellen direkt als Markdown mit detaillierten Metadaten
       - Behandelt als separate, hochwertige Chunks
    
    2. **Unstructured-Verarbeitung:**
       - Verarbeitet restlichen Text-Content
       - Fallback für übersehene Tabellen
       - Hierarchische Header-Analyse
    
    3. **Semantische Gruppierung:**
       - Kombiniert img2table-Tabellen mit Unstructured-Text
       - Intelligente Abschnitts-Gruppierung
       - Angereicherte Metadaten
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        use_img2table: Ob img2table-Extraktion verwendet werden soll (Standard: True)
        
    Returns:
        Liste von Document-Objekten mit img2table-Tabellen als hochwertige Chunks
    """
    print(f"[SEMANTIC] Verarbeite {pdf_path} mit img2table-priorisierter Pipeline")
    
    # **SCHRITT 1: img2table-Tabellen-Extraktion (PRIORITÄT)**
    img2table_documents = []
    if use_img2table:
        print(f"[IMG2TABLE] 🎯 Starte priorisierte img2table-Extraktion...")
        try:
            pdf_path_obj = Path(pdf_path)
            
            # img2table-Extraktion mit optimierten Parametern
            markdown_tables, table_metadata = extract_tables_with_img2table(
                pdf_path_obj,
                use_img2table=True,
                lang='deu+eng',  # Deutsch + Englisch
                dpi=300,         # Hohe Auflösung für bessere OCR
                borderless_tables=True,  # Tabellen ohne Rahmen erkennen
                implicit_rows=False,     # Konservative Einstellung
                implicit_columns=False   # Konservative Einstellung
            )
            
            if markdown_tables:
                print(f"[IMG2TABLE] ✅ {len(markdown_tables)} Tabellen erfolgreich extrahiert")
                
                # Konvertiere img2table-Ergebnisse zu Documents
                for i, (markdown, metadata) in enumerate(zip(markdown_tables, table_metadata)):
                    # Tabellen-Text mit Kontext-Header
                    table_text = f"**Tabelle {i+1}** (Seite {metadata.get('page_number', '?')}):\n\n{markdown}"
                    
                    # Erweiterte Metadaten für img2table-Tabellen
                    doc_metadata = {
                        "file_path": pdf_path,
                        "file_directory": os.path.dirname(pdf_path),
                        "filename": os.path.basename(pdf_path),
                        "section_title": f"img2table-Tabelle {i+1}",
                        "chunk_index": i,
                        "semantic_chunk": True,
                        "extraction_method": "img2table",
                        "extraction_priority": "primary",
                        "page_number": metadata.get('page_number'),
                        "table_id": metadata.get('table_id'),
                        "table_index": metadata.get('table_index'),
                        "table_rows": metadata.get('rows'),
                        "table_columns": metadata.get('columns'),
                        "table_quality": metadata.get('table_quality'),
                        "character_count": metadata.get('character_count'),
                        "is_table": True,
                        "content_type": "table",
                        "node_type": "img2table_table"
                    }
                    
                    # Bounding Box als String für ChromaDB-Kompatibilität
                    if metadata.get('bbox'):
                        bbox = metadata.get('bbox')
                        doc_metadata["bbox_coordinates"] = f"x1:{bbox.get('x1', 0):.1f},y1:{bbox.get('y1', 0):.1f},x2:{bbox.get('x2', 0):.1f},y2:{bbox.get('y2', 0):.1f}"
                    
                    # Erstelle img2table-Document
                    img2table_doc = Document(
                        text=table_text,
                        metadata=doc_metadata
                    )
                    img2table_documents.append(img2table_doc)
                    
                    print(f"  [IMG2TABLE] 📊 Tabelle {i+1}: {metadata.get('rows', 0)}x{metadata.get('columns', 0)} - {metadata.get('table_quality', 'unknown')} Qualität")
                
            else:
                print(f"[IMG2TABLE] ℹ️ Keine Tabellen mit img2table gefunden")
                
        except Exception as e:
            print(f"[IMG2TABLE] ⚠️ img2table-Extraktion fehlgeschlagen: {str(e)}")
            print(f"   Fallback zu unstructured-only Pipeline")
    else:
        print(f"[IMG2TABLE] ℹ️ img2table deaktiviert, verwende nur unstructured")
    
    # **SCHRITT 2: Unstructured-Verarbeitung (Text + Fallback-Tabellen)**
    print(f"[UNSTRUCTURED] 📄 Starte unstructured-Verarbeitung für Text-Content und Fallback-Tabellen...")
    
    try:
        from unstructured.partition.pdf import partition_pdf
        
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",  # Hohe Qualität für Layout-Erkennung
            include_page_breaks=True,
            include_metadata=True,
            combine_text_under_n_chars=0,
            infer_table_structure=True  # Fallback-Tabellen-Erkennung
        )
        
        print(f"[UNSTRUCTURED] ✅ {len(elements)} PDF-Elemente geladen (hi_res mit Fallback-Tabellen)")
        
    except Exception as e:
        print(f"❌ Fehler beim Laden der PDF-Elemente: {str(e)}")
        # Wenn unstructured komplett fehlschlägt, gib wenigstens img2table-Tabellen zurück
        if img2table_documents:
            print(f"⚠️ Fallback: Nur img2table-Tabellen zurückgeben ({len(img2table_documents)} Dokumente)")
            return img2table_documents
        return []
    
    # **SCHRITT 3: Hierarchische Header-Analyse**
    print(f"[SEMANTIC] 🔍 Starte hierarchische Header-Analyse...")
    header_mapping = analyze_hierarchical_headers(elements)
    
    # **SCHRITT 4: Semantische Abschnitts-Gruppierung**
    print(f"[SEMANTIC] 🧩 Gruppiere Elemente zu semantischen Abschnitten...")
    semantic_chunks = group_elements_into_semantic_sections(elements)
    
    # **SCHRITT 5: Konvertierung zu Documents**
    print(f"[SEMANTIC] 📋 Konvertiere zu LlamaIndex Documents...")
    unstructured_documents = convert_semantic_chunks_to_documents(
        semantic_chunks, 
        header_mapping, 
        elements, 
        pdf_path
    )
    
    # **SCHRITT 6: Kombiniere img2table + unstructured Documents**
    all_documents = img2table_documents + unstructured_documents
    
    print(f"[SEMANTIC] 🎯 Pipeline abgeschlossen:")
    print(f"   📊 img2table-Tabellen: {len(img2table_documents)} hochwertige Chunks")
    print(f"   📄 Unstructured-Chunks: {len(unstructured_documents)} semantische Abschnitte")
    print(f"   📋 Gesamt-Documents: {len(all_documents)}")
    
    # **SCHRITT 7: Zusammenfassung und Export**
    print_semantic_analysis_summary(semantic_chunks, all_documents, img2table_count=len(img2table_documents))
    
    # Export mit img2table-Informationen
    export_documents_to_txt(all_documents, pdf_path, img2table_count=len(img2table_documents))
    
    return all_documents


def export_documents_to_txt(documents: List[Document], pdf_path: str, img2table_count: int = 0) -> None:
    """
    Exportiert alle erstellten Dokumente mit Metadaten und Inhalt in eine TXT-Datei.
    
    **ERWEITERT für img2table-Integration:**
    - Trennt img2table-Tabellen von unstructured-Content
    - Zusätzliche Statistiken für img2table-Qualität
    - Verbesserte Formatierung für Tabellen-Chunks
    
    Args:
        documents: Liste der Document-Objekte
        pdf_path: Pfad zur ursprünglichen PDF-Datei
        img2table_count: Anzahl der img2table-Tabellen
    """
    try:
        # Pfad für TXT-Export erstellen
        pdf_file = Path(pdf_path)
        pdf_name = pdf_file.stem
        export_path = pdf_file.parent / f"{pdf_name}_enhanced_documents.txt"
        
        print(f"[EXPORT] 📤 Exportiere {len(documents)} Dokumente nach: {export_path.name}")
        
        with open(export_path, 'w', encoding='utf-8') as f:
            # Header für die Export-Datei
            f.write("=" * 80 + "\n")
            f.write(f"ENHANCED DOCUMENTS FROM: {pdf_file.name}\n")
            f.write(f"Pipeline: img2table-priorisiert + unstructured\n")
            f.write(f"Anzahl Dokumente: {len(documents)}\n")
            f.write(f"img2table-Tabellen: {img2table_count}\n")
            f.write(f"Unstructured-Chunks: {len(documents) - img2table_count}\n")
            f.write("=" * 80 + "\n\n")
            
            # Trennung: img2table-Tabellen zuerst
            img2table_docs = [doc for doc in documents if doc.metadata.get('extraction_method') == 'img2table']
            unstructured_docs = [doc for doc in documents if doc.metadata.get('extraction_method') != 'img2table']
            
            # img2table-Tabellen-Sektion
            if img2table_docs:
                f.write("🎯 IMG2TABLE-TABELLEN (PRIORITÄT)\n")
                f.write("=" * 50 + "\n\n")
                
                for i, doc in enumerate(img2table_docs, 1):
                    f.write(f"📊 IMG2TABLE-TABELLE {i:02d}/{len(img2table_docs):02d}\n")
                    f.write("-" * 60 + "\n")
                    
                    # Tabellen-spezifische Metadaten
                    f.write("🏷️  TABELLEN-METADATEN:\n")
                    key_meta = ['page_number', 'table_rows', 'table_columns', 'table_quality', 'character_count']
                    for key in key_meta:
                        if key in doc.metadata:
                            f.write(f"   {key}: {doc.metadata[key]}\n")
                    
                    f.write("\n📝 TABELLEN-INHALT:\n")
                    f.write("-" * 30 + "\n")
                    f.write(doc.text)
                    f.write("\n" + "=" * 60 + "\n\n")
            
            # Unstructured-Chunks-Sektion
            if unstructured_docs:
                f.write("📄 UNSTRUCTURED-CHUNKS (SEMANTISCHE ABSCHNITTE)\n")
                f.write("=" * 50 + "\n\n")
                
                for i, doc in enumerate(unstructured_docs, 1):
                    f.write(f"📋 UNSTRUCTURED-CHUNK {i:03d}/{len(unstructured_docs):03d}\n")
                    f.write("-" * 60 + "\n")
                    
                    # Standard-Metadaten
                    f.write("🏷️  METADATEN:\n")
                    for key, value in doc.metadata.items():
                        if isinstance(value, str) and len(value) > 100:
                            display_value = value[:100] + "... [gekürzt]"
                        else:
                            display_value = value
                        f.write(f"   {key}: {display_value}\n")
                    
                    f.write("\n📝 INHALT:\n")
                    f.write("-" * 30 + "\n")
                    f.write(doc.text)
                    f.write("\n" + "=" * 60 + "\n\n")
            
            # Erweiterte Zusammenfassung
            f.write("📋 ENHANCED EXPORT-ZUSAMMENFASSUNG\n")
            f.write("=" * 40 + "\n")
            
            # Pipeline-Statistiken
            f.write("Pipeline-Ergebnisse:\n")
            f.write(f"  🎯 img2table-Tabellen: {len(img2table_docs)} (Priorität)\n")
            f.write(f"  📄 Unstructured-Chunks: {len(unstructured_docs)} (Text + Fallback)\n")
            f.write(f"  📋 Gesamt-Documents: {len(documents)}\n\n")
            
            # Qualitäts-Statistiken für img2table-Tabellen
            if img2table_docs:
                quality_stats = {}
                for doc in img2table_docs:
                    quality = doc.metadata.get('table_quality', 'unknown')
                    quality_stats[quality] = quality_stats.get(quality, 0) + 1
                
                f.write("img2table Qualitäts-Verteilung:\n")
                for quality, count in quality_stats.items():
                    f.write(f"  {quality}: {count} Tabellen\n")
            
            # Gesamt-Statistiken
            total_chars = sum(len(doc.text) for doc in documents)
            total_words = sum(len(doc.text.split()) for doc in documents)
            
            f.write(f"\nGesamt-Statistiken:\n")
            f.write(f"  Zeichen: {total_chars:,}\n")
            f.write(f"  Wörter: {total_words:,}\n")
            f.write(f"  Durchschnitt Zeichen/Dokument: {total_chars // len(documents):,}\n")
            
        print(f"✅ Enhanced Export erfolgreich: {len(documents)} Dokumente → {export_path.name}")
        print(f"   🎯 img2table-Tabellen: {img2table_count}")
        print(f"   📄 Unstructured-Chunks: {len(documents) - img2table_count}")
        print(f"   📁 Gespeichert in: {export_path}")
        
    except Exception as e:
        print(f"❌ Export-Fehler: {str(e)}")
        import traceback
        print(f"   Details: {traceback.format_exc()}")