"""
Qualitäts-Validierung für PDF-Dokumente

Dieses Modul enthält Funktionen zur:
- Dokumenten-Qualitäts-Validierung
- Qualitäts-Scoring und Metriken
- Tabellen-spezifische Qualitätsbewertung
- Warnung und Empfehlungen
"""

from typing import List, Dict, Any
from llama_index.core import Document

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