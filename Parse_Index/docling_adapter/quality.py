"""
QualityAnalyzer
---------------
Analysiert die Qualität von PDF-Dokumenten und Docling-Parsing-Ergebnissen.

Berechnet verschiedene Kennzahlen:
- OCR-Quote (Anteil der OCR-verarbeiteten Blöcke)
- Auflösungsqualität
- Strukturqualität (Überschriften, Tabellen erkannt)
- Gesamtqualitätsscore für RAG-Priorisierung
"""

from typing import Dict, Any
import logging


class QualityAnalyzer:
    """
    Analysiert die Qualität von PDF-Dokumenten nach dem Docling-Parsing.
    
    Liefert Kennzahlen, die für die RAG-Priorisierung und Debugging
    nützlich sind.
    """
    
    def __init__(self):
        """Initialisiert den QualityAnalyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, docling_doc: Any) -> Dict[str, Any]:
        """
        Hauptmethode: Analysiert ein Docling-Document und berechnet Qualitätskennzahlen.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            Dictionary mit Qualitätskennzahlen:
            - quality_score: Gesamtscore 0.0-1.0
            - ocr_blocks: Anzahl OCR-verarbeiteter Blöcke
            - total_blocks: Gesamtanzahl Blöcke
            - ocr_ratio: Verhältnis OCR/Gesamt
            - has_structure: Boolean, ob Struktur erkannt wurde
            - confidence: Vertrauenswert der Analyse
        """
        try:
            # Grundlegende Statistiken sammeln
            stats = self._collect_basic_stats(docling_doc)
            
            # OCR-Analyse
            ocr_stats = self._analyze_ocr_quality(docling_doc, stats)
            
            # Strukturanalyse
            structure_stats = self._analyze_structure_quality(docling_doc, stats)
            
            # Gesamtscore berechnen
            overall_score = self._calculate_overall_score(ocr_stats, structure_stats)
            
            # Ergebnisse zusammenfassen
            result = {
                **stats,
                **ocr_stats,
                **structure_stats,
                "quality_score": overall_score,
                "confidence": self._calculate_confidence(stats),
            }
            
            self.logger.debug(f"Qualitätsanalyse abgeschlossen: Score {overall_score:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei Qualitätsanalyse: {str(e)}")
            # Fallback: Neutrale Werte
            return {
                "quality_score": 0.5,
                "ocr_blocks": 0,
                "total_blocks": 0,
                "ocr_ratio": 0.0,
                "has_structure": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _collect_basic_stats(self, docling_doc: Any) -> Dict[str, Any]:
        """
        Sammelt grundlegende Statistiken über das Dokument.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            Dictionary mit Grundstatistiken
        """
        # TODO: Echte Docling-API verwenden
        # Für jetzt: Dummy-Implementierung
        
        if hasattr(docling_doc, 'pages'):
            page_count = len(docling_doc.pages)
            # Simuliere Blöcke pro Seite
            total_blocks = page_count * 5  # Durchschnittlich 5 Blöcke pro Seite
        else:
            page_count = 1
            total_blocks = 5
            
        return {
            "page_count": page_count,
            "total_blocks": total_blocks,
            "avg_blocks_per_page": total_blocks / max(1, page_count),
        }
    
    def _analyze_ocr_quality(self, docling_doc: Any, basic_stats: Dict) -> Dict[str, Any]:
        """
        Analysiert die OCR-Qualität des Dokuments.
        
        Args:
            docling_doc: Docling Document-Objekt
            basic_stats: Grundstatistiken
            
        Returns:
            Dictionary mit OCR-Statistiken
        """
        # TODO: Echte OCR-Analyse implementieren
        # Docling markiert OCR-Blöcke mit source="ocr"
        
        total_blocks = basic_stats["total_blocks"]
        
        # Dummy: Simuliere OCR-Erkennung
        # In echten Docling-Docs: [b for page in doc.pages for b in page.blocks if b.source == "ocr"]
        ocr_blocks = max(0, total_blocks // 4)  # 25% OCR-Blöcke als Beispiel
        
        ocr_ratio = ocr_blocks / max(1, total_blocks)
        
        # OCR-Qualitätsscore: Weniger OCR = bessere Qualität
        ocr_quality = 1.0 - min(1.0, ocr_ratio * 1.5)  # Penalty für hohe OCR-Quote
        
        return {
            "ocr_blocks": ocr_blocks,
            "ocr_ratio": round(ocr_ratio, 3),
            "ocr_quality": round(ocr_quality, 3),
        }
    
    def _analyze_structure_quality(self, docling_doc: Any, basic_stats: Dict) -> Dict[str, Any]:
        """
        Analysiert die Strukturqualität (Überschriften, Tabellen, etc.).
        
        Args:
            docling_doc: Docling Document-Objekt
            basic_stats: Grundstatistiken
            
        Returns:
            Dictionary mit Strukturstatistiken
        """
        # TODO: Echte Strukturanalyse implementieren
        # Docling erkennt: headings, tables, images, formulas, etc.
        
        total_blocks = basic_stats["total_blocks"]
        
        # Dummy: Simuliere Strukturerkennung
        headings = max(1, total_blocks // 10)  # ~10% Überschriften
        tables = max(0, total_blocks // 20)    # ~5% Tabellen
        images = max(0, total_blocks // 30)    # ~3% Bilder
        
        has_structure = headings > 0
        structure_diversity = len([x for x in [headings, tables, images] if x > 0])
        
        # Strukturqualität: Mehr erkannte Elemente = bessere Struktur
        structure_quality = min(1.0, (headings + tables * 2 + images) / max(1, total_blocks) * 10)
        
        return {
            "headings": headings,
            "tables": tables,
            "images": images,
            "has_structure": has_structure,
            "structure_diversity": structure_diversity,
            "structure_quality": round(structure_quality, 3),
        }
    
    def _calculate_overall_score(self, ocr_stats: Dict, structure_stats: Dict) -> float:
        """
        Berechnet den Gesamtqualitätsscore.
        
        Args:
            ocr_stats: OCR-Statistiken
            structure_stats: Strukturstatistiken
            
        Returns:
            Gesamtscore zwischen 0.0 und 1.0
        """
        # Gewichtete Kombination verschiedener Qualitätsfaktoren
        ocr_weight = 0.4
        structure_weight = 0.6
        
        ocr_score = ocr_stats.get("ocr_quality", 0.5)
        structure_score = structure_stats.get("structure_quality", 0.5)
        
        overall = (ocr_score * ocr_weight) + (structure_score * structure_weight)
        
        # Auf 3 Dezimalstellen runden
        return round(min(1.0, max(0.0, overall)), 3)
    
    def _calculate_confidence(self, basic_stats: Dict) -> float:
        """
        Berechnet das Vertrauen in die Qualitätsanalyse.
        
        Args:
            basic_stats: Grundstatistiken
            
        Returns:
            Vertrauenswert zwischen 0.0 und 1.0
        """
        # Mehr Seiten/Blöcke = höheres Vertrauen in die Analyse
        page_count = basic_stats.get("page_count", 1)
        total_blocks = basic_stats.get("total_blocks", 1)
        
        # Logarithmische Skalierung für Vertrauen
        import math
        confidence = min(1.0, math.log(total_blocks + 1) / math.log(50))  # Max bei ~50 Blöcken
        
        return round(confidence, 3)
    
    def needs_ocr(self, docling_doc: Any) -> bool:
        """
        Bestimmt, ob ein Dokument OCR benötigt.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            True wenn OCR empfohlen wird
        """
        stats = self.analyze(docling_doc)
        
        # OCR empfehlen wenn:
        # - Hohe OCR-Quote bereits vorhanden (schlechte Qualität)
        # - Wenig Struktur erkannt
        ocr_ratio = stats.get("ocr_ratio", 0.0)
        has_structure = stats.get("has_structure", False)
        
        return ocr_ratio > 0.5 or not has_structure
    
    def get_quality_summary(self, docling_doc: Any) -> str:
        """
        Erstellt eine menschenlesbare Zusammenfassung der Dokumentqualität.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            Textuelle Zusammenfassung
        """
        stats = self.analyze(docling_doc)
        
        score = stats.get("quality_score", 0.0)
        ocr_ratio = stats.get("ocr_ratio", 0.0)
        has_structure = stats.get("has_structure", False)
        
        if score >= 0.8:
            quality_desc = "Ausgezeichnet"
        elif score >= 0.6:
            quality_desc = "Gut"
        elif score >= 0.4:
            quality_desc = "Mittelmäßig"
        else:
            quality_desc = "Schlecht"
        
        summary = f"Qualität: {quality_desc} ({score:.1%})"
        
        if ocr_ratio > 0.3:
            summary += f", {ocr_ratio:.1%} OCR-Inhalt"
        
        if has_structure:
            summary += ", Struktur erkannt"
        else:
            summary += ", keine klare Struktur"
        
        return summary


# Hilfsfunktionen für externe Nutzung
def quick_quality_check(docling_doc: Any) -> float:
    """
    Schnelle Qualitätsprüfung ohne detaillierte Analyse.
    
    Args:
        docling_doc: Docling Document-Objekt
        
    Returns:
        Qualitätsscore zwischen 0.0 und 1.0
    """
    analyzer = QualityAnalyzer()
    result = analyzer.analyze(docling_doc)
    return result.get("quality_score", 0.5)


def is_high_quality(docling_doc: Any, threshold: float = 0.7) -> bool:
    """
    Prüft, ob ein Dokument als hochqualitativ eingestuft wird.
    
    Args:
        docling_doc: Docling Document-Objekt
        threshold: Mindestqualitätsscore (Standard: 0.7)
        
    Returns:
        True wenn Qualität über Schwellwert
    """
    score = quick_quality_check(docling_doc)
    return score >= threshold 