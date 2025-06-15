"""
QualityAnalyzer
---------------
Analysiert die Qualität von PDF-Dokumenten und Docling-Parsing-Ergebnissen.

Berechnet verschiedene Kennzahlen:
- OCR-Quote (Anteil der OCR-verarbeiteten Blöcke)
- Auflösungsqualität und DPI-Analyse
- Strukturqualität (Überschriften, Tabellen erkannt)
- Text-Qualität (Lesbarkeit, Vollständigkeit)
- Gesamtqualitätsscore für RAG-Priorisierung
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import math
import re
from dataclasses import dataclass, asdict
from collections import Counter


@dataclass
class QualityMetrics:
    """Strukturierte Repräsentation von Qualitäts-Metriken."""
    # Gesamt-Score
    quality_score: float = 0.5
    confidence: float = 0.0
    
    # Basis-Statistiken
    page_count: int = 0
    total_elements: int = 0
    total_text_length: int = 0
    avg_elements_per_page: float = 0.0
    
    # OCR-Analyse
    ocr_elements: int = 0
    ocr_ratio: float = 0.0
    ocr_quality_score: float = 0.5
    
    # Struktur-Analyse
    headings: int = 0
    tables: int = 0
    images: int = 0
    formulas: int = 0
    lists: int = 0
    has_structure: bool = False
    structure_diversity: int = 0
    structure_quality_score: float = 0.5
    
    # Text-Qualität
    avg_text_length_per_element: float = 0.0
    text_density_score: float = 0.5
    readability_score: float = 0.5
    
    # Technische Qualität
    has_reading_order: bool = False
    has_layout_analysis: bool = False
    resolution_quality_score: float = 0.5
    
    # Element-Details
    element_type_counts: Optional[Dict[str, int]] = None
    element_confidence_avg: float = 0.0
    
    # Fehler und Warnungen
    parsing_errors: int = 0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class QualityAnalyzer:
    """
    Analysiert die Qualität von PDF-Dokumenten nach dem Docling-Parsing.
    
    Liefert detaillierte Kennzahlen für RAG-Priorisierung und Debugging.
    """
    
    def __init__(self, cache_results: bool = True):
        """
        Initialisiert den QualityAnalyzer.
        
        Args:
            cache_results: Ob Analyse-Ergebnisse gecacht werden sollen
        """
        self.logger = logging.getLogger(__name__)
        self.cache_results = cache_results
        self._analysis_cache: Dict[str, QualityMetrics] = {}
        
        # Konfigurierbare Schwellwerte
        self.thresholds = {
            "min_text_length": 10,
            "high_ocr_ratio": 0.7,
            "low_structure_ratio": 0.05,
            "min_elements_per_page": 2,
            "high_quality_threshold": 0.8,
            "low_quality_threshold": 0.3,
        }
        
        # Gewichtungen für Gesamtscore
        self.weights = {
            "ocr_quality": 0.25,
            "structure_quality": 0.35,
            "text_quality": 0.25,
            "technical_quality": 0.15,
        }
    
    def analyze(self, docling_doc: Any, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Hauptmethode: Analysiert ein Docling-Document und berechnet Qualitätskennzahlen.
        
        Args:
            docling_doc: Docling Document-Objekt
            force_refresh: Cache ignorieren und neu analysieren
            
        Returns:
            Dictionary mit Qualitätskennzahlen
        """
        # Cache-Key generieren
        cache_key = self._generate_cache_key(docling_doc)
        
        # Cache prüfen
        if not force_refresh and self.cache_results and cache_key in self._analysis_cache:
            self.logger.debug("Verwende gecachte Qualitätsanalyse")
            return asdict(self._analysis_cache[cache_key])
        
        try:
            self.logger.debug("Starte Qualitätsanalyse")
            
            # Neue Analyse durchführen
            metrics = QualityMetrics()
            
            # 1. Basis-Statistiken sammeln
            self._collect_basic_stats(docling_doc, metrics)
            
            # 2. OCR-Qualität analysieren
            self._analyze_ocr_quality(docling_doc, metrics)
            
            # 3. Struktur-Qualität analysieren
            self._analyze_structure_quality(docling_doc, metrics)
            
            # 4. Text-Qualität analysieren
            self._analyze_text_quality(docling_doc, metrics)
            
            # 5. Technische Qualität analysieren
            self._analyze_technical_quality(docling_doc, metrics)
            
            # 6. Gesamtscore berechnen
            metrics.quality_score = self._calculate_overall_score(metrics)
            
            # 7. Confidence berechnen
            metrics.confidence = self._calculate_confidence(metrics)
            
            # 8. Validierung und Warnungen
            self._validate_and_warn(metrics)
            
            # Cache speichern
            if self.cache_results:
                self._analysis_cache[cache_key] = metrics
            
            self.logger.debug(f"Qualitätsanalyse abgeschlossen: Score {metrics.quality_score:.3f}")
            return asdict(metrics)
            
        except Exception as e:
            self.logger.error(f"Fehler bei Qualitätsanalyse: {str(e)}")
            # Fallback: Minimale Metriken
            fallback_metrics = QualityMetrics(
                quality_score=0.1,
                confidence=0.0,
                warnings=[f"Analyse-Fehler: {str(e)}"]
            )
            return asdict(fallback_metrics)
    
    def _collect_basic_stats(self, docling_doc: Any, metrics: QualityMetrics):
        """Sammelt grundlegende Statistiken über das Dokument."""
        try:
            if hasattr(docling_doc, 'pages') and docling_doc.pages:
                metrics.page_count = len(docling_doc.pages)
                
                total_elements = 0
                total_text_length = 0
                
                for page in docling_doc.pages:
                    if hasattr(page, 'elements'):
                        page_elements = len(page.elements)
                        total_elements += page_elements
                        
                        # Text-Länge sammeln
                        for element in page.elements:
                            element_text = getattr(element, 'text', '')
                            if element_text:
                                total_text_length += len(element_text)
                
                metrics.total_elements = total_elements
                metrics.total_text_length = total_text_length
                metrics.avg_elements_per_page = total_elements / max(1, metrics.page_count)
                
                if total_elements > 0:
                    metrics.avg_text_length_per_element = total_text_length / total_elements
            else:
                # Fallback für Dummy-Dokumente
                metrics.page_count = 1
                metrics.total_elements = 1
                metrics.total_text_length = 100
                metrics.avg_elements_per_page = 1.0
                metrics.avg_text_length_per_element = 100.0
                metrics.warnings.append("Keine echten Docling-Seiten gefunden")
                
        except Exception as e:
            self.logger.warning(f"Fehler bei Basis-Statistiken: {e}")
            metrics.warnings.append(f"Basis-Statistik-Fehler: {str(e)}")
    
    def _analyze_ocr_quality(self, docling_doc: Any, metrics: QualityMetrics):
        """Analysiert die OCR-Qualität basierend auf echten Docling-Flags."""
        try:
            if not hasattr(docling_doc, 'pages') or not docling_doc.pages:
                metrics.ocr_ratio = 0.0
                metrics.ocr_quality_score = 0.8  # Annahme: kein OCR = gut
                return
            
            ocr_elements = 0
            total_elements = 0
            confidence_scores = []
            
            for page in docling_doc.pages:
                if hasattr(page, 'elements'):
                    for element in page.elements:
                        total_elements += 1
                        
                        # OCR-Erkennung über verschiedene Attribute
                        is_ocr = False
                        
                        # Methode 1: source-Attribut
                        if hasattr(element, 'source') and element.source == 'ocr':
                            is_ocr = True
                        
                        # Methode 2: prov-Attribut (Docling-spezifisch)
                        elif hasattr(element, 'prov') and 'ocr' in str(element.prov).lower():
                            is_ocr = True
                        
                        # Methode 3: confidence-basierte Heuristik
                        elif hasattr(element, 'confidence'):
                            confidence = float(element.confidence)
                            confidence_scores.append(confidence)
                            # Niedrige Confidence deutet auf OCR hin
                            if confidence < 0.8:
                                is_ocr = True
                        
                        # Methode 4: Text-Qualitäts-Heuristik
                        elif hasattr(element, 'text'):
                            text = element.text
                            if self._is_likely_ocr_text(text):
                                is_ocr = True
                        
                        if is_ocr:
                            ocr_elements += 1
            
            metrics.ocr_elements = ocr_elements
            metrics.ocr_ratio = ocr_elements / max(1, total_elements)
            
            # OCR-Qualitätsscore berechnen
            if metrics.ocr_ratio == 0:
                metrics.ocr_quality_score = 1.0  # Kein OCR = perfekt
            elif metrics.ocr_ratio < 0.3:
                metrics.ocr_quality_score = 0.8  # Wenig OCR = gut
            elif metrics.ocr_ratio < 0.7:
                metrics.ocr_quality_score = 0.5  # Mittlerer OCR-Anteil
            else:
                metrics.ocr_quality_score = 0.2  # Viel OCR = schlecht
            
            # Confidence-basierte Anpassung
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                metrics.element_confidence_avg = avg_confidence
                # Niedrige durchschnittliche Confidence verschlechtert Score
                if avg_confidence < 0.7:
                    metrics.ocr_quality_score *= 0.8
            
            # Warnungen
            if metrics.ocr_ratio > self.thresholds["high_ocr_ratio"]:
                metrics.warnings.append(f"Hoher OCR-Anteil: {metrics.ocr_ratio:.1%}")
                
        except Exception as e:
            self.logger.warning(f"Fehler bei OCR-Analyse: {e}")
            metrics.warnings.append(f"OCR-Analyse-Fehler: {str(e)}")
            metrics.ocr_quality_score = 0.5  # Neutral bei Fehlern
    
    def _analyze_structure_quality(self, docling_doc: Any, metrics: QualityMetrics):
        """Analysiert die Strukturqualität basierend auf erkannten Elementen."""
        try:
            if not hasattr(docling_doc, 'pages') or not docling_doc.pages:
                metrics.structure_quality_score = 0.1
                return
            
            element_counts = Counter()
            
            for page in docling_doc.pages:
                if hasattr(page, 'elements'):
                    for element in page.elements:
                        element_type = getattr(element, 'type', 'unknown')
                        element_counts[element_type] += 1
            
            # Spezifische Element-Typen zählen
            heading_types = ['heading', 'title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
            table_types = ['table', 'table_cell', 'table_row']
            image_types = ['image', 'figure', 'picture', 'photo']
            formula_types = ['formula', 'equation', 'math']
            list_types = ['list', 'list_item', 'bullet_list', 'numbered_list']
            
            metrics.headings = sum(element_counts.get(t, 0) for t in heading_types)
            metrics.tables = sum(element_counts.get(t, 0) for t in table_types)
            metrics.images = sum(element_counts.get(t, 0) for t in image_types)
            metrics.formulas = sum(element_counts.get(t, 0) for t in formula_types)
            metrics.lists = sum(element_counts.get(t, 0) for t in list_types)
            
            # Element-Type-Counts speichern
            metrics.element_type_counts = dict(element_counts)
            
            # Struktur-Indikatoren
            metrics.has_structure = metrics.headings > 0
            metrics.structure_diversity = len([x for x in [metrics.headings, metrics.tables, 
                                                         metrics.images, metrics.formulas, 
                                                         metrics.lists] if x > 0])
            
            # Struktur-Qualitätsscore berechnen
            total_elements = metrics.total_elements
            if total_elements == 0:
                metrics.structure_quality_score = 0.0
                return
            
            score = 0.0
            
            # Überschriften (wichtig für Navigation)
            heading_ratio = metrics.headings / total_elements
            if heading_ratio > 0:
                score += min(0.4, heading_ratio * 5)  # Max 40% für Überschriften
            
            # Tabellen (strukturierte Daten)
            table_ratio = metrics.tables / total_elements
            if table_ratio > 0:
                score += min(0.3, table_ratio * 10)  # Max 30% für Tabellen
            
            # Listen (strukturierte Aufzählungen)
            list_ratio = metrics.lists / total_elements
            if list_ratio > 0:
                score += min(0.2, list_ratio * 8)  # Max 20% für Listen
            
            # Vielfalt der Element-Typen
            if metrics.structure_diversity > 2:
                score += min(0.1, (metrics.structure_diversity - 2) * 0.03)
            
            metrics.structure_quality_score = min(1.0, score)
            
            # Warnungen
            if not metrics.has_structure:
                metrics.warnings.append("Keine Überschriften-Struktur erkannt")
            
            structure_ratio = (metrics.headings + metrics.tables + metrics.lists) / total_elements
            if structure_ratio < self.thresholds["low_structure_ratio"]:
                metrics.warnings.append(f"Niedrige Strukturqualität: {structure_ratio:.1%}")
                
        except Exception as e:
            self.logger.warning(f"Fehler bei Struktur-Analyse: {e}")
            metrics.warnings.append(f"Struktur-Analyse-Fehler: {str(e)}")
            metrics.structure_quality_score = 0.5
    
    def _analyze_text_quality(self, docling_doc: Any, metrics: QualityMetrics):
        """Analysiert die Text-Qualität (Lesbarkeit, Vollständigkeit)."""
        try:
            if metrics.total_text_length == 0:
                metrics.text_density_score = 0.0
                metrics.readability_score = 0.0
                return
            
            # Text-Dichte bewerten
            if metrics.total_elements > 0:
                avg_text_per_element = metrics.total_text_length / metrics.total_elements
                
                # Optimale Text-Länge pro Element: 50-500 Zeichen
                if 50 <= avg_text_per_element <= 500:
                    metrics.text_density_score = 1.0
                elif avg_text_per_element < 50:
                    # Zu kurze Texte (möglicherweise fragmentiert)
                    metrics.text_density_score = avg_text_per_element / 50
                else:
                    # Zu lange Texte (möglicherweise nicht segmentiert)
                    metrics.text_density_score = max(0.3, 500 / avg_text_per_element)
            
            # Lesbarkeits-Analyse (vereinfacht)
            if hasattr(docling_doc, 'pages') and docling_doc.pages:
                text_samples = []
                
                for page in docling_doc.pages[:3]:  # Nur erste 3 Seiten analysieren
                    if hasattr(page, 'elements'):
                        for element in page.elements:
                            element_text = getattr(element, 'text', '')
                            if len(element_text) > self.thresholds["min_text_length"]:
                                text_samples.append(element_text)
                
                if text_samples:
                    metrics.readability_score = self._calculate_readability_score(text_samples)
                else:
                    metrics.readability_score = 0.1
                    metrics.warnings.append("Keine ausreichenden Text-Samples für Lesbarkeits-Analyse")
            
            # Warnungen
            if metrics.avg_text_length_per_element < self.thresholds["min_text_length"]:
                metrics.warnings.append(f"Sehr kurze Texte: {metrics.avg_text_length_per_element:.1f} Zeichen/Element")
                
        except Exception as e:
            self.logger.warning(f"Fehler bei Text-Qualitäts-Analyse: {e}")
            metrics.warnings.append(f"Text-Qualitäts-Fehler: {str(e)}")
            metrics.text_density_score = 0.5
            metrics.readability_score = 0.5
    
    def _analyze_technical_quality(self, docling_doc: Any, metrics: QualityMetrics):
        """Analysiert technische Qualitäts-Aspekte."""
        try:
            score = 0.0
            
            # Reading Order verfügbar?
            if hasattr(docling_doc, 'reading_order') or hasattr(docling_doc, 'layout'):
                metrics.has_reading_order = True
                score += 0.3
            
            # Layout-Analyse verfügbar?
            if hasattr(docling_doc, 'layout') or hasattr(docling_doc, 'page_layout'):
                metrics.has_layout_analysis = True
                score += 0.3
            
            # Auflösungs-/DPI-Analyse (wenn verfügbar)
            resolution_scores = []
            if hasattr(docling_doc, 'pages'):
                for page in docling_doc.pages:
                    # DPI-Information suchen
                    if hasattr(page, 'dpi'):
                        dpi = float(page.dpi)
                        if dpi >= 300:
                            resolution_scores.append(1.0)
                        elif dpi >= 150:
                            resolution_scores.append(0.7)
                        else:
                            resolution_scores.append(0.3)
                    
                    # Bild-Qualität über Dimensionen schätzen
                    elif hasattr(page, 'width') and hasattr(page, 'height'):
                        width, height = page.width, page.height
                        # Heuristik: Größere Dimensionen = bessere Qualität
                        if width * height > 2000000:  # > 2MP
                            resolution_scores.append(0.8)
                        elif width * height > 500000:  # > 0.5MP
                            resolution_scores.append(0.6)
                        else:
                            resolution_scores.append(0.4)
            
            if resolution_scores:
                metrics.resolution_quality_score = sum(resolution_scores) / len(resolution_scores)
                score += 0.4 * metrics.resolution_quality_score
            else:
                metrics.resolution_quality_score = 0.5  # Neutral wenn unbekannt
                score += 0.2
            
            metrics.technical_quality_score = min(1.0, score)
            
        except Exception as e:
            self.logger.warning(f"Fehler bei technischer Qualitäts-Analyse: {e}")
            metrics.warnings.append(f"Technische Analyse-Fehler: {str(e)}")
            metrics.resolution_quality_score = 0.5
    
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Berechnet den gewichteten Gesamtqualitätsscore."""
        try:
            # Gewichtete Kombination der Teil-Scores
            overall = (
                metrics.ocr_quality_score * self.weights["ocr_quality"] +
                metrics.structure_quality_score * self.weights["structure_quality"] +
                metrics.text_density_score * self.weights["text_quality"] +
                getattr(metrics, 'technical_quality_score', 0.5) * self.weights["technical_quality"]
            )
            
            # Penalty für kritische Probleme
            if metrics.total_elements < self.thresholds["min_elements_per_page"] * metrics.page_count:
                overall *= 0.8  # 20% Penalty für zu wenige Elemente
            
            if len(metrics.warnings) > 3:
                overall *= 0.9  # 10% Penalty für viele Warnungen
            
            return round(min(1.0, max(0.0, overall)), 3)
            
        except Exception as e:
            self.logger.warning(f"Fehler bei Gesamtscore-Berechnung: {e}")
            return 0.5
    
    def _calculate_confidence(self, metrics: QualityMetrics) -> float:
        """Berechnet das Vertrauen in die Qualitätsanalyse."""
        try:
            confidence = 0.0
            
            # Basis-Confidence für erfolgreiche Analyse
            confidence += 0.2
            
            # Mehr Elemente = höheres Vertrauen
            if metrics.total_elements > 10:
                confidence += min(0.3, math.log(metrics.total_elements) / math.log(100))
            
            # Mehr Seiten = höheres Vertrauen
            if metrics.page_count > 1:
                confidence += min(0.2, math.log(metrics.page_count) / math.log(20))
            
            # Strukturelle Vielfalt erhöht Vertrauen
            if metrics.structure_diversity > 2:
                confidence += 0.1
            
            # Confidence-Werte von Elementen berücksichtigen
            if metrics.element_confidence_avg > 0:
                confidence += 0.2 * metrics.element_confidence_avg
            
            # Penalty für viele Warnungen
            if len(metrics.warnings) > 2:
                confidence *= 0.8
            
            return round(min(1.0, max(0.0, confidence)), 3)
            
        except Exception as e:
            self.logger.warning(f"Fehler bei Confidence-Berechnung: {e}")
            return 0.5
    
    def _validate_and_warn(self, metrics: QualityMetrics):
        """Validiert Metriken und fügt Warnungen hinzu."""
        # Konsistenz-Checks
        if metrics.total_elements == 0 and metrics.page_count > 0:
            metrics.warnings.append("Keine Elemente trotz vorhandener Seiten")
        
        if metrics.ocr_ratio > 0.9:
            metrics.warnings.append("Dokument fast vollständig OCR-basiert")
        
        if metrics.quality_score < self.thresholds["low_quality_threshold"]:
            metrics.warnings.append(f"Niedrige Gesamtqualität: {metrics.quality_score:.2f}")
        
        # Duplikate entfernen
        metrics.warnings = list(set(metrics.warnings))
    
    def _is_likely_ocr_text(self, text: str) -> bool:
        """Heuristik zur OCR-Text-Erkennung."""
        if not text or len(text) < 10:
            return False
        
        # Häufige OCR-Artefakte
        ocr_indicators = [
            r'[Il1|]{3,}',  # Verwechslung von I, l, 1, |
            r'\b[A-Z]{1}[a-z]{1}[A-Z]{1}',  # Ungewöhnliche Groß-/Kleinschreibung
            r'[^\w\s]{3,}',  # Viele Sonderzeichen hintereinander
            r'\b\w{1}\s\w{1}\s\w{1}\b',  # Einzelne Buchstaben mit Leerzeichen
        ]
        
        ocr_score = 0
        for pattern in ocr_indicators:
            if re.search(pattern, text):
                ocr_score += 1
        
        return ocr_score >= 2
    
    def _calculate_readability_score(self, text_samples: List[str]) -> float:
        """Vereinfachte Lesbarkeits-Analyse."""
        try:
            if not text_samples:
                return 0.0
            
            total_score = 0.0
            
            for text in text_samples[:10]:  # Max 10 Samples
                score = 0.5  # Basis-Score
                
                # Satzlänge bewerten
                sentences = re.split(r'[.!?]+', text)
                if sentences:
                    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                    if 10 <= avg_sentence_length <= 25:  # Optimale Satzlänge
                        score += 0.2
                    elif avg_sentence_length < 5 or avg_sentence_length > 40:
                        score -= 0.1
                
                # Wort-Komplexität (vereinfacht)
                words = text.split()
                if words:
                    avg_word_length = sum(len(w) for w in words) / len(words)
                    if 4 <= avg_word_length <= 8:  # Optimale Wortlänge
                        score += 0.2
                    elif avg_word_length > 12:
                        score -= 0.1
                
                # Interpunktion vorhanden?
                if re.search(r'[.!?,:;]', text):
                    score += 0.1
                
                total_score += score
            
            return min(1.0, max(0.0, total_score / len(text_samples)))
            
        except Exception as e:
            self.logger.warning(f"Fehler bei Lesbarkeits-Analyse: {e}")
            return 0.5
    
    def _generate_cache_key(self, docling_doc: Any) -> str:
        """Generiert einen Cache-Key für das Dokument."""
        try:
            # Basis-Informationen für Cache-Key
            key_parts = []
            
            if hasattr(docling_doc, 'source_path'):
                key_parts.append(str(docling_doc.source_path))
            
            if hasattr(docling_doc, 'pages'):
                key_parts.append(f"pages_{len(docling_doc.pages)}")
            
            # Hash über ersten Text-Inhalt
            if hasattr(docling_doc, 'pages') and docling_doc.pages:
                first_page = docling_doc.pages[0]
                if hasattr(first_page, 'elements') and first_page.elements:
                    first_text = getattr(first_page.elements[0], 'text', '')
                    if first_text:
                        key_parts.append(f"text_{hash(first_text[:100])}")
            
            return "_".join(key_parts) if key_parts else "unknown_doc"
            
        except Exception:
            return "cache_error"
    
    # Public Interface Methods
    
    def needs_ocr(self, docling_doc: Any) -> bool:
        """
        Bestimmt, ob ein Dokument OCR benötigt.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            True wenn OCR empfohlen wird
        """
        try:
            analysis = self.analyze(docling_doc)
            
            # OCR empfehlen wenn:
            ocr_ratio = analysis.get("ocr_ratio", 0.0)
            ocr_quality = analysis.get("ocr_quality_score", 1.0)
            structure_quality = analysis.get("structure_quality_score", 1.0)
            
            return (ocr_ratio > 0.5 or 
                    ocr_quality < 0.5 or 
                    structure_quality < 0.3)
                    
        except Exception as e:
            self.logger.error(f"Fehler bei OCR-Empfehlung: {e}")
            return False
    
    def get_quality_summary(self, docling_doc: Any) -> str:
        """
        Erstellt eine menschenlesbare Zusammenfassung der Dokumentqualität.
        
        Args:
            docling_doc: Docling Document-Objekt
            
        Returns:
            Textuelle Zusammenfassung
        """
        try:
            analysis = self.analyze(docling_doc)
            
            score = analysis.get("quality_score", 0.0)
            ocr_ratio = analysis.get("ocr_ratio", 0.0)
            has_structure = analysis.get("has_structure", False)
            warnings = analysis.get("warnings", [])
            
            # Qualitäts-Kategorie
            if score >= self.thresholds["high_quality_threshold"]:
                quality_desc = "Ausgezeichnet"
            elif score >= 0.6:
                quality_desc = "Gut"
            elif score >= 0.4:
                quality_desc = "Mittelmäßig"
            else:
                quality_desc = "Schlecht"
            
            summary = f"Qualität: {quality_desc} ({score:.1%})"
            
            # Zusätzliche Informationen
            if ocr_ratio > 0.1:
                summary += f", {ocr_ratio:.1%} OCR-Inhalt"
            
            if has_structure:
                summary += ", Struktur erkannt"
            else:
                summary += ", keine klare Struktur"
            
            if warnings:
                summary += f", {len(warnings)} Warnung(en)"
            
            return summary
            
        except Exception as e:
            return f"Fehler bei Qualitäts-Zusammenfassung: {str(e)}"
    
    def is_high_quality(self, docling_doc: Any, threshold: Optional[float] = None) -> bool:
        """
        Prüft, ob ein Dokument als hochqualitativ eingestuft wird.
        
        Args:
            docling_doc: Docling Document-Objekt
            threshold: Mindestqualitätsscore (Standard aus Konfiguration)
            
        Returns:
            True wenn Qualität über Schwellwert
        """
        if threshold is None:
            threshold = self.thresholds["high_quality_threshold"]
        
        try:
            analysis = self.analyze(docling_doc)
            return analysis.get("quality_score", 0.0) >= threshold
        except Exception as e:
            self.logger.error(f"Fehler bei Qualitätsprüfung: {e}")
            return False
    
    def clear_cache(self):
        """Leert den Analyse-Cache."""
        self._analysis_cache.clear()
        self.logger.debug("Qualitäts-Analyse-Cache geleert")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Gibt Cache-Statistiken zurück."""
        return {
            "cached_analyses": len(self._analysis_cache),
            "cache_enabled": self.cache_results
        }


# Hilfsfunktionen für externe Nutzung
def quick_quality_check(docling_doc: Any) -> float:
    """
    Schnelle Qualitätsprüfung ohne detaillierte Analyse.
    
    Args:
        docling_doc: Docling Document-Objekt
        
    Returns:
        Qualitätsscore zwischen 0.0 und 1.0
    """
    analyzer = QualityAnalyzer(cache_results=False)
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
    analyzer = QualityAnalyzer(cache_results=False)
    return analyzer.is_high_quality(docling_doc, threshold)


def analyze_quality_batch(docling_docs: List[Any]) -> List[Dict[str, Any]]:
    """
    Analysiert die Qualität mehrerer Dokumente in einem Batch.
    
    Args:
        docling_docs: Liste von Docling Document-Objekten
        
    Returns:
        Liste von Qualitäts-Analysen
    """
    analyzer = QualityAnalyzer(cache_results=True)
    results = []
    
    for doc in docling_docs:
        try:
            analysis = analyzer.analyze(doc)
            results.append(analysis)
        except Exception as e:
            results.append({
                "quality_score": 0.1,
                "error": str(e)
            })
    
    return results