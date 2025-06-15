"""
Debug-Test für Metadaten-Extraktion
===================================

Dieser Test führt eine detaillierte Analyse der Metadaten-Extraktion durch,
um Fehler zu identifizieren und die Qualität der Extraktion zu bewerten.
"""

import sys
from pathlib import Path
import logging
import traceback
from typing import Dict, Any, List
import json

# Projekt-Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent))

from Parse_Index.docling_adapter import DoclingAdapter, MetadataExtractor
from Parse_Index.docling_adapter.metadata import DocumentMetadata


def setup_detailed_logging():
    """Konfiguriert detailliertes Logging für Debug-Zwecke."""
    # Logs-Verzeichnis erstellen
    log_dir = Path("Test_Skripts/logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / 'metadata_debug.log', mode='w', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


def test_metadata_extraction_single_pdf(pdf_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Testet die Metadaten-Extraktion für eine einzelne PDF-Datei.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        logger: Logger-Instanz
        
    Returns:
        Test-Ergebnisse
    """
    logger.info(f"🔍 TESTE METADATEN-EXTRAKTION: {pdf_path.name}")
    
    results = {
        "pdf_path": str(pdf_path),
        "filename": pdf_path.name,
        "file_exists": pdf_path.exists(),
        "file_size_mb": round(pdf_path.stat().st_size / (1024*1024), 2) if pdf_path.exists() else 0,
        "errors": [],
        "warnings": [],
        "metadata": {},
        "extraction_stages": {},
        "success": False
    }
    
    if not pdf_path.exists():
        results["errors"].append(f"PDF-Datei existiert nicht: {pdf_path}")
        return results
    
    try:
        # === SCHRITT 1: Minimale Metadaten-Extraktion ===
        logger.info("📋 Schritt 1: Minimale Metadaten-Extraktion (ohne Docling)")
        
        try:
            extractor = MetadataExtractor()
            minimal_metadata = extractor.extract_minimal(pdf_path)
            results["extraction_stages"]["minimal"] = {
                "success": True,
                "data": minimal_metadata,
                "field_count": len(minimal_metadata)
            }
            logger.info(f"✅ Minimale Metadaten extrahiert: {len(minimal_metadata)} Felder")
            
        except Exception as e:
            error_msg = f"Fehler beim minimalen Metadaten-Extraktion: {str(e)}"
            results["errors"].append(error_msg)
            results["extraction_stages"]["minimal"] = {
                "success": False,
                "error": error_msg,
                "traceback": traceback.format_exc()
            }
            logger.error(error_msg)
        
        # === SCHRITT 2: Docling-Adapter initialisieren ===
        logger.info("🔧 Schritt 2: Docling-Adapter initialisieren")
        
        try:
            adapter = DoclingAdapter(config={
                "timeout_seconds": 60,
                "ocr_enabled": False,  # Schneller für Tests
                "extract_metadata": True,
                "log_level": "DEBUG"
            })
            
            if not adapter.is_docling_available():
                results["warnings"].append("Docling nicht verfügbar - verwende Dummy-Modus")
                logger.warning("⚠️ Docling nicht verfügbar")
            
            results["extraction_stages"]["adapter_init"] = {
                "success": True,
                "docling_available": adapter.is_docling_available()
            }
            
        except Exception as e:
            error_msg = f"Fehler beim Initialisieren des Docling-Adapters: {str(e)}"
            results["errors"].append(error_msg)
            results["extraction_stages"]["adapter_init"] = {
                "success": False,
                "error": error_msg,
                "traceback": traceback.format_exc()
            }
            logger.error(error_msg)
            return results
        
        # === SCHRITT 3: PDF parsen ===
        logger.info("📄 Schritt 3: PDF mit Docling parsen")
        
        try:
            # Verwende strict_mode=False für robustes Parsing
            documents = adapter.parse_pdf(pdf_path, strict_mode=False)
            
            results["extraction_stages"]["pdf_parsing"] = {
                "success": True,
                "document_count": len(documents),
                "documents_created": len(documents) > 0
            }
            
            logger.info(f"✅ PDF erfolgreich geparst: {len(documents)} Dokumente erstellt")
            
            # Sammle Metadaten aus den erstellten Dokumenten
            if documents:
                sample_doc = documents[0]
                doc_metadata = sample_doc.metadata
                
                # Analysiere Metadaten-Struktur
                metadata_analysis = analyze_metadata_structure(doc_metadata, logger)
                results["extraction_stages"]["metadata_analysis"] = metadata_analysis
                
                # Speichere Metadaten für weitere Analyse
                results["metadata"] = doc_metadata
                
                # Prüfe auf spezifische Fehler
                if "extraction_error" in doc_metadata:
                    results["errors"].append(f"Metadaten-Extraktion-Fehler: {doc_metadata['extraction_error']}")
                
                # Bewerte Metadaten-Qualität
                quality_score = evaluate_metadata_quality(doc_metadata, logger)
                results["metadata_quality"] = quality_score
                
                results["success"] = True
                
            else:
                results["warnings"].append("Keine Dokumente erstellt")
                logger.warning("⚠️ Keine Dokumente erstellt")
                
        except Exception as e:
            error_msg = f"Fehler beim PDF-Parsing: {str(e)}"
            results["errors"].append(error_msg)
            results["extraction_stages"]["pdf_parsing"] = {
                "success": False,
                "error": error_msg,
                "traceback": traceback.format_exc()
            }
            logger.error(error_msg)
            
        # === SCHRITT 4: Docling-Objekt direkt testen ===
        logger.info("🔬 Schritt 4: Docling-Objekt direkt testen")
        
        if adapter.is_docling_available():
            try:
                # Direkter Zugriff auf Docling-Parsing
                docling_doc = adapter._parse_with_docling(pdf_path, strict_mode=False)
                
                if docling_doc:
                    # Teste direkte Metadaten-Extraktion
                    direct_metadata = extractor.extract(docling_doc, pdf_path)
                    
                    results["extraction_stages"]["direct_docling"] = {
                        "success": True,
                        "metadata": direct_metadata,
                        "field_count": len(direct_metadata)
                    }
                    
                    logger.info(f"✅ Direkte Docling-Metadaten extrahiert: {len(direct_metadata)} Felder")
                    
                else:
                    results["warnings"].append("Docling-Parsing lieferte None zurück")
                    
            except Exception as e:
                error_msg = f"Fehler beim direkten Docling-Test: {str(e)}"
                results["errors"].append(error_msg)
                results["extraction_stages"]["direct_docling"] = {
                    "success": False,
                    "error": error_msg,
                    "traceback": traceback.format_exc()
                }
                logger.error(error_msg)
        
        # === SCHRITT 5: Typ-Validierung ===
        logger.info("🔍 Schritt 5: Typ-Validierung der Metadaten")
        
        try:
            type_validation = validate_metadata_types(results.get("metadata", {}), logger)
            results["type_validation"] = type_validation
            
        except Exception as e:
            error_msg = f"Fehler bei Typ-Validierung: {str(e)}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
            
    except Exception as e:
        error_msg = f"Unerwarteter Fehler: {str(e)}"
        results["errors"].append(error_msg)
        logger.error(f"💥 {error_msg}")
        logger.error(traceback.format_exc())
    
    return results


def analyze_metadata_structure(metadata: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Analysiert die Struktur der extrahierten Metadaten.
    
    Args:
        metadata: Metadaten-Dictionary
        logger: Logger-Instanz
        
    Returns:
        Analyse-Ergebnisse
    """
    analysis = {
        "total_fields": len(metadata),
        "field_types": {},
        "empty_fields": [],
        "problematic_fields": [],
        "core_fields": {
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "filename": metadata.get("filename"),
            "page_count": metadata.get("page_count")
        }
    }
    
    # Analysiere jeden Metadaten-Eintrag
    for key, value in metadata.items():
        value_type = type(value).__name__
        
        if value_type not in analysis["field_types"]:
            analysis["field_types"][value_type] = 0
        analysis["field_types"][value_type] += 1
        
        # Leere Felder identifizieren
        if value is None or (isinstance(value, str) and not value.strip()):
            analysis["empty_fields"].append(key)
        
        # Problematische Felder identifizieren
        if key == "element_counts":
            if isinstance(value, str):
                try:
                    # Versuche JSON zu parsen
                    json.loads(value)
                    logger.debug(f"✅ element_counts ist gültiger JSON-String")
                except json.JSONDecodeError:
                    analysis["problematic_fields"].append({
                        "field": key,
                        "issue": "Ungültiger JSON-String",
                        "value": value[:100] + "..." if len(str(value)) > 100 else value
                    })
                    logger.warning(f"⚠️ element_counts ist ungültiger JSON: {value}")
            elif isinstance(value, dict):
                logger.debug(f"✅ element_counts ist Dictionary mit {len(value)} Einträgen")
            else:
                analysis["problematic_fields"].append({
                    "field": key,
                    "issue": f"Unerwarteter Typ: {value_type}",
                    "value": str(value)[:100]
                })
    
    return analysis


def evaluate_metadata_quality(metadata: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Bewertet die Qualität der extrahierten Metadaten.
    
    Args:
        metadata: Metadaten-Dictionary
        logger: Logger-Instanz
        
    Returns:
        Qualitätsbewertung
    """
    quality = {
        "score": 0.0,
        "max_score": 100.0,
        "criteria": {},
        "recommendations": []
    }
    
    # Bewertungskriterien
    criteria = {
        "has_title": 20,
        "has_author": 15,
        "has_filename": 5,
        "has_page_count": 10,
        "has_file_size": 5,
        "has_extraction_timestamp": 5,
        "has_element_counts": 15,
        "has_structure_info": 10,
        "has_language": 5,
        "has_norms": 10
    }
    
    # Bewerte jedes Kriterium
    for criterion, max_points in criteria.items():
        achieved = 0
        
        if criterion == "has_title":
            if metadata.get("title"):
                achieved = max_points
            else:
                quality["recommendations"].append("Titel fehlt - prüfe PDF-Metadaten und Heuristiken")
        
        elif criterion == "has_author":
            if metadata.get("author"):
                achieved = max_points
            else:
                quality["recommendations"].append("Autor fehlt - prüfe PDF-Metadaten")
        
        elif criterion == "has_filename":
            if metadata.get("filename"):
                achieved = max_points
        
        elif criterion == "has_page_count":
            if metadata.get("page_count"):
                achieved = max_points
            else:
                quality["recommendations"].append("Seitenzahl fehlt - prüfe PDF-Struktur")
        
        elif criterion == "has_file_size":
            if metadata.get("file_size_bytes") or metadata.get("file_size_mb"):
                achieved = max_points
        
        elif criterion == "has_extraction_timestamp":
            if metadata.get("extraction_timestamp"):
                achieved = max_points
        
        elif criterion == "has_element_counts":
            if metadata.get("element_counts"):
                achieved = max_points
                # Bonus für strukturierte Element-Counts
                if isinstance(metadata.get("element_counts"), dict) and len(metadata["element_counts"]) > 0:
                    achieved += 5
            else:
                quality["recommendations"].append("Element-Zählung fehlt - prüfe Docling-Integration")
        
        elif criterion == "has_structure_info":
            structure_fields = ["has_tables", "has_images", "has_headings", "structure_score"]
            found_structure = sum(1 for field in structure_fields if metadata.get(field) is not None)
            achieved = (found_structure / len(structure_fields)) * max_points
        
        elif criterion == "has_language":
            if metadata.get("language"):
                achieved = max_points
        
        elif criterion == "has_norms":
            if metadata.get("din_norm") or metadata.get("iso_norm"):
                achieved = max_points
        
        quality["criteria"][criterion] = {
            "achieved": achieved,
            "max_points": max_points,
            "percentage": (achieved / max_points) * 100 if max_points > 0 else 0
        }
        
        quality["score"] += achieved
    
    # Berechne Gesamtprozentsatz
    quality["percentage"] = (quality["score"] / quality["max_score"]) * 100
    
    # Qualitätsbewertung
    if quality["percentage"] >= 80:
        quality["rating"] = "Excellent"
    elif quality["percentage"] >= 60:
        quality["rating"] = "Good"
    elif quality["percentage"] >= 40:
        quality["rating"] = "Fair"
    else:
        quality["rating"] = "Poor"
    
    return quality


def validate_metadata_types(metadata: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Validiert die Typen aller Metadaten-Felder.
    
    Args:
        metadata: Metadaten-Dictionary
        logger: Logger-Instanz
        
    Returns:
        Validierungsergebnisse
    """
    validation = {
        "total_fields": len(metadata),
        "valid_fields": 0,
        "invalid_fields": [],
        "type_distribution": {},
        "warnings": []
    }
    
    # Erwartete Typen für bestimmte Felder
    expected_types = {
        "filename": str,
        "title": str,
        "author": str,
        "page_count": int,
        "file_size_bytes": int,
        "file_size_mb": float,
        "year": int,
        "extraction_confidence": float,
        "has_tables": bool,
        "has_images": bool,
        "has_formulas": bool,
    }
    
    for field_name, field_value in metadata.items():
        actual_type = type(field_value)
        
        # Zähle Typ-Verteilung
        type_name = actual_type.__name__
        if type_name not in validation["type_distribution"]:
            validation["type_distribution"][type_name] = 0
        validation["type_distribution"][type_name] += 1
        
        # Prüfe erwartete Typen
        if field_name in expected_types:
            expected_type = expected_types[field_name]
            
            if isinstance(field_value, expected_type):
                validation["valid_fields"] += 1
            else:
                validation["invalid_fields"].append({
                    "field": field_name,
                    "expected_type": expected_type.__name__,
                    "actual_type": type_name,
                    "value": str(field_value)[:100]
                })
        else:
            # Unbekanntes Feld - als gültig betrachten
            validation["valid_fields"] += 1
        
        # Spezielle Validierungen
        if field_name == "element_counts":
            if isinstance(field_value, str):
                try:
                    json.loads(field_value)
                except json.JSONDecodeError:
                    validation["warnings"].append(f"element_counts ist ungültiger JSON: {field_value}")
            elif not isinstance(field_value, dict):
                validation["warnings"].append(f"element_counts sollte Dict oder JSON-String sein, ist aber {type_name}")
    
    return validation


def run_metadata_debug_test():
    """Führt den kompletten Metadaten-Debug-Test durch."""
    # Logging konfigurieren
    logger = setup_detailed_logging()
    
    # Logs-Verzeichnis erstellen
    log_dir = Path("Test_Skripts/logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.info("🚀 STARTE METADATEN-DEBUG-TEST")
    
    # Test-PDFs definieren
    test_pdfs = []
    
    # Suche nach verfügbaren Test-PDFs
    sample_dirs = [
        Path("Test_Skripts/Sample_PDFs"),
        Path("Test_Skripts/sample_pdfs"),
        Path("Sample_PDFs"),
        Path("sample_pdfs")
    ]
    
    for sample_dir in sample_dirs:
        if sample_dir.exists():
            pdf_files = list(sample_dir.glob("*.pdf"))
            test_pdfs.extend(pdf_files)
            logger.info(f"📁 Gefunden: {len(pdf_files)} PDFs in {sample_dir}")
            break
    
    if not test_pdfs:
        logger.error("❌ Keine Test-PDFs gefunden!")
        return
    
    # Führe Tests für jede PDF durch
    all_results = []
    
    for i, pdf_path in enumerate(test_pdfs[:3]):  # Teste nur erste 3 PDFs
        logger.info(f"\n{'='*80}")
        logger.info(f"📄 TEST {i+1}/{min(3, len(test_pdfs))}: {pdf_path.name}")
        logger.info(f"{'='*80}")
        
        try:
            result = test_metadata_extraction_single_pdf(pdf_path, logger)
            all_results.append(result)
            
            # Zusammenfassung für diese PDF
            success_msg = "✅ ERFOLGREICH" if result["success"] else "❌ FEHLGESCHLAGEN"
            error_count = len(result["errors"])
            warning_count = len(result["warnings"])
            
            logger.info(f"📊 ERGEBNIS: {success_msg}")
            logger.info(f"📊 Fehler: {error_count}, Warnungen: {warning_count}")
            
            if result.get("metadata_quality"):
                quality = result["metadata_quality"]
                logger.info(f"📊 Metadaten-Qualität: {quality['rating']} ({quality['percentage']:.1f}%)")
            
        except Exception as e:
            logger.error(f"💥 Kritischer Fehler bei {pdf_path.name}: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Gesamtstatistik
    logger.info(f"\n{'='*80}")
    logger.info("📈 GESAMTSTATISTIK")
    logger.info(f"{'='*80}")
    
    successful_tests = sum(1 for r in all_results if r["success"])
    total_tests = len(all_results)
    
    logger.info(f"📊 Erfolgreiche Tests: {successful_tests}/{total_tests}")
    logger.info(f"📊 Erfolgsrate: {(successful_tests/total_tests)*100:.1f}%")
    
    # Häufigste Fehler
    all_errors = []
    for result in all_results:
        all_errors.extend(result["errors"])
    
    if all_errors:
        logger.info("❌ Häufigste Fehler:")
        for error in set(all_errors):
            count = all_errors.count(error)
            logger.info(f"   • {error} ({count}x)")
    
    # Speichere detaillierte Ergebnisse
    results_file = log_dir / "metadata_debug_results.json"
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"📄 Detaillierte Ergebnisse gespeichert: {results_file}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Ergebnisse: {e}")
    
    logger.info("🏁 METADATEN-DEBUG-TEST ABGESCHLOSSEN")


if __name__ == "__main__":
    run_metadata_debug_test() 