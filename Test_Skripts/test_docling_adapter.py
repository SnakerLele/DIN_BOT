#!/usr/bin/env python3
"""
Test-Skript für das neue Docling-Adapter System
===============================================

Dieses Skript testet die neue docling_adapter Integration und gibt
detaillierte Ergebnisse als TXT-Datei aus.

Funktionen:
- Lädt und verarbeitet PDFs mit dem neuen DoclingAdapter
- Analysiert die erstellten LlamaIndex Documents
- Exportiert alle Dokumente als strukturierte TXT-Datei
- Zeigt detaillierte Statistiken und Metadaten
- Testet verschiedene Konfigurationen

Verwendung:
    python test_docling_adapter.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Füge Parse_Index zum Python-Pfad hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from Parse_Index.docling_adapter import DoclingAdapter
    from Parse_Index.config import DOCLING_CONFIG, PDF_FOLDER
    from llama_index.core import Document
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    print("Stelle sicher, dass Parse_Index korrekt installiert ist.")
    sys.exit(1)


class DoclingTester:
    """Testklasse für das Docling-Adapter System."""
    
    def __init__(self, output_dir: str = "./Test_Outputs"):
        """
        Initialisiert den Docling-Tester.
        
        Args:
            output_dir: Verzeichnis für Ausgabe-Dateien
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test-Konfigurationen
        self.test_configs = {
            "standard": DOCLING_CONFIG,
            "minimal": {
                "ocr_enabled": False,
                "table_extraction": False,
                "image_extraction": False,
                "formula_extraction": False,
                "layout_analysis": True,
                "reading_order": True,
                "export_format": "markdown",
                "chunk_by_page": False,
                "preserve_formatting": True,
                "extract_metadata": True,
            },
            "maximum": {
                "ocr_enabled": True,
                "table_extraction": True,
                "image_extraction": True,
                "formula_extraction": True,
                "layout_analysis": True,
                "reading_order": True,
                "export_format": "markdown",
                "chunk_by_page": False,
                "preserve_formatting": True,
                "extract_metadata": True,
                "timeout_seconds": 600,
                "chunk_strategy": "hybrid",
                "max_chunk_size": 3000,
                "chunk_overlap": 300,
            }
        }
        
        self.results = {}
    
    def run_all_tests(self):
        """Führt alle Tests aus."""
        print("🧪 Starte Docling-Adapter Tests")
        print("=" * 50)
        
        # 1. Verfügbarkeitstest
        self.test_docling_availability()
        
        # 2. PDF-Dateien finden
        pdf_files = self.find_test_pdfs()
        if not pdf_files:
            print("❌ Keine PDF-Dateien zum Testen gefunden!")
            return
        
        # 3. Tests für verschiedene Konfigurationen
        for config_name, config in self.test_configs.items():
            print(f"\n📋 Teste Konfiguration: {config_name}")
            print("-" * 30)
            
            self.test_configuration(config_name, config, pdf_files[0])  # Teste mit erstem PDF
        
        # 4. Batch-Test mit allen PDFs
        print(f"\n📦 Batch-Test mit allen {len(pdf_files)} PDFs")
        print("-" * 40)
        self.test_batch_processing(pdf_files)
        
        # 5. Ergebnisse exportieren
        self.export_results()
        
        print("\n✅ Alle Tests abgeschlossen!")
        print(f"📁 Ergebnisse gespeichert in: {self.output_dir}")
    
    def test_docling_availability(self):
        """Testet die Docling-Verfügbarkeit."""
        print("🔍 Teste Docling-Verfügbarkeit...")
        
        try:
            adapter = DoclingAdapter()
            is_available = adapter.is_docling_available()
            
            if is_available:
                print("✅ Docling ist verfügbar und funktionsfähig")
                config = adapter.get_config()
                print(f"   Konfiguration: {len(config)} Parameter")
            else:
                print("❌ Docling ist nicht verfügbar")
                
            self.results["availability"] = {
                "docling_available": is_available,
                "config_parameters": len(config) if is_available else 0
            }
            
        except Exception as e:
            print(f"❌ Fehler beim Verfügbarkeitstest: {e}")
            self.results["availability"] = {
                "docling_available": False,
                "error": str(e)
            }
    
    def find_test_pdfs(self) -> List[Path]:
        """Findet PDF-Dateien zum Testen."""
        print("📁 Suche Test-PDFs...")
        
        pdf_paths = []
        
        # Suche in PDF_FOLDER
        pdf_folder = Path(PDF_FOLDER)
        if pdf_folder.exists():
            pdfs = list(pdf_folder.glob("*.pdf"))
            pdf_paths.extend(pdfs)
            print(f"   Gefunden in {PDF_FOLDER}: {len(pdfs)} PDFs")
        
        # Suche in Test_Skripts/Sample_PDFs
        sample_folder = Path(__file__).parent / "Sample_PDFs"
        if sample_folder.exists():
            sample_pdfs = list(sample_folder.glob("*.pdf"))
            pdf_paths.extend(sample_pdfs)
            print(f"   Gefunden in Sample_PDFs: {len(sample_pdfs)} PDFs")
        
        print(f"📊 Gesamt gefunden: {len(pdf_paths)} PDF-Dateien")
        
        for pdf in pdf_paths[:3]:  # Zeige nur erste 3
            print(f"   - {pdf.name} ({pdf.stat().st_size / 1024 / 1024:.1f} MB)")
        
        return pdf_paths
    
    def test_configuration(self, config_name: str, config: Dict[str, Any], pdf_path: Path):
        """
        Testet eine spezifische Konfiguration.
        
        Args:
            config_name: Name der Konfiguration
            config: Konfigurationsdictionary
            pdf_path: Pfad zur Test-PDF
        """
        print(f"📄 Teste mit: {pdf_path.name}")
        
        try:
            # Adapter mit Test-Konfiguration erstellen
            adapter = DoclingAdapter(config=config)
            
            # PDF verarbeiten
            start_time = datetime.now()
            documents = adapter.parse_pdf(pdf_path)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            # Statistiken sammeln
            stats = adapter.get_last_processing_stats()
            
            # Ergebnisse analysieren
            analysis = self.analyze_documents(documents)
            
            # Ergebnisse speichern
            self.results[config_name] = {
                "pdf_file": pdf_path.name,
                "config": config,
                "processing_time_seconds": processing_time,
                "document_count": len(documents),
                "processing_stats": stats,
                "document_analysis": analysis,
                "success": True
            }
            
            # Dokumente als TXT exportieren
            self.export_documents_txt(documents, f"{config_name}_{pdf_path.stem}")
            
            print(f"✅ Erfolgreich: {len(documents)} Dokumente in {processing_time:.1f}s")
            print(f"   📊 Seiten: {stats.get('pages', 'N/A')}")
            print(f"   📋 Tabellen: {stats.get('tables', 0)}")
            print(f"   🖼️ Bilder: {stats.get('images', 0)}")
            print(f"   📐 Formeln: {stats.get('formulas', 0)}")
            
        except Exception as e:
            print(f"❌ Fehler bei Konfiguration {config_name}: {e}")
            self.results[config_name] = {
                "pdf_file": pdf_path.name,
                "config": config,
                "error": str(e),
                "success": False
            }
    
    def test_batch_processing(self, pdf_paths: List[Path]):
        """Testet Batch-Verarbeitung mehrerer PDFs."""
        if len(pdf_paths) <= 1:
            print("⚠️ Nicht genügend PDFs für Batch-Test")
            return
        
        try:
            adapter = DoclingAdapter(config=self.test_configs["standard"])
            
            # Batch-Verarbeitung
            start_time = datetime.now()
            all_documents = adapter.parse_multiple_pdfs(
                pdf_paths[:3],  # Teste nur erste 3 PDFs
                strict_mode=False,
                show_progress=True
            )
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            # Ergebnisse analysieren
            analysis = self.analyze_documents(all_documents)
            
            self.results["batch_processing"] = {
                "pdf_count": min(3, len(pdf_paths)),
                "total_documents": len(all_documents),
                "processing_time_seconds": processing_time,
                "documents_per_second": len(all_documents) / processing_time if processing_time > 0 else 0,
                "document_analysis": analysis,
                "success": True
            }
            
            # Batch-Ergebnisse als TXT exportieren
            self.export_documents_txt(all_documents, "batch_processing_results")
            
            print(f"✅ Batch-Verarbeitung erfolgreich:")
            print(f"   📄 {min(3, len(pdf_paths))} PDFs verarbeitet")
            print(f"   📋 {len(all_documents)} Dokumente erstellt")
            print(f"   ⏱️ {processing_time:.1f}s Gesamtzeit")
            print(f"   🚀 {len(all_documents) / processing_time:.1f} Dokumente/s")
            
        except Exception as e:
            print(f"❌ Fehler bei Batch-Verarbeitung: {e}")
            self.results["batch_processing"] = {
                "error": str(e),
                "success": False
            }
    
    def analyze_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Analysiert die erstellten Dokumente.
        
        Args:
            documents: Liste von LlamaIndex Documents
            
        Returns:
            Dictionary mit Analyse-Ergebnissen
        """
        if not documents:
            return {"empty": True}
        
        # Text-Statistiken
        text_lengths = [len(doc.text) for doc in documents]
        total_text_length = sum(text_lengths)
        
        # Metadaten-Analyse
        metadata_keys = set()
        for doc in documents:
            if doc.metadata:
                metadata_keys.update(doc.metadata.keys())
        
        # Qualitäts-Scores sammeln
        quality_scores = []
        for doc in documents:
            if doc.metadata and 'quality_score' in doc.metadata:
                quality_scores.append(doc.metadata['quality_score'])
        
        return {
            "document_count": len(documents),
            "total_text_length": total_text_length,
            "average_text_length": total_text_length / len(documents),
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths),
            "metadata_keys": list(metadata_keys),
            "metadata_key_count": len(metadata_keys),
            "quality_scores": quality_scores,
            "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else None
        }
    
    def export_documents_txt(self, documents: List[Document], filename_prefix: str):
        """
        Exportiert Dokumente als strukturierte TXT-Datei.
        
        Args:
            documents: Liste von LlamaIndex Documents
            filename_prefix: Präfix für Dateiname
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{filename_prefix}_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Docling-Adapter Test-Ergebnisse\n")
            f.write(f"================================\n")
            f.write(f"Erstellt: {datetime.now().isoformat()}\n")
            f.write(f"Anzahl Dokumente: {len(documents)}\n")
            f.write(f"Konfiguration: {filename_prefix}\n\n")
            
            for i, doc in enumerate(documents, 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"DOKUMENT {i:03d} von {len(documents)}\n")
                f.write(f"{'='*60}\n")
                
                # Metadaten
                if doc.metadata:
                    f.write(f"\n📋 METADATEN:\n")
                    f.write(f"{'-'*20}\n")
                    for key, value in doc.metadata.items():
                        # Lange Werte kürzen
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        f.write(f"{key}: {value}\n")
                
                # Text-Inhalt
                f.write(f"\n📄 INHALT:\n")
                f.write(f"{'-'*20}\n")
                f.write(f"Länge: {len(doc.text)} Zeichen\n")
                f.write(f"Zeilen: {doc.text.count(chr(10)) + 1}\n\n")
                f.write(doc.text)
                f.write(f"\n\n{'='*60}\n")
        
        print(f"📝 Dokumente exportiert: {output_file}")
    
    def export_results(self):
        """Exportiert Test-Ergebnisse als JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"docling_test_results_{timestamp}.json"
        
        # Ergebnisse für JSON serialisierbar machen
        serializable_results = {}
        for key, value in self.results.items():
            serializable_results[key] = self._make_serializable(value)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Test-Ergebnisse exportiert: {results_file}")
        
        # Zusammenfassung ausgeben
        self.print_summary()
    
    def _make_serializable(self, obj):
        """Macht Objekte JSON-serialisierbar."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def print_summary(self):
        """Gibt eine Zusammenfassung der Test-Ergebnisse aus."""
        print(f"\n📊 TEST-ZUSAMMENFASSUNG")
        print(f"{'='*50}")
        
        successful_tests = sum(1 for result in self.results.values() 
                             if isinstance(result, dict) and result.get('success', False))
        total_tests = len(self.results)
        
        print(f"✅ Erfolgreiche Tests: {successful_tests}/{total_tests}")
        
        if 'availability' in self.results:
            avail = self.results['availability']
            print(f"🔧 Docling verfügbar: {'✅' if avail.get('docling_available') else '❌'}")
        
        # Beste Konfiguration finden
        best_config = None
        best_score = 0
        
        for config_name, result in self.results.items():
            if isinstance(result, dict) and result.get('success') and 'document_count' in result:
                score = result['document_count'] / result.get('processing_time_seconds', 1)
                if score > best_score:
                    best_score = score
                    best_config = config_name
        
        if best_config:
            print(f"🏆 Beste Konfiguration: {best_config} ({best_score:.1f} Docs/s)")
        
        print(f"📁 Alle Ergebnisse in: {self.output_dir}")


def main():
    """Hauptfunktion für den Docling-Test."""
    print("🧪 Docling-Adapter Test-Suite")
    print("=" * 50)
    
    # Tester erstellen und ausführen
    tester = DoclingTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main() 