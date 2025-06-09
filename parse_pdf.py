#!/usr/bin/env python3
"""
Neuer Einstiegspunkt für die PDF-Indexierung mit modularer Struktur

Dieses Skript ist der neue Ersatz für parse_pdf.py und verwendet 
das modulare Parse_Index System für bessere Code-Organisation.

Verwendung:
    python parse_pdf_new.py

Das Skript verwendet die modulare Struktur in Parse_Index/ für:
- Konfiguration (config.py)
- Utility-Funktionen (utils.py) 
- Node-Parser (node_parsers.py)
- Unstructured-Integration (unstructured_wrapper.py)
- Dokumentverbesserung (document_enhancer.py)
- Hauptorchestrierung (main_parser.py)
"""

if __name__ == "__main__":
    try:
        # Importiere und starte das modulare System
        from Parse_Index.main_parser import main
        
        print("🚀 Starte modulares PDF-Indexierung System")
        print("📁 Verwendet Parse_Index/ Module für bessere Code-Organisation")
        print()
        
        # Starte die Hauptfunktion
        main()
        
    except ImportError as e:
        print("❌ Fehler beim Importieren der Parse_Index Module:")
        print(f"   {str(e)}")
        print()
        print("💡 Lösungsvorschläge:")
        print("   1. Prüfe ob der Parse_Index/ Ordner existiert")
        print("   2. Prüfe ob alle Module in Parse_Index/ vorhanden sind")
        print("   3. Stelle sicher, dass Python die Module finden kann")
        
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {str(e)}")
        import traceback
        print(traceback.format_exc()) 