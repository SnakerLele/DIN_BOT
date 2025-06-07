#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BFT Chat Bot Launcher
---------------------
Ein CustomTkinter-basierter Launcher für den BFT Chat Bot mit Indizierungs- und Chat-Funktionen.
"""

import os
import sys
import subprocess
import webbrowser
import shutil
import logging
import time
import threading
import requests
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bft_launcher.log')
    ]
)
logger = logging.getLogger(__name__)

# Angepasste Farbpalette mit weicheren Tönen
COLORS = {
    "bg_light": "#FFF5EB",    # Sehr helles Pfirsich für Haupthintergrund
    "bg_frame": "#FFF0E5",    # Leicht dunkler für Frames
    "accent_blue": "#5DA9DD", # Sanftes Blau für Buttons
    "accent_blue_hover": "#4A95C9", # Dunkleres Blau für Hover
    "text_dark": "#333333",   # Dunkelgrau für Text (guter Kontrast)
    "border": "#F0D0C0",      # Sanfte Umrandungsfarbe
}

# Ersetze die Theme-Einstellungen
ctk.set_appearance_mode("light")  # Heller Modus für weichere Farben
ctk.set_default_color_theme("blue")  # Grundfarbe blau beibehalten

class LogWindow(ctk.CTkToplevel):
    """Separates Fenster zur Anzeige von Logs"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Log-Anzeige")
        self.geometry("800x500")
        
        # Größenverhalten konfigurieren
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Log-Textfeld
        self.log_text = ctk.CTkTextbox(self, corner_radius=0)
        self.log_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.log_text.configure(state="disabled")
        
        # Schaltfläche zum Schließen
        close_button = ctk.CTkButton(
            self, text="Schließen", command=self.destroy,
            height=30
        )
        close_button.grid(row=1, column=0, padx=10, pady=10)
    
    def update_log(self, text):
        """Aktualisiert den Log-Text"""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")  # Scrolle zum Ende
        self.log_text.configure(state="disabled")

class TextRedirector:
    """Umleitung der Ausgabe zur GUI"""
    def __init__(self, widget, tag=""):
        self.widget = widget
        self.tag = tag

    def write(self, string):
        """Schreibt Text in das Widget"""
        self.widget.update_log(string)
    
    def flush(self):
        """Dummy-Methode für Kompatibilität"""
        pass

class MainApp(ctk.CTk):
    """Hauptfenster der Anwendung"""
    def __init__(self):
        super().__init__()
        self.title("BFT Chat Bot Launcher")
        self.geometry("900x600")
        self.minsize(600, 400)  # Minimale Fenstergröße setzen
        
        # Skalierungsverhalten für das Hauptfenster
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.server_process = None
        self.selected_folder = None
        self.log_window = None
        self.init_ui()
        
        logger.info("BFT Chat Bot Launcher gestartet")
    
    def init_ui(self):
        """Initialisiert die Benutzeroberfläche"""
        # Hauptcontainer-Hintergrund
        self.configure(fg_color=COLORS["bg_light"])
        
        # Frame für den Inhalt - abgerundete Ecken und weicher Schatten
        main_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_light"], corner_radius=0, border_width=0)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Skalierungsverhalten für das Hauptframe
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=0)  # Überschrift
        main_frame.grid_rowconfigure(1, weight=1)  # Inhalt
        main_frame.grid_rowconfigure(2, weight=0)  # Status
        
        # Überschrift mit angepasster Farbe
        header = ctk.CTkLabel(
            main_frame, text="BFT Chat Bot Launcher", 
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS["text_dark"]
        )
        header.grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        # Linker Frame - Chat starten
        chat_frame = ctk.CTkFrame(
            main_frame, corner_radius=15, 
            fg_color=COLORS["bg_frame"],
            border_width=1, border_color=COLORS["border"]
        )
        chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # Skalierungsverhalten für den Chat-Frame
        chat_frame.grid_columnconfigure(0, weight=1)
        chat_frame.grid_rowconfigure(0, weight=0)  # Header
        chat_frame.grid_rowconfigure(1, weight=1)  # Description
        chat_frame.grid_rowconfigure(2, weight=0)  # Button
        
        # Chat-Überschrift
        chat_header = ctk.CTkLabel(
            chat_frame, text="Chat Bot", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        chat_header.grid(row=0, column=0, pady=10)
        
        # Chat-Beschreibung
        chat_desc = ctk.CTkLabel(
            chat_frame, 
            text="Starte den Chat Bot mit bestehenden Indizes.",
            wraplength=300
        )
        chat_desc.grid(row=1, column=0, pady=(5, 15), padx=20)
        
        # Chat-Button
        chat_button = ctk.CTkButton(
            chat_frame, text="Chatten starten", 
            command=self.start_chat,
            height=40, corner_radius=8,
            fg_color=COLORS["accent_blue"],
            hover_color=COLORS["accent_blue_hover"],
            font=ctk.CTkFont(size=15, weight="bold")
        )
        chat_button.grid(row=2, column=0, pady=(5, 20), padx=20)
        
        # Rechter Frame - Indizierung
        index_frame = ctk.CTkFrame(
            main_frame, corner_radius=15, 
            fg_color=COLORS["bg_frame"],
            border_width=1, border_color=COLORS["border"]
        )
        index_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        # Skalierungsverhalten für den Index-Frame
        index_frame.grid_columnconfigure(0, weight=1)
        index_frame.grid_rowconfigure(0, weight=0)  # Header
        index_frame.grid_rowconfigure(1, weight=1)  # Description
        index_frame.grid_rowconfigure(2, weight=0)  # Folder Button
        index_frame.grid_rowconfigure(3, weight=0)  # Index Buttons
        
        # Indizierungs-Überschrift
        index_header = ctk.CTkLabel(
            index_frame, text="Dateien indizieren", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        index_header.grid(row=0, column=0, pady=10)
        
        # Indizierungs-Beschreibung
        index_desc = ctk.CTkLabel(
            index_frame, 
            text="Wähle einen Ordner mit PDF-Dateien und starte die Indizierung.",
            wraplength=300
        )
        index_desc.grid(row=1, column=0, pady=(5, 15), padx=20)
        
        # Ordner auswählen Button
        folder_button = ctk.CTkButton(
            index_frame, text="Ordner auswählen", 
            command=self.select_folder,
            height=40, corner_radius=8,
            fg_color=COLORS["accent_blue"],
            hover_color=COLORS["accent_blue_hover"],
            font=ctk.CTkFont(size=15, weight="bold")
        )
        folder_button.grid(row=2, column=0, pady=(5, 15), padx=20)
        
        # Indizierungs-Buttons (Horizontal nebeneinander)
        index_buttons_frame = ctk.CTkFrame(index_frame, fg_color="transparent")
        index_buttons_frame.grid(row=3, column=0, pady=(0, 20))
        index_buttons_frame.grid_columnconfigure(0, weight=1)
        index_buttons_frame.grid_columnconfigure(1, weight=1)
        index_buttons_frame.grid_columnconfigure(2, weight=1)  # Für den Löschen-Button
        
        # Button: Alles neu indizieren
        index_new_button = ctk.CTkButton(
            index_buttons_frame, text="Alles neu indizieren", 
            command=self.start_indexing_new,
            height=35, corner_radius=8,
            fg_color=COLORS["accent_blue"],
            hover_color=COLORS["accent_blue_hover"],
            font=ctk.CTkFont(size=14)
        )
        index_new_button.grid(row=0, column=0, padx=8)
        
        # Button: Zu Bestehendem hinzufügen
        index_add_button = ctk.CTkButton(
            index_buttons_frame, text="Hinzufügen", 
            command=self.start_indexing_add,
            height=35, corner_radius=8,
            fg_color=COLORS["accent_blue"],
            hover_color=COLORS["accent_blue_hover"],
            font=ctk.CTkFont(size=14)
        )
        index_add_button.grid(row=0, column=1, padx=8)
        
        # Button: Alle Daten löschen
        index_delete_button = ctk.CTkButton(
            index_buttons_frame, text="Alle Daten löschen", 
            command=self.delete_all_data,
            height=35, corner_radius=8,
            fg_color="#E57373",  # Rötlich für Warnung
            hover_color="#D32F2F",  # Dunkleres Rot für Hover
            font=ctk.CTkFont(size=14)
        )
        index_delete_button.grid(row=0, column=2, padx=8)
        
        # Status-Frame mit weicherem Design
        status_frame = ctk.CTkFrame(
            main_frame, corner_radius=10,
            fg_color=COLORS["bg_frame"],
            border_width=1, border_color=COLORS["border"]
        )
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(15, 0))
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_columnconfigure(1, weight=0)
        
        # Status-Label
        self.status_label = ctk.CTkLabel(status_frame, text="Bereit", font=ctk.CTkFont(size=13))
        self.status_label.grid(row=0, column=0, sticky="w", padx=12, pady=10)
        
        # Log-Button
        log_button = ctk.CTkButton(
            status_frame, text="Logs anzeigen", 
            command=self.show_logs,
            width=120, height=30, corner_radius=8,
            fg_color=COLORS["accent_blue"],
            hover_color=COLORS["accent_blue_hover"],
            font=ctk.CTkFont(size=13)
        )
        log_button.grid(row=0, column=1, padx=12, pady=10)
        
    def select_folder(self):
        """Öffnet Dialog zur Auswahl eines Ordners"""
        folder_path = filedialog.askdirectory(title="Ordner mit PDF-Dateien auswählen")
        
        if folder_path:
            self.selected_folder = folder_path
            logger.info(f"Ausgewählter Ordner: {folder_path}")
            self.status_label.configure(text=f"Ordner ausgewählt: {folder_path}")
    
    def start_indexing_new(self):
        """Startet die Indizierung mit vollständig neuem Index"""
        self._start_indexing(mode="new")
    
    def start_indexing_add(self):
        """Startet die Indizierung und fügt Dateien zum bestehenden Index hinzu"""
        self._start_indexing(mode="add")
    
    def _start_indexing(self, mode="new"):
        """Startet die Indizierung mit dem angegebenen Modus"""
        if not self.selected_folder:
            messagebox.showwarning("Warnung", "Bitte wählen Sie zuerst einen Ordner aus.")
            return
        
        # Anzeigen, dass die Indizierung läuft
        self.status_label.configure(text="Indizierung wird vorbereitet...")
        
        # Ziel-Verzeichnis für PDFs
        target_dir = Path("Test_PDF")
        target_dir.mkdir(exist_ok=True)
        
        try:
            # Status aktualisieren
            self.update_status(f"Kopiere Dateien aus {self.selected_folder}...")
            
            # Lösche bestehende Dateien, wenn vollständig neu indiziert wird
            if mode == "new":
                self.update_status("Lösche vorhandene Dateien...")
                for file in target_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                        logger.debug(f"Gelöschte Datei: {file}")
            
            # Kopiere Dateien
            source_dir = Path(self.selected_folder)
            pdf_files = list(source_dir.glob("**/*.pdf"))
            
            if not pdf_files:
                self.update_status("Keine PDF-Dateien gefunden!")
                messagebox.showwarning("Warnung", "Keine PDF-Dateien im ausgewählten Ordner gefunden.")
                return
            
            for i, pdf_file in enumerate(pdf_files):
                self.update_status(f"Kopiere Datei {i+1}/{len(pdf_files)}: {pdf_file.name}")
                shutil.copy2(pdf_file, target_dir / pdf_file.name)
            
            # Starte den PDF-Parser über die Batch-Datei
            self.update_status("Starte PDF-Indizierung mit Docker...")
            if os.path.exists("run_pdf_parser.bat"):
                # Starte die Batch-Datei
                subprocess.Popen(["run_pdf_parser.bat"], shell=True)
                self.update_status("PDF-Parser gestartet. Überprüfen Sie das Docker-Fenster für den Fortschritt.")
                messagebox.showinfo(
                    "PDF-Parser gestartet", 
                    "Die Indizierung wurde über Docker gestartet. Bitte überprüfen Sie das neue Fenster für den Fortschritt."
                )
            else:
                error_msg = "Die Datei run_pdf_parser.bat wurde nicht gefunden."
                self.update_status(error_msg)
                messagebox.showerror("Fehler", error_msg)
                
        except Exception as e:
            error_msg = f"Fehler: {str(e)}"
            logger.exception("Fehler bei der Indizierung:")
            self.update_status(f"Fehler: {error_msg}")
            messagebox.showerror("Fehler", error_msg)
        
        finally:
            self.update_status("Bereit")
    
    def start_chat(self):
        """Startet den RAG-API-Server und öffnet den Chat im Browser"""
        # Prüfe, ob Docker läuft
        if not self.is_docker_running():
            messagebox.showerror(
                "Fehler", 
                "Docker scheint nicht zu laufen. Bitte starten Sie Docker, damit OpenWebUI funktionieren kann."
            )
            return

        # Prüfe, ob der Server bereits läuft
        if self.is_server_running():
            messagebox.showinfo("Information", "Der Chat Bot läuft bereits. Der Browser wird geöffnet.")
            webbrowser.open("http://localhost:3000/")
            return
        
        # Starte den Server-Thread
        self.update_status("Starte Chat Bot...")
        threading.Thread(target=self._run_server, daemon=True).start()
    
    def _run_server(self):
        """Startet den RAG-API-Server im Hintergrund"""
        try:
            # Prüfe, ob rag_api.py existiert
            if not Path("rag_api.py").exists():
                error_msg = "Die Datei rag_api.py wurde nicht gefunden."
                self.update_status(error_msg)
                messagebox.showerror("Fehler", error_msg)
                return
            
            # Starte den Server in einem separaten Prozess
            self.update_status("Starte RAG-API-Server...")
            
            # Windows-spezifischer Start in separatem Fenster
            cmd = f'start cmd /K "uvicorn rag_api:app --host 0.0.0.0 --port 8000"'
            subprocess.Popen(cmd, shell=True)
            
            # Warte, bis der Server läuft
            self.update_status("Warte auf Server-Start...")
            server_online = False
            for i in range(30):  # 30 Sekunden Timeout
                self.update_status(f"Warte auf Server... ({i+1}/30)")
                if self.is_server_running():
                    server_online = True
                    break
                time.sleep(1)
            
            if not server_online:
                self.update_status("Server konnte nicht gestartet werden!")
                messagebox.showerror("Fehler", "Der Server konnte nicht gestartet werden.")
                return
            
            # Prüfe Ollama
            self.update_status("Prüfe, ob Ollama läuft...")
            if not self.is_ollama_running():
                self.update_status("Warnung: Ollama scheint nicht zu laufen!")
                messagebox.showwarning(
                    "Warnung", 
                    "Ollama scheint nicht zu laufen. Der Chat Bot funktioniert möglicherweise nicht korrekt."
                )
            
            # Prüfe, ob OpenWebUI läuft
            self.update_status("Prüfe, ob OpenWebUI läuft...")
            if not self.is_openwebui_running():
                self.update_status("Warnung: OpenWebUI scheint nicht zu laufen!")
                messagebox.showwarning(
                    "Warnung", 
                    "OpenWebUI (Port 3000) scheint nicht zu laufen. Bitte stellen Sie sicher, dass der Docker-Container gestartet wurde."
                )
                # Frage, ob der Benutzer trotzdem fortfahren möchte
                if not messagebox.askyesno("OpenWebUI nicht gefunden", "Möchten Sie trotzdem den Browser öffnen?"):
                    self.update_status("Vorgang abgebrochen.")
                    return
            
            # Öffne Browser
            self.update_status("Öffne Browser...")
            webbrowser.open("http://localhost:3000/")
            self.update_status("Chat Bot läuft!")
            
        except Exception as e:
            error_msg = f"Fehler beim Starten des Servers: {str(e)}"
            logger.exception("Fehler beim Starten des Servers:")
            self.update_status(error_msg)
            messagebox.showerror("Fehler", error_msg)
    
    def is_server_running(self):
        """Prüft, ob der RAG-API-Server läuft"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def is_ollama_running(self):
        """Prüft, ob Ollama läuft"""
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def is_docker_running(self):
        """Prüft, ob Docker läuft"""
        try:
            result = subprocess.run(
                ["docker", "info"], 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def is_openwebui_running(self):
        """Prüft, ob OpenWebUI auf Port 3000 läuft"""
        try:
            response = requests.get("http://localhost:3000/", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def update_status(self, message):
        """Aktualisiert die Statusanzeige"""
        logger.info(message)
        # Aktualisiere GUI im Haupt-Thread
        self.after(0, lambda: self.status_label.configure(text=message))
        # Aktualisiere Log-Fenster, falls vorhanden
        if self.log_window:
            self.after(0, lambda: self.log_window.update_log(message))
    
    def show_logs(self):
        """Zeigt das Log-Fenster an"""
        if not self.log_window or not self.log_window.winfo_exists():
            self.log_window = LogWindow(self)
            
            # Fülle das Log-Fenster mit dem aktuellen Log-Inhalt
            try:
                with open('bft_launcher.log', 'r') as f:
                    for line in f.readlines()[-100:]:  # Zeige die letzten 100 Zeilen
                        self.log_window.update_log(line.strip())
            except:
                self.log_window.update_log("Keine Log-Datei gefunden.")
                
        self.log_window.focus()

    def delete_all_data(self):
        """Löscht alle indizierten Daten"""
        # Sicherheitsabfrage
        response = messagebox.askyesno(
            "Achtung", 
            "Möchten Sie wirklich ALLE indizierten Daten löschen?\n\nDieser Vorgang kann nicht rückgängig gemacht werden."
        )
        
        if not response:
            return
        
        self.update_status("Lösche alle indizierten Daten...")
        
        try:
            # Lösche Dateien im Test_PDF Ordner
            target_dir = Path("Test_PDF")
            if target_dir.exists():
                for file in target_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                        logger.debug(f"Gelöschte Datei: {file}")
            
            # Lösche Chroma DB Verzeichnis
            chroma_dir = Path("chroma_db_store")
            if chroma_dir.exists():
                shutil.rmtree(chroma_dir, ignore_errors=True)
                logger.info("Chroma DB Verzeichnis gelöscht")
            
            success_msg = "Alle indizierten Daten wurden erfolgreich gelöscht!"
            self.update_status(success_msg)
            messagebox.showinfo("Erfolg", success_msg)
                
        except Exception as e:
            error_msg = f"Fehler: {str(e)}"
            logger.exception("Fehler beim Löschen der Daten:")
            self.update_status(f"Fehler: {error_msg}")
            messagebox.showerror("Fehler", error_msg)
        
        finally:
            self.update_status("Bereit")

def main():
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    main() 