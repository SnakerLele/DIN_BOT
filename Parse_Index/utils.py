"""
Hilfsfunktionen für das Parse_Index System

Dieses Modul enthält allgemeine Utility-Funktionen für:
- GPU-Status und Hardware-Checks
- Requirements-Validierung  
- Ressourcen-Monitoring
- System-Diagnostik
"""

import os
import sys
import psutil
import torch
import logging

def check_embedding_requirements():
    """
    Prüft, ob alle erforderlichen Bibliotheken für das Embedding-Modell installiert sind.
    
    Returns:
        bool: True wenn alle Requirements erfüllt sind, False sonst
    """
    print("\n===== EMBEDDING REQUIREMENTS =====")
    
    requirements_ok = True
    
    # Prüfe sentence-transformers
    try:
        import sentence_transformers
        version = sentence_transformers.__version__
        print(f"✓ sentence-transformers Version: {version}")
        
        # Prüfe ob Version >= 2.0.0 (ausreichend für paraphrase-multilingual-mpnet-base-v2)
        try:
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse("2.0.0"):
                print(f"⚠ sentence-transformers Version {version} ist zu alt!")
                print("  Mindestversion: 2.0.0")
                print("  Upgrade mit: pip install --upgrade sentence-transformers")
                requirements_ok = False
            else:
                print("✓ sentence-transformers Version ist kompatibel")
        except ImportError:
            print("⚠ packaging nicht verfügbar, überspringe Versions-Check")
            
    except ImportError:
        print("❌ sentence-transformers nicht installiert")
        print("  Installiere mit: pip install sentence-transformers")
        requirements_ok = False
    
    # Prüfe transformers (benötigt für HuggingFace Integration)
    try:
        import transformers
        version = transformers.__version__
        print(f"✓ transformers Version: {version}")
            
    except ImportError:
        print("❌ transformers nicht installiert")
        print("  Installiere mit: pip install transformers")
        requirements_ok = False
    
    print("=" * 35)
    
    if not requirements_ok:
        print("\n❌ Nicht alle erforderlichen Bibliotheken sind installiert!")
        print("Bitte installiere die fehlenden Pakete vor dem Fortfahren.")
        return False
    else:
        print("\n✓ Alle erforderlichen Bibliotheken sind verfügbar!")
        return True

def check_gpu_status():
    """
    Prüft den GPU-Status und gibt detaillierte Informationen aus.
    
    Returns:
        dict: GPU-Status Informationen
    """
    print("\n===== GPU-STATUS =====")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA verfügbar: {torch.cuda.is_available()}")
    
    gpu_info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpus": []
    }
    
    if torch.cuda.is_available():
        gpu_info["cuda_version"] = torch.version.cuda
        gpu_info["gpu_count"] = torch.cuda.device_count()
        
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Anzahl GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
            
            gpu_device_info = {
                "index": i,
                "name": gpu_name,
                "memory_total": None,
                "memory_used": None
            }
            
            # GPU-Speichernutzung prüfen, falls verfügbar
            try:
                # Prüfe ob nvidia-smi installiert ist
                import nvidia_smi
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = info.used // 1024 // 1024
                memory_total_mb = info.total // 1024 // 1024
                print(f"  Speicher: {memory_used_mb} MB / {memory_total_mb} MB")
                
                gpu_device_info["memory_used"] = memory_used_mb
                gpu_device_info["memory_total"] = memory_total_mb
                nvidia_smi.nvmlShutdown()
            except ImportError:
                print("  nvidia-smi nicht installiert - Speicherinfo nicht verfügbar")
            except Exception as e:
                print(f"  Fehler beim Abrufen der Speicherinfo: {str(e)}")
            
            gpu_info["gpus"].append(gpu_device_info)
    
    print("======================\n")
    return gpu_info

def log_resources(message: str, logger=None):
    """
    Loggt aktuelle Ressourcennutzung mit einer beschreibenden Nachricht.
    
    Args:
        message: Beschreibende Nachricht für den Log-Eintrag
        logger: Optional Logger-Instanz, verwendet print() falls None
    """
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        log_msg = f"[RESSOURCEN] {message}: Speicher: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%"
        
        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)
            
    except Exception as e:
        error_msg = f"[RESSOURCEN] Fehler beim Ressourcen-Logging: {str(e)}"
        if logger:
            logger.warning(error_msg)
        else:
            print(error_msg)

def check_disk_space(directory: str, required_gb: float = 1.0):
    """
    Prüft verfügbaren Festplattenspeicher in einem Verzeichnis.
    
    Args:
        directory: Zu prüfendes Verzeichnis
        required_gb: Mindestens erforderlicher Speicher in GB
        
    Returns:
        bool: True wenn genügend Speicher verfügbar ist
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        stat = os.statvfs(directory) if hasattr(os, 'statvfs') else None
        if stat:
            # Unix/Linux
            available_bytes = stat.f_bavail * stat.f_frsize
        else:
            # Windows
            import shutil
            available_bytes = shutil.disk_usage(directory).free
            
        available_gb = available_bytes / (1024**3)
        
        print(f"Verfügbarer Speicher in '{directory}': {available_gb:.2f} GB")
        
        if available_gb < required_gb:
            print(f"⚠ Warnung: Nur {available_gb:.2f} GB verfügbar, {required_gb} GB empfohlen")
            return False
        else:
            print(f"✓ Genügend Speicher verfügbar ({available_gb:.2f} GB)")
            return True
            
    except Exception as e:
        print(f"❌ Fehler beim Prüfen des Speicherplatzes: {str(e)}")
        return False

def setup_logging(level="INFO", format_string=None):
    """
    Konfiguriert das Logging-System.
    
    Args:
        level: Logging-Level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom Format-String für Log-Nachrichten
        
    Returns:
        logging.Logger: Konfigurierte Logger-Instanz
    """
    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
        format=format_string,
        force=True  # Überschreibt existierende Konfiguration
    )
    
    logger = logging.getLogger("Parse_Index")
    logger.info(f"Logging konfiguriert mit Level: {level}")
    
    return logger

def validate_environment():
    """
    Führt eine umfassende Umgebungsvalidierung durch.
    
    Returns:
        dict: Validierungsergebnisse mit Details
    """
    print("\n===== UMGEBUNGSVALIDIERUNG =====")
    
    validation_results = {
        "requirements_ok": False,
        "gpu_available": False,
        "disk_space_ok": False,
        "python_version_ok": False,
        "errors": [],
        "warnings": []
    }
    
    # Python-Version prüfen
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✓ Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        validation_results["python_version_ok"] = True
    else:
        error = f"❌ Python Version {python_version.major}.{python_version.minor} zu alt (mindestens 3.8 erforderlich)"
        print(error)
        validation_results["errors"].append(error)
    
    # Requirements prüfen
    validation_results["requirements_ok"] = check_embedding_requirements()
    
    # GPU-Status prüfen  
    gpu_info = check_gpu_status()
    validation_results["gpu_available"] = gpu_info["cuda_available"]
    
    # Speicherplatz prüfen
    from .config import PDF_FOLDER, PERSIST_DIR
    validation_results["disk_space_ok"] = (
        check_disk_space(PDF_FOLDER, 0.5) and 
        check_disk_space(PERSIST_DIR, 2.0)
    )
    
    print("=" * 33)
    
    # Zusammenfassung
    all_ok = (
        validation_results["requirements_ok"] and 
        validation_results["python_version_ok"] and
        validation_results["disk_space_ok"]
    )
    
    if all_ok:
        print("\n✅ Umgebungsvalidierung erfolgreich!")
        if not validation_results["gpu_available"]:
            print("⚠ GPU nicht verfügbar - Embedding-Berechnung läuft auf CPU")
            validation_results["warnings"].append("GPU nicht verfügbar")
    else:
        print("\n❌ Umgebungsvalidierung fehlgeschlagen!")
        print("Bitte behebe die oben genannten Probleme.")
    
    return validation_results 