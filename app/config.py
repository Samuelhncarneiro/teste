# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Diretórios
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = os.path.join(BASE_DIR, "temp_uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CONVERTED_DIR = os.path.join(BASE_DIR, "converted_images")

# Configuração de APIs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

# Configurações de aplicação
APP_TITLE = "Extrator de Documentos com IA"
APP_DESCRIPTION = "API para extrair informações de documentos usando modelos de visão computacional"
APP_VERSION = "1.0.0"

# Configurações de limpeza
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "6"))
TEMP_RETENTION_HOURS = int(os.getenv("TEMP_RETENTION_HOURS", "24"))
RESULTS_RETENTION_HOURS = int(os.getenv("RESULTS_RETENTION_HOURS", "72"))

# Configurações de logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = "INFO"

# Criar diretórios necessários
for dir_path in [TEMP_DIR, RESULTS_DIR, CONVERTED_DIR]:
    os.makedirs(dir_path, exist_ok=True)