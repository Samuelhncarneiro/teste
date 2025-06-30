# app/utils/integration.py
"""
Utilitário para integrar o sistema de recuperação na aplicação existente.
"""
import logging
import json
import math
from typing import Dict, Any, List, Optional
from app.utils.recovery_system import integrate_recovery_system
from app.extractors.gemini_extractor import GeminiExtractor
import pandas as pd
import re
import app.main

logger = logging.getLogger(__name__)

def setup_recovery_system():
    """
    Configura o sistema de recuperação na aplicação.
    Deve ser chamado durante a inicialização.
    """
    try:
        integrate_recovery_system(GeminiExtractor)
        
        logger.info("Sistema de recuperação configurado com sucesso")
        return True
    except ImportError as e:
        logger.error(f"Erro ao configurar sistema de recuperação: {str(e)}")
        return False

def patch_json_encoder():
    """
    Substitui o encoder JSON padrão para lidar com valores NaN e Infinity.
    """
    original_default = json.JSONEncoder.default
    
    def patched_default(self, obj):
        """Encoder personalizado que trata valores NaN e Infinity"""
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return original_default(self, obj)
    
    # Aplicar o patch
    json.JSONEncoder.default = patched_default
    logger.info("Encoder JSON modificado para tratar valores NaN e Infinity")

def monkey_patch_dataframe_conversion():
    """
    Aplica monkey patch à função de conversão para DataFrame
    para garantir sanitização de valores.
    """
    try:
        # Definir função de sanitização básica
        def sanitize_value(value):
            """Sanitiza um valor para evitar problemas com NaN"""
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return 0.0
            return value
        
        # Função wrapper para sanitizar valores
        def sanitized_create_dataframe(*args, **kwargs):
            """Wrapper que garante sanitização de valores na criação do DataFrame"""
            # Chamar função original
            df = create_dataframe_from_extraction(*args, **kwargs)
            
            # Sanitizar valores NaN
            for col in df.columns:
                if df[col].dtype == 'float64' or df[col].dtype == 'float32':
                    df[col] = df[col].fillna(0.0)
            
            return df
        
        # Aplicar o patch
        app.main.create_dataframe_from_extraction = sanitized_create_dataframe
        logger.info("Função de conversão para DataFrame modificada para sanitizar valores")
        
        return True
    except (ImportError, AttributeError) as e:
        logger.error(f"Erro ao aplicar patch na função de DataFrame: {str(e)}")
        return False

def initialize_recovery_features():
    """
    Inicializa todas as funcionalidades de recuperação.
    Deve ser chamado na inicialização da aplicação.
    """
    patch_json_encoder()
    monkey_patch_dataframe_conversion()
    
    recovery_configured = setup_recovery_system()
    
    if recovery_configured:
        logger.info("Sistema de recuperação inicializado com sucesso")
    else:
        logger.warning("Inicialização parcial do sistema de recuperação")
    
    return recovery_configured