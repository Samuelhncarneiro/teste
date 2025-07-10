# app/extractors/recovery_integration.py
"""
Integração dos sistemas de recuperação no GeminiExtractor existente
Adiciona capacidades de recuperação sem alterar a estrutura principal
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional
from PIL import Image
import google.generativeai as genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

def robust_json_parse(response_text: str, page_number: int) -> Dict[str, Any]:
    """
    Parser JSON robusto que substitui o método original
    Múltiplas estratégias de recuperação para JSON malformado
    """
    logger.debug(f"Aplicando parse robusto para página {page_number}")
    
    # Estratégia 1: Bloco de código JSON
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, response_text)
    
    if matches:
        for match in matches:
            try:
                cleaned_json = _clean_json_string(match)
                result = json.loads(cleaned_json)
                if isinstance(result, dict) and "products" in result:
                    logger.info(f"✅ JSON extraído de bloco de código (página {page_number})")
                    return _sanitize_result(result)
            except json.JSONDecodeError:
                continue
    
    # Estratégia 2: JSON completo no texto
    json_pattern = r'(\{[\s\S]*"products"[\s\S]*\})'
    matches = re.findall(json_pattern, response_text)
    
    if matches:
        for match in matches:
            try:
                cleaned_json = _clean_json_string(match)
                result = json.loads(cleaned_json)
                if isinstance(result, dict):
                    logger.info(f"✅ JSON extraído do texto (página {page_number})")
                    return _sanitize_result(result)
            except json.JSONDecodeError:
                continue
    
    # Estratégia 3: Recuperação parcial
    try:
        # Extrair array de produtos
        product_pattern = r'"products"\s*:\s*(\[[\s\S]*?\])'
        product_matches = re.findall(product_pattern, response_text)
        
        if product_matches:
            products_json = product_matches[0]
            products = json.loads(products_json)
            
            if isinstance(products, list):
                logger.warning(f"⚠️ Recuperação parcial aplicada (página {page_number})")
                return _sanitize_result({
                    "products": products,
                    "order_info": {},
                    "_partial_recovery": True
                })
    except:
        pass
    
    # Estratégia 4: Recuperação de emergência por regex
    material_codes = re.findall(r'"material_code"\s*:\s*"([^"]+)"', response_text)
    
    if material_codes:
        products = []
        for code in material_codes[:10]:  # Máximo 10 na emergência
            products.append({
                "material_code": code,
                "name": f"Produto {code}",
                "colors": [{
                    "color_code": "001",
                    "color_name": "Padrão", 
                    "sizes": [{"size": "UN", "quantity": 1}],
                "unit_price": 000.00,
                "sales_price": 000.00,
                "subtotal": 000.00
                }]
            })
        
        logger.warning(f"⚠️ Recuperação de emergência: {len(products)} códigos (página {page_number})")
        return _sanitize_result({
            "products": products,
            "order_info": {},
            "_emergency_recovery": True
        })
    
    # Se chegou aqui, falhou tudo
    raise ValueError(f"Nenhuma estratégia de parse funcionou para página {page_number}")


def _clean_json_string(json_str: str) -> str:
    """Limpa string JSON para melhorar chances de parse"""
    # Remover comentários
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # Corrigir aspas simples para duplas
    json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
    json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
    
    # Remover vírgulas extras
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Corrigir valores problemáticos
    json_str = re.sub(r'\bundefined\b', 'null', json_str, flags=re.IGNORECASE)
    json_str = re.sub(r'\bNaN\b', '0', json_str, flags=re.IGNORECASE)
    
    return json_str.strip()


def _sanitize_result(result: Dict[str, Any]) -> Dict[str, Any]:

    if "products" not in result:
        result["products"] = []
    
    if "order_info" not in result:
        result["order_info"] = {}
    
    if "products" in result:
        preserved = 0
        sanitized = []

        for product in result["products"]:
            if not isinstance(product, dict):
                continue

            if not product.get("material_code"):
                logger.warning("Produto sem código - mantido para revisão")
                sanitized.append(product)
                continue

            # Garantir que pelo menos uma cor ou tamanho válido exista
            has_valid_color = False
            for color in product.get("colors", []):
                if isinstance(color, dict) and color.get("color_code"):
                    has_valid_color = True
                    break

            if not has_valid_color:
                logger.warning(f"Produto {product['material_code']} sem cores válidas - mantido com aviso")
            
            sanitized.append(product)

        result["products"] = sanitized
    return result

async def recover_failed_page(api_key: str, image_path: str, original_error: str, page_number: int, context: str) -> Dict[str, Any]:
    """
    Tenta recuperar dados de uma página que falhou
    """
    logger.info(f"🔄 Tentando recuperar página {page_number}")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    # Estratégias de recuperação baseadas no erro
    if "json" in original_error.lower():
        strategies = [
            ("JSON Simplificado", _simplified_extraction),
            ("Estrutura Fixa", _fixed_structure_extraction),
        ]
    else:
        strategies = [
            ("Extração Conservadora", _conservative_extraction),
            ("Dados Mínimos", _minimal_extraction),
        ]
    
    image = Image.open(image_path)
    
    for strategy_name, strategy_func in strategies:
        try:
            logger.info(f"🎯 Tentando estratégia: {strategy_name}")
            
            result = await strategy_func(model, image, page_number, context)
            
            if result and result.get("products"):
                logger.info(f"✅ Recuperação bem-sucedida: {strategy_name}")
                result["recovery_info"] = {
                    "strategy": strategy_name,
                    "original_error": original_error
                }
                return result
                
        except Exception as e:
            logger.debug(f"Estratégia {strategy_name} falhou: {str(e)}")
            continue
    
    # Se tudo falhou, retornar resultado vazio
    logger.warning(f"❌ Falha na recuperação da página {page_number}")
    return {
        "products": [],
        "order_info": {},
        "error": f"Recuperação falhou: {original_error}",
        "page_number": page_number
    }


async def _simplified_extraction(model, image, page_number, context):
    """Extração com JSON muito simples"""
    prompt = f"""
    RECUPERAÇÃO SIMPLES - PÁGINA {page_number}
    
    Use APENAS estrutura muito básica. Se não tiver certeza, não incluir.
    
    {{
        "products": [
            {{
                "material_code": "CODIGO_VISIVEL",
                "name": "Nome Básico",
                "colors": [
                    {{
                        "color_code": "001",
                        "color_name": "Cor",
                        "sizes": [
                            {{"size": "M", "quantity": 1}}
                        ]
                    }}
                ]
            }}
        ]
    }}
    
    Extrair apenas o que está CLARAMENTE visível.
    """
    
    response = model.generate_content([prompt, image])
    return robust_json_parse(response.text, page_number)


async def _fixed_structure_extraction(model, image, page_number, context):
    """Extração com estrutura pré-definida"""
    prompt = f"""
    RECUPERAÇÃO ESTRUTURADA - PÁGINA {page_number}
    
    Use exatamente esta estrutura, substituindo pelos dados reais:
    
    {{
        "products": [
            {{
                "material_code": "SUBSTITUA_PELO_CODIGO_REAL",
                "name": "SUBSTITUA_PELO_NOME_REAL", 
                "colors": [
                    {{
                        "color_code": "001",
                        "color_name": "SUBSTITUA_PELA_COR_REAL",
                        "sizes": [
                            {{"size": "S", "quantity": 1}},
                            {{"size": "M", "quantity": 1}}
                        ]
                    }}
                ]
            }}
        ]
    }}
    """
    
    response = model.generate_content([prompt, image])
    return robust_json_parse(response.text, page_number)


async def _conservative_extraction(model, image, page_number, context):
    """Extração conservadora - apenas dados essenciais"""
    prompt = f"""
    EXTRAÇÃO CONSERVADORA - PÁGINA {page_number}
    
    Extrair APENAS dados com 100% de certeza:
    - Códigos de material claramente visíveis
    - Nomes básicos de produtos
    - Uma cor simples se visível
    
    Se não tiver certeza absoluta, não incluir.
    
    {{
        "products": [
            {{
                "material_code": "CERTEZA_ABSOLUTA",
                "name": "Nome Conservador",
                "colors": [
                    {{
                        "color_code": "001", 
                        "color_name": "Padrão",
                        "sizes": [
                            {{"size": "UN", "quantity": 1}}
                        ]
                    }}
                ]
            }}
        ]
    }}
    """
    
    response = model.generate_content([prompt, image])
    return robust_json_parse(response.text, page_number)


async def _minimal_extraction(model, image, page_number, context):
    """Extração mínima - pelo menos algo"""
    prompt = f"""
    EXTRAÇÃO MÍNIMA - PÁGINA {page_number}
    
    Extrair o mínimo possível para não perder a página.
    Mesmo que seja apenas 1 produto com dados básicos.
    
    {{
        "products": [
            {{
                "material_code": "MIN{page_number:03d}",
                "name": "Produto Página {page_number}",
                "colors": [
                    {{
                        "color_code": "999",
                        "color_name": "N/A",
                        "sizes": [
                            {{"size": "UN", "quantity": 1}}
                        ]
                    }}
                ]
            }}
        ]
    }}
    """
    
    response = model.generate_content([prompt, image])
    return robust_json_parse(response.text, page_number)

def patch_gemini_extractor():
    """
    Aplica patches no GeminiExtractor existente para adicionar recuperação
    """
    from app.extractors.gemini_extractor import GeminiExtractor
    from app.extractors.extraction_agent import ExtractionAgent
    
    # 1. PATCH NO EXTRACTION AGENT - Parser JSON robusto
    original_extract_and_clean = ExtractionAgent._extract_and_clean_json
    
    def enhanced_extract_and_clean_json(self, response_text: str, page_number: int) -> Dict[str, Any]:
        """Versão melhorada que usa o parser robusto"""
        try:
            # Tentar parser robusto primeiro
            return robust_json_parse(response_text, page_number)
        except Exception as e:
            logger.warning(f"Parser robusto falhou, tentando original: {str(e)}")
            try:
                return original_extract_and_clean(self, response_text, page_number)
            except Exception as e2:
                logger.error(f"Ambos os parsers falharam para página {page_number}")
                raise e2
    
    # Aplicar patch
    ExtractionAgent._extract_and_clean_json = enhanced_extract_and_clean_json
    
    # 2. PATCH NO PROCESS_PAGE - Sistema de recuperação
    original_process_page = GeminiExtractor.process_page
    
    async def enhanced_process_page(self, image_path, context, page_number, total_pages, previous_result=None):
        """Versão melhorada com sistema de recuperação"""
        try:
            # Tentar processamento normal
            result = await original_process_page(self, image_path, context, page_number, total_pages, previous_result)
            
            # Se deu certo e tem produtos, retornar
            if result and (result.get("products") or not result.get("error")):
                return result
            
            # Se chegou aqui, algo deu errado
            raise Exception(result.get("error", "Erro desconhecido no processamento"))
            
        except Exception as e:
            logger.warning(f"⚠️ Falha na página {page_number}: {str(e)}")
            
            # Tentar recuperação apenas se não for a primeira página ou se o erro for recuperável
            if page_number > 1 or "json" in str(e).lower():
                try:
                    recovery_result = await recover_failed_page(
                        api_key=self.api_key,
                        image_path=image_path,
                        original_error=str(e),
                        page_number=page_number,
                        context=context
                    )
                    
                    if recovery_result.get("products"):
                        logger.info(f"✅ Página {page_number} recuperada")
                        return recovery_result
                    
                except Exception as recovery_error:
                    logger.error(f"❌ Falha na recuperação: {str(recovery_error)}")
            
            # Para primeira página, sempre falhar
            if page_number == 1:
                raise e
            
            # Para outras páginas, retornar resultado vazio mas válido
            logger.warning(f"⚠️ Página {page_number} ignorada")
            return {
                "products": [],
                "order_info": {},
                "error": str(e),
                "page_skipped": True
            }
    
    # Aplicar patch
    GeminiExtractor.process_page = enhanced_process_page
    
    # 3. MELHORAR LOGGING NO EXTRACT_DOCUMENT
    original_extract_document = GeminiExtractor.extract_document
    
    async def enhanced_extract_document(self, document_path, job_id, jobs_store, update_progress_callback):
        """Versão com logging melhorado de falhas"""
        result = await original_extract_document(self, document_path, job_id, jobs_store, update_progress_callback)
        
        # Adicionar estatísticas de recuperação aos metadados
        if hasattr(self, 'page_results_history'):
            failed_pages = []
            recovered_pages = []
            
            for i, page_result in enumerate(self.page_results_history, 1):
                if page_result.get("error"):
                    failed_pages.append(i)
                if page_result.get("recovery_info"):
                    recovered_pages.append(i)
            
            if "_metadata" not in result:
                result["_metadata"] = {}
            
            result["_metadata"].update({
                "failed_pages": failed_pages,
                "recovered_pages": recovered_pages,
                "recovery_system_enabled": True
            })
            
            if failed_pages:
                logger.warning(f"⚠️ Páginas com falha: {failed_pages}")
            if recovered_pages:
                logger.info(f"✅ Páginas recuperadas: {recovered_pages}")
        
        return result
    
    # Aplicar patch
    GeminiExtractor.extract_document = enhanced_extract_document
    
    logger.info("✅ Sistema de recuperação integrado no GeminiExtractor existente")

def initialize_recovery_system():
    """
    Inicializa todo o sistema de recuperação
    Deve ser chamado na inicialização da aplicação
    """
    try:
        patch_gemini_extractor()
        logger.info("🛡️ Sistema de recuperação de páginas ativado")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar sistema de recuperação: {str(e)}")
        return False