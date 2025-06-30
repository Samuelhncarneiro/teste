# app/utils/recovery_system.py
import os
import json
import logging
import math
import re
from typing import Dict, Any, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

class ProcessingRecovery:
    """
    Sistema de recuperação para processamento de documentos que lida com
    erros comuns e garante a continuidade do processamento.
    """
    
    @staticmethod
    def sanitize_json_data(data: Any) -> Any:
        """
        Sanitiza dados para garantir que sejam compatíveis com JSON
        
        Args:
            data: Dados a serem sanitizados
            
        Returns:
            Any: Dados sanitizados
        """
        if data is None:
            return None
            
        # Processar dicionários recursivamente
        if isinstance(data, dict):
            return {k: ProcessingRecovery.sanitize_json_data(v) for k, v in data.items()}
            
        # Processar listas recursivamente
        if isinstance(data, list):
            return [ProcessingRecovery.sanitize_json_data(item) for item in data]
            
        # Tratar valores numéricos
        if isinstance(data, (int, float)):
            # Substituir NaN e infinito por None
            if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
                return None
            return data
            
        # Retornar outros tipos sem modificação
        return data
    
    @staticmethod
    def fix_product_prices(
        product: Dict[str, Any], 
        supplier: str = "", 
        default_markup: float = 2.73
    ) -> Dict[str, Any]:
        """
        Corrige preços em um produto, garantindo valores válidos
        
        Args:
            product: Produto a ser corrigido
            supplier: Nome do fornecedor para cálculo de markup
            default_markup: Markup padrão se não for possível determinar pelo fornecedor
            
        Returns:
            Dict: Produto com preços corrigidos
        """
        try:
            from app.data.reference_data import get_supplier_code, get_markup
            
            # Determinar markup
            markup = default_markup
            if supplier:
                supplier_code = get_supplier_code(supplier)
                if supplier_code:
                    supplier_markup = get_markup(supplier_code)
                    if supplier_markup:
                        markup = supplier_markup
        except ImportError:
            # Se não conseguir importar, usar markup padrão
            markup = default_markup
        
        # Processar cada cor
        if "colors" in product and isinstance(product["colors"], list):
            for color in product["colors"]:
                # Verificar preço unitário
                if "unit_price" not in color or color["unit_price"] is None or (
                    isinstance(color["unit_price"], float) and (math.isnan(color["unit_price"]) or math.isinf(color["unit_price"]))
                ):
                    # Usar valor default
                    color["unit_price"] = 0.0
                
                # Verificar preço de venda
                if "sales_price" not in color or color["sales_price"] is None or (
                    isinstance(color["sales_price"], float) and (math.isnan(color["sales_price"]) or math.isinf(color["sales_price"]))
                ):
                    # Calcular baseado no preço unitário
                    color["sales_price"] = round(color["unit_price"] * markup, 2)
                
                # Verificar tamanhos e calcular subtotal
                total_quantity = 0
                if "sizes" in color and isinstance(color["sizes"], list):
                    for size in color["sizes"]:
                        if "quantity" not in size or size["quantity"] is None or (
                            isinstance(size["quantity"], float) and (math.isnan(size["quantity"]) or math.isinf(size["quantity"]))
                        ):
                            size["quantity"] = 0
                        else:
                            # Garantir que é um número inteiro positivo
                            try:
                                qty = float(size["quantity"])
                                if qty > 0:
                                    size["quantity"] = int(qty) if qty.is_integer() else qty
                                    total_quantity += size["quantity"]
                                else:
                                    size["quantity"] = 0
                            except (ValueError, TypeError):
                                size["quantity"] = 0
                
                # Recalcular subtotal baseado nas quantidades
                if "subtotal" not in color or color["subtotal"] is None or (
                    isinstance(color["subtotal"], float) and (math.isnan(color["subtotal"]) or math.isinf(color["subtotal"]))
                ):
                    color["subtotal"] = round(color["unit_price"] * total_quantity, 2)
        
        # Recalcular total_price
        if "total_price" not in product or product["total_price"] is None or (
            isinstance(product["total_price"], float) and (math.isnan(product["total_price"]) or math.isinf(product["total_price"]))
        ):
            subtotals = [
                color.get("subtotal", 0) 
                for color in product.get("colors", []) 
                if color.get("subtotal") is not None and not (
                    isinstance(color.get("subtotal"), float) and 
                    (math.isnan(color.get("subtotal")) or math.isinf(color.get("subtotal")))
                )
            ]
            product["total_price"] = sum(subtotals)
        
        return product
    
    @staticmethod
    def clean_product_name(name: str) -> str:
        """
        Remove números e códigos do nome do produto
        
        Args:
            name: Nome original do produto
            
        Returns:
            str: Nome limpo do produto
        """
        if not name:
            return ""
            
        # Padrão para identificar nomes de produtos (ex: Paddy 10241663 01)
        pattern = r'^([A-Za-z\s]+)(?:\s+\d+.*)?$'
        match = re.match(pattern, name)
        
        if match:
            # Extrair apenas o nome (ex: Paddy)
            return match.group(1).strip()
        
        # Se não conseguir extrair com o padrão, remover todos os números
        cleaned = re.sub(r'\d+', '', name).strip()
        
        # Remover espaços duplos que podem ter ficado
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    @staticmethod
    def format_product_description(product_name: str, color_code: str, size: str) -> str:
        """
        Formata descrição padronizada para produtos
        
        Args:
            product_name: Nome do produto
            color_code: Código da cor
            size: Tamanho
            
        Returns:
            str: Descrição formatada
        """
        # Limpar nome do produto para garantir que está sem números
        clean_name = ProcessingRecovery.clean_product_name(product_name)
        
        # Formato padrão: Nome[COR/TAMANHO]
        return f"{clean_name}[{color_code}/{size}]"
    
    @staticmethod
    def fix_extraction_result(
        extraction_result: Dict[str, Any], 
        supplier: str = ""
    ) -> Dict[str, Any]:
        """
        Corrige um resultado completo de extração para garantir que é válido
        
        Args:
            extraction_result: Resultado da extração
            supplier: Nome do fornecedor para cálculo de markup
            
        Returns:
            Dict: Resultado corrigido
        """
        # Verificar se é um resultado válido
        if not extraction_result or not isinstance(extraction_result, dict):
            logger.warning("Resultado de extração inválido ou vazio")
            return {"products": [], "order_info": {}}
        
        # Garantir que os campos básicos existam
        if "products" not in extraction_result:
            extraction_result["products"] = []
        
        if "order_info" not in extraction_result:
            extraction_result["order_info"] = {}
        
        # Obter informações do contexto
        order_info = extraction_result.get("order_info", {})
        if not supplier:
            supplier = order_info.get("supplier", "")
        
        # Corrigir produtos
        fixed_products = []
        for product in extraction_result.get("products", []):
            # Verificar se é um produto válido
            if not isinstance(product, dict):
                continue
            
            # Limpar nome do produto
            if "name" in product and product["name"]:
                product["name"] = ProcessingRecovery.clean_product_name(product["name"])
            
            # Corrigir preços
            product = ProcessingRecovery.fix_product_prices(product, supplier)
            
            # Verificar se tem cores e tamanhos válidos
            has_valid_items = False
            if "colors" in product and isinstance(product["colors"], list):
                for color in product["colors"]:
                    if "sizes" in color and isinstance(color["sizes"], list) and any(
                        size.get("quantity", 0) > 0 for size in color["sizes"]
                    ):
                        has_valid_items = True
                        break
            
            # Adicionar apenas produtos com items válidos
            if has_valid_items:
                fixed_products.append(product)
        
        # Atualizar lista de produtos
        extraction_result["products"] = fixed_products
        
        # Sanitizar o resultado final
        sanitized_result = ProcessingRecovery.sanitize_json_data(extraction_result)
        
        return sanitized_result
    
    @staticmethod
    def safe_save_json(data: Any, file_path: str) -> bool:
        """
        Salva dados como JSON de forma segura, com sanitização
        
        Args:
            data: Dados a serem salvos
            file_path: Caminho do arquivo para salvar
            
        Returns:
            bool: True se salvou com sucesso, False caso contrário
        """
        try:
            # Sanitizar os dados
            sanitized_data = ProcessingRecovery.sanitize_json_data(data)
            
            # Garantir que o diretório existe
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Salvar o arquivo
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar JSON: {str(e)}")
            
            # Tentativa de recuperação - sanitização mais agressiva
            try:
                # Converter para string e depois voltar para dict
                str_data = json.dumps(data, default=str)
                clean_data = json.loads(str_data)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(clean_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Arquivo JSON salvo após recuperação de emergência: {file_path}")
                return True
            except Exception as e2:
                logger.error(f"Falha na recuperação de emergência: {str(e2)}")
                return False
    
    @staticmethod
    def retry_processing_with_fixes(
        process_func: Callable, 
        max_retries: int = 3,
        **kwargs
    ) -> Any:
        """
        Executa uma função de processamento com tentativas de recuperação em caso de falha
        
        Args:
            process_func: Função de processamento a ser executada
            max_retries: Número máximo de tentativas
            **kwargs: Argumentos para passar para a função de processamento
            
        Returns:
            Any: Resultado da função de processamento ou None em caso de falha
        """
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                # Tentar executar a função normalmente
                result = process_func(**kwargs)
                
                # Se chegou aqui, funcionou - verificar se há valores NaN
                sanitized_result = ProcessingRecovery.sanitize_json_data(result)
                
                # Verificar se a sanitização modificou o resultado
                if sanitized_result != result:
                    logger.warning(f"Resultado sanitizado para remover valores NaN (tentativa {retries+1})")
                    
                    # Se for a primeira tentativa, tentar novamente com o resultado sanitizado
                    if retries == 0:
                        retries += 1
                        continue
                
                # Retornar o resultado sanitizado
                return sanitized_result
                
            except Exception as e:
                last_error = e
                retries += 1
                
                logger.warning(f"Falha na tentativa {retries}/{max_retries}: {str(e)}")
                
                # Tentar recuperar com base no tipo de erro
                if "NaN" in str(e) or "is not valid JSON" in str(e):
                    logger.info("Detectado erro de valores NaN no JSON, aplicando correções...")
                    
                    # Se tiver informações do último resultado, tentar corrigir
                    if "result" in kwargs:
                        kwargs["result"] = ProcessingRecovery.fix_extraction_result(kwargs["result"])
                        logger.info("Resultado corrigido para próxima tentativa")
                
                # Esperar um pouco antes da próxima tentativa
                import time
                time.sleep(1)
        
        # Se chegou aqui, todas as tentativas falharam
        logger.error(f"Todas as {max_retries} tentativas falharam. Último erro: {str(last_error)}")
        return None

# Função auxiliar para integração com o código existente
def apply_recovery_to_extraction_result(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aplica o sistema de recuperação a um resultado de extração
    
    Args:
        extraction_result: Resultado da extração
        
    Returns:
        Dict: Resultado corrigido
    """
    return ProcessingRecovery.fix_extraction_result(extraction_result)

# Função para integração com o pipeline de processamento
def integrate_recovery_system(extraction_module):
    """
    Integra o sistema de recuperação ao módulo de extração
    
    Args:
        extraction_module: Módulo de extração a ser modificado
    """
    original_extract_document = extraction_module.extract_document
    original_post_process = extraction_module._post_process_products
    
    # Substituir método de extração
    async def extract_document_with_recovery(self, *args, **kwargs):
        try:
            # Tentar método original
            result = await original_extract_document(self, *args, **kwargs)
            
            # Aplicar recuperação ao resultado
            if isinstance(result, dict) and "products" in result:
                return apply_recovery_to_extraction_result(result)
            
            return result
        except Exception as e:
            logger.error(f"Erro na extração: {str(e)}")
            
            # Tentar recuperação de emergência
            try:
                # Obter contexto do self se disponível
                context_info = getattr(self, "current_context_info", {})
                supplier = context_info.get("supplier", "")
                
                # Criar resultado mínimo
                min_result = {
                    "products": [],
                    "order_info": context_info
                }
                
                logger.warning("Criando resultado mínimo para recuperação de emergência")
                return min_result
            except:
                # Falha total, retornar erro
                return {"error": str(e), "products": []}
    
    # Substituir método de pós-processamento
    def post_process_with_recovery(self, products, context_info):
        try:
            # Tentar método original
            processed = original_post_process(self, products, context_info)
            
            # Verificar se há produtos válidos
            if not processed:
                logger.warning("Nenhum produto válido após pós-processamento, aplicando recuperação")
                
                # Tentar recuperar
                supplier = context_info.get("supplier", "")
                
                # Criar produtos recuperados
                recovered_products = []
                for product in products:
                    try:
                        fixed_product = ProcessingRecovery.fix_product_prices(product, supplier)
                        recovered_products.append(fixed_product)
                    except:
                        # Ignorar produto problemático
                        continue
                
                return recovered_products
            
            return processed
        except Exception as e:
            logger.error(f"Erro no pós-processamento: {str(e)}")
            
            # Recuperação de emergência
            try:
                # Tentar corrigir cada produto individualmente
                supplier = context_info.get("supplier", "")
                
                recovered_products = []
                for product in products:
                    try:
                        # Corrigir nome
                        if "name" in product:
                            product["name"] = ProcessingRecovery.clean_product_name(product["name"])
                        
                        # Corrigir preços
                        product = ProcessingRecovery.fix_product_prices(product, supplier)
                        
                        # Verificar se tem dados válidos
                        if product.get("colors") and any(len(color.get("sizes", [])) > 0 for color in product["colors"]):
                            recovered_products.append(product)
                    except:
                        # Ignorar produto problemático
                        continue
                
                return recovered_products
            except:
                # Falha total, retornar lista vazia
                return []
    
    # Aplicar as substituições
    extraction_module.extract_document = extract_document_with_recovery
    extraction_module._post_process_products = post_process_with_recovery
    
    logger.info("Sistema de recuperação integrado ao módulo de extração")