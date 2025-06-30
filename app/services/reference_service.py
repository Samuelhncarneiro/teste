# app/services/reference_service.py
import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ReferenceService:
    """Serviço para geração de referências de produtos"""
    
    def __init__(self):
        """Inicializa o serviço de referências"""
        pass
    
    def generate_references(self, extraction_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Gera referências sequenciais para produtos com base no código do material.
        
        Args:
            extraction_result: Resultado da extração do documento
            
        Returns:
            List[Dict]: Lista de produtos com referências padronizadas
        """
        if not extraction_result or "products" not in extraction_result:
            logger.warning("Resultado de extração vazio ou inválido")
            return []
        
        # Lista para armazenar todos os itens processados
        processed_items = []
        
        # Dicionário para rastrear contadores por referência base (material_code)
        ref_counters = {}
        
        # Informações do pedido
        order_info = extraction_result.get("order_info", {})
        supplier = order_info.get("supplier", "")
        brand = order_info.get("brand", "")
        order_number = order_info.get("order_number", "")
        order_date = order_info.get("date", "")
        season = order_info.get("season", "")
        
        # Processar cada produto
        for product in extraction_result.get("products", []):
            # Extrair informações do produto
            product_name = product.get("name", "")
            material_code = product.get("material_code", "")
            category = product.get("category", "")
            model = product.get("model", "")
            composition = product.get("composition", "")
            product_brand = product.get("brand", brand)
            
            # Inicializar contador para este código de material se não existir
            if material_code not in ref_counters:
                ref_counters[material_code] = 0
            
            # Para cada cor do produto
            for color in product.get("colors", []):
                color_code = color.get("color_code", "")
                color_name = color.get("color_name", "")
                unit_price = color.get("unit_price", 0)
                sales_price = color.get("sales_price", 0)
                
                # Para cada tamanho da cor
                for size_info in color.get("sizes", []):
                    size = size_info.get("size", "")
                    quantity = size_info.get("quantity", 0)
                    
                    if not size or quantity <= 0:
                        continue  # Pular tamanhos inválidos ou sem quantidade
                    
                    # Incrementar contador para este código de material
                    ref_counters[material_code] += 1
                    counter = ref_counters[material_code]
                    
                    # Formar a referência completa no formato material_code.counter
                    reference = f"{material_code}.{counter}"
                    
                    # Formar descrição do produto
                    description = f"{product_name} - {model} [{color_name}/{size}]"
                    
                    # Criar item processado
                    processed_item = {
                        "Referência Base": material_code,
                        "Contador": counter,
                        "Referência": reference,
                        "Nome": product_name,
                        "Modelo": model,
                        "Categoria": category,
                        "Cor-Código": color_code,
                        "Cor-Nome": color_name,
                        "Tamanho": size,
                        "Descrição": description,
                        "Composição": composition,
                        "Quantidade": quantity,
                        "Preço Custo": unit_price,
                        "Preço de Venda": sales_price,
                        "Marca": product_brand if product_brand else brand,
                        "Fornecedor": supplier,
                        "Pedido": order_number,
                        "Data": order_date,
                        "Temporada": season
                    }
                    
                    processed_items.append(processed_item)
        
        logger.info(f"Geradas {len(processed_items)} referências de produtos.")
        return processed_items
    
    def export_to_excel(
        self, 
        extraction_result: Dict[str, Any], 
        output_path: str
    ) -> str:
        """
        Exporta os produtos com referências geradas para um arquivo Excel
        
        Args:
            extraction_result: Resultado da extração
            output_path: Caminho do arquivo de saída
            
        Returns:
            str: Caminho do arquivo Excel gerado
        """
        # Gerar referências
        processed_items = self.generate_references(extraction_result)
        
        # Criar diretório de saída se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Criar DataFrame
        df = pd.DataFrame(processed_items)
        
        # Reordenar colunas para melhor visualização
        column_order = [
            "Referência", "Referência Base", "Contador", "Nome", "Modelo", "Categoria",
            "Cor-Código", "Cor-Nome", "Tamanho", "Quantidade", "Preço Custo", 
            "Preço de Venda", "Composição", "Descrição", "Marca", "Fornecedor",
            "Pedido", "Data", "Temporada"
        ]
        
        # Filtrar apenas colunas existentes
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        # Salvar como Excel
        df.to_excel(output_path, index=False)
        
        logger.info(f"Exportado para Excel: {output_path}")
        return output_path
    
    def export_to_json(
        self, 
        extraction_result: Dict[str, Any], 
        output_path: str
    ) -> str:
        """
        Exporta os produtos com referências geradas para um arquivo JSON
        
        Args:
            extraction_result: Resultado da extração
            output_path: Caminho do arquivo de saída
            
        Returns:
            str: Caminho do arquivo JSON gerado
        """
        # Gerar referências
        processed_items = self.generate_references(extraction_result)
        
        # Criar diretório de saída se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Salvar como JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_items, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exportado para JSON: {output_path}")
        return output_path
    
    def process_job_result(
        self, 
        job_result: Dict[str, Any], 
        output_path: str, 
        format: str = "excel"
    ) -> str:
        """
        Processa o resultado completo de um job e exporta para o formato especificado
        
        Args:
            job_result: Resultado completo do job
            output_path: Caminho do arquivo de saída
            format: Formato de saída ("excel" ou "json")
            
        Returns:
            str: Caminho do arquivo gerado
        """
        # Extrair o resultado do modelo Gemini do job completo
        if "model_results" in job_result and "gemini" in job_result["model_results"]:
            extraction_result = job_result["model_results"]["gemini"].get("result", {})
        else:
            extraction_result = job_result  # Assumir que o próprio input já é o extraction_result
        
        # Exportar no formato especificado
        if format.lower() == "json":
            return self.export_to_json(extraction_result, output_path)
        else:
            return self.export_to_excel(extraction_result, output_path)

# Instância global do serviço de referências
reference_service = None

def get_reference_service():
    """
    Obtém a instância global do serviço de referências
    
    Returns:
        ReferenceService: Instância do serviço
    """
    global reference_service
    
    if reference_service is None:
        reference_service = ReferenceService()
        
    return reference_service