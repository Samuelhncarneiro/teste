# app/utils/barcode_generator.py
import logging
from typing import Dict, Any, List, Optional
from app.data.reference_data import (
    COLOR_CODE_MAP, SIZE_MAP, SUPPLIER_CODE_MAP,
    get_color_code, get_size_code, get_supplier_code
)
from app.utils.supplier_utils import match_supplier_name, get_normalized_supplier

logger = logging.getLogger(__name__)

def generate_barcode(
    supplier: str,
    product_counter: int,
    color_code: str,
    size: str,
    season_code: str = "00"
) -> str:
    """
    Gerador de código de barras com tratamentos para casos especiais
    """
    try:
        from app.utils.supplier_utils import get_normalized_supplier
        from app.data.reference_data import get_size_code
        
        # Normalizar fornecedor
        normalized_supplier, supplier_code = get_normalized_supplier(supplier)
        
        # Fallback para casos sem código de fornecedor
        if not supplier_code:
            supplier_code = "00"
        
        # Garantir 2 dígitos para código de fornecedor
        supplier_code = str(supplier_code).zfill(2)
        
        # Calcular contador (3 dígitos: 100 + contador)
        counter_code = str(100 + min(product_counter, 899)).zfill(3)
        
        # Normalizar código de cor
        color_code = str(color_code).zfill(3) if color_code else "001"
        
        # Obter código de tamanho
        size_code = get_size_code(size) or "001"
        size_code = str(size_code).zfill(3)
        
        # Compor código de barras
        barcode = f"{season_code}{supplier_code}{counter_code}{color_code}{size_code}"
        
        return barcode
    
    except Exception as e:
        logging.error(f"Erro ao gerar código de barras: {str(e)}")
        return f"0001100{str(product_counter).zfill(3)}001001"

def normalize_size_value(size: str) -> str:
    if not size:
        return size
    
    # Se for uma string numérica, remover zeros à esquerda
    if size.isdigit():
        return str(int(size))
    
    # Caso contrário, retornar o tamanho original
    return size

def add_barcodes_to_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        product_counters = {}
        
        for product in products:
            material_code = product.get("material_code", "")
            brand = product.get("brand", "")
            
            # Reiniciar contador por produto para garantir numeração única
            product_counters[material_code] = 0
            
            references_with_barcodes = []
            
            # Iterar por todas as cores do produto
            for color in product.get("colors", []):
                color_code = color.get("color_code", "")
                color_name = color.get("color_name", "")
                supplier = color.get("supplier", brand)
                
                # Iterar por todos os tamanhos da cor
                for size_info in color.get("sizes", []):
                    size = size_info.get("size", "")
                    quantity = size_info.get("quantity", 0)
                    
                    if quantity <= 0:
                        continue
                    
                    # Incrementar contador global para o produto
                    product_counters[material_code] += 1
                    counter = product_counters[material_code]
                    
                    # Criar referência com código de barras
                    reference = {
                        "reference": f"{material_code}.{counter}",
                        "counter": counter,
                        "color_code": color_code,
                        "color_name": color_name,
                        "size": size,
                        "quantity": quantity,
                        "barcode": generate_barcode(
                            supplier=supplier,
                            product_counter=counter,
                            color_code=color_code,
                            size=size
                        )
                    }
                    
                    references_with_barcodes.append(reference)
            
            # Substituir referencias do produto
            product["references"] = references_with_barcodes
        
        return products
    
    except Exception as e:
        logging.error(f"Erro ao adicionar códigos de barras: {str(e)}")
        return products

def add_barcodes_to_extraction_result(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adiciona códigos de barras ao resultado da extração
    
    Args:
        extraction_result: Resultado da extração
        
    Returns:
        Dict: Resultado da extração com códigos de barras
    """
    if not extraction_result or "products" not in extraction_result:
        return extraction_result
    
    try:
        # Extrair informações de contexto relevantes
        context = extraction_result.get("context", {})
        global_supplier = context.get("supplier", "")
        global_brand = context.get("brand", "")
        
        # Normalizar fornecedor/marca global
        if global_supplier:
            normalized_supplier, _ = get_normalized_supplier(global_supplier)
            
            if normalized_supplier != global_supplier:
                logger.info(f"Fornecedor global normalizado: '{global_supplier}' → '{normalized_supplier}'")
                extraction_result["context"]["supplier"] = normalized_supplier
        
        # Para cada produto, garantir que tenha informações de fornecedor
        products = extraction_result.get("products", [])
        for product in products:
            # Se o produto não tem uma marca definida, usar a marca global
            if not product.get("brand") and global_brand:
                product["brand"] = global_brand
                logger.info(f"Atribuindo marca global '{global_brand}' ao produto {product.get('material_code', 'N/A')}")
            
            # Para cada cor, verificar se tem um fornecedor
            for color in product.get("colors", []):
                if not color.get("supplier"):
                    # Se não tem fornecedor definido, tentar usar o fornecedor global
                    if global_supplier:
                        color["supplier"] = global_supplier
                        logger.info(f"Atribuindo fornecedor global '{global_supplier}' à cor {color.get('color_code', 'N/A')}")
                    # Se ainda não tem, usar a marca do produto
                    elif product.get("brand"):
                        color["supplier"] = product.get("brand")
                        logger.info(f"Atribuindo marca do produto '{product.get('brand')}' como fornecedor da cor {color.get('color_code', 'N/A')}")
        
        # Adicionar códigos de barras aos produtos
        extraction_result["products"] = add_barcodes_to_products(products)
        
        return extraction_result
    except Exception as e:
        logger.error(f"Erro ao adicionar códigos de barras ao resultado da extração: {str(e)}")
        # Retornar o resultado original em caso de erro
        return extraction_result