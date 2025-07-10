# app/utils/json_utils.py
import json
import math
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

def is_json_serializable(obj: Any) -> bool:
    """
    Verifica se um objeto é serializável para JSON
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def sanitize_for_json(
    obj: Any, 
    default_number: float = 0.0,
    default_str: str = "",
    max_depth: int = 100,
    current_depth: int = 0
) -> Any:
    """
    Sanitiza recursivamente um objeto para garantir que seja serializável para JSON.
    Substitui valores problemáticos como NaN e Infinity por defaults seguros.
    """
    if current_depth > max_depth:
        logger.warning(f"Profundidade máxima de recursão atingida ({max_depth})")
        return None
    
    if obj is None:
        return None
    
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            logger.debug(f"Valor numérico inválido (NaN/Infinity) substituído por {default_number}")
            return default_number
        return obj
    
    if isinstance(obj, str):
        if not obj or not is_json_serializable(obj):
            return default_str
        return obj
    
    if isinstance(obj, dict):
        return {
            k: sanitize_for_json(
                v, 
                default_number=default_number,
                default_str=default_str,
                max_depth=max_depth,
                current_depth=current_depth + 1
            ) 
            for k, v in obj.items()
        }
    
    if isinstance(obj, (list, tuple)):
        return [
            sanitize_for_json(
                item, 
                default_number=default_number,
                default_str=default_str,
                max_depth=max_depth,
                current_depth=current_depth + 1
            ) 
            for item in obj
        ]
    
    if is_json_serializable(obj):
        return obj
    
    try:
        return str(obj)
    except:
        logger.warning(f"Objeto não serializável do tipo {type(obj)} substituído por None")
        return None

def safe_json_dump(obj: Any, file_path: str, **kwargs) -> bool:
    """
    Salva um objeto como JSON de forma segura, garantindo sanitização prévia
    """
    try:
        sanitized_obj = sanitize_for_json(obj)
        
        if 'indent' not in kwargs:
            kwargs['indent'] = 2
        if 'ensure_ascii' not in kwargs:
            kwargs['ensure_ascii'] = False
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sanitized_obj, f, **kwargs)
        
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar JSON: {str(e)}")
        
        try:
            logger.warning("Tentando recuperação com sanitização agressiva")
            sanitized_obj = sanitize_for_json(obj, default_number=0.0, default_str="")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_obj, f, **kwargs)
            
            return True
        except Exception as e2:
            logger.error(f"Falha na recuperação: {str(e2)}")
            return False

def fix_nan_in_products(products: List[Dict[str, Any]], markup: float = 2.73) -> List[Dict[str, Any]]:

    if not products:
        logger.warning("fix_nan_in_products recebeu lista vazia")
        return []
    
    logger.debug(f"fix_nan_in_products: processando {len(products)} produtos")
    fixed_products = []
    
    for idx, product in enumerate(products):
        if not isinstance(product, dict):
            logger.warning(f"Produto {idx} não é dict: {type(product)}")
            continue
        
        # Criar cópia do produto para não modificar o original
        fixed_product = product.copy()
        
        # Processar cores se existirem
        if "colors" in fixed_product and isinstance(fixed_product["colors"], list):
            fixed_colors = []
            
            for color_idx, color in enumerate(fixed_product["colors"]):
                if not isinstance(color, dict):
                    logger.warning(f"Cor {color_idx} do produto {idx} não é dict")
                    continue
                
                # Criar cópia da cor
                fixed_color = color.copy()
                
                # Corrigir unit_price
                if "unit_price" not in fixed_color or fixed_color["unit_price"] is None or (
                    isinstance(fixed_color["unit_price"], float) and math.isnan(fixed_color["unit_price"])
                ):
                    fixed_color["unit_price"] = 0.0
                
                # Corrigir sales_price
                if "sales_price" not in fixed_color or fixed_color["sales_price"] is None or (
                    isinstance(fixed_color["sales_price"], float) and math.isnan(fixed_color["sales_price"])
                ):
                    fixed_color["sales_price"] = round(fixed_color["unit_price"] * markup, 2)
                
                # Corrigir subtotal
                if "subtotal" not in fixed_color or fixed_color["subtotal"] is None or (
                    isinstance(fixed_color["subtotal"], float) and math.isnan(fixed_color["subtotal"])
                ):
                    # Calcular subtotal baseado nas quantidades
                    total_quantity = 0
                    if "sizes" in fixed_color and isinstance(fixed_color["sizes"], list):
                        for size in fixed_color["sizes"]:
                            if isinstance(size, dict) and "quantity" in size:
                                qty = size.get("quantity", 0)
                                if qty and not (isinstance(qty, float) and math.isnan(qty)):
                                    total_quantity += qty
                    
                    fixed_color["subtotal"] = round(fixed_color["unit_price"] * total_quantity, 2)
                
                # Processar tamanhos
                if "sizes" in fixed_color and isinstance(fixed_color["sizes"], list):
                    fixed_sizes = []
                    for size in fixed_color["sizes"]:
                        if isinstance(size, dict):
                            fixed_size = size.copy()
                            # Garantir que quantity não é NaN
                            if "quantity" in fixed_size:
                                qty = fixed_size["quantity"]
                                if qty is None or (isinstance(qty, float) and math.isnan(qty)):
                                    fixed_size["quantity"] = 0
                            fixed_sizes.append(fixed_size)
                    fixed_color["sizes"] = fixed_sizes
                
                fixed_colors.append(fixed_color)
            
            fixed_product["colors"] = fixed_colors
        
        # Corrigir total_price do produto
        if "total_price" not in fixed_product or fixed_product["total_price"] is None or (
            isinstance(fixed_product["total_price"], float) and math.isnan(fixed_product["total_price"])
        ):
            # Calcular total baseado nos subtotais das cores
            total = 0.0
            if "colors" in fixed_product:
                for color in fixed_product["colors"]:
                    if isinstance(color, dict) and "subtotal" in color:
                        subtotal = color["subtotal"]
                        if subtotal and not (isinstance(subtotal, float) and math.isnan(subtotal)):
                            total += subtotal
            fixed_product["total_price"] = round(total, 2)
        
        # Adicionar produto fixado
        fixed_products.append(fixed_product)
    
    logger.debug(f"fix_nan_in_products: retornando {len(fixed_products)} produtos corrigidos")
    
    return fixed_products