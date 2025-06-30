# app/utils/supplier_assignment.py
import logging
from typing import Dict, Any, List, Optional, Tuple
from app.utils.supplier_utils import match_supplier_name, get_supplier_info
from app.data.reference_data import SUPPLIER_MAP, get_supplier_code, get_markup

logger = logging.getLogger(__name__)

def determine_best_supplier(context_info: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[float]]:
    supplier_from_context = context_info.get("supplier", "")
    brand_from_context = context_info.get("brand", "")
    
    logger.info(f"Determinando fornecedor do documento - Supplier: '{supplier_from_context}', Brand: '{brand_from_context}'")
    
    candidates = []
    
    # 1. Adicionar supplier do contexto se existir
    if supplier_from_context and supplier_from_context not in ["", "Fornecedor não identificado"]:
        candidates.append(("supplier_context", supplier_from_context))
    
    # 2. Adicionar brand do contexto se existir e for diferente do supplier
    if (brand_from_context and 
        brand_from_context not in ["", "Marca não identificada"] and 
        brand_from_context != supplier_from_context):
        candidates.append(("brand_context", brand_from_context))
    
    if not candidates:
        logger.warning("Nenhum candidato a fornecedor encontrado no contexto")
        return "Fornecedor não identificado", None, None
    
    # 3. Avaliar cada candidato usando as funções existentes
    best_match = None
    best_supplier_code = None
    best_score = 0.0
    best_source = None
    
    for source, candidate in candidates:
        # Usar a função existente de correspondência
        matched_supplier = match_supplier_name(candidate)
        
        # Verificar se o match resultou em um fornecedor conhecido
        supplier_code = get_supplier_code(matched_supplier)
        
        if supplier_code and matched_supplier in SUPPLIER_MAP.values():
            # Calcular score baseado na qualidade do match
            if matched_supplier == candidate:
                score = 1.0  # Match exato
            elif matched_supplier in SUPPLIER_MAP.values():
                score = 0.8  # Match encontrado no reference_data
            else:
                score = 0.3  # Match fraco
            
            # Dar preferência ao supplier do contexto sobre a marca
            if source == "supplier_context":
                score += 0.1
            
            if score > best_score:
                best_match = matched_supplier
                best_supplier_code = supplier_code
                best_score = score
                best_source = source
                
            logger.info(f"Candidato '{candidate}' ({source}) -> '{matched_supplier}' (score: {score:.2f})")
        else:
            logger.info(f"Candidato '{candidate}' ({source}) -> '{matched_supplier}' (não encontrado no reference_data)")
    
    # 4. Determinar o fornecedor final
    if best_match:
        final_supplier = best_match
        supplier_code = best_supplier_code
        markup = get_markup(supplier_code) if supplier_code else None
        
        logger.info(f"Fornecedor final determinado: '{final_supplier}' (código: {supplier_code}, markup: {markup}) - fonte: {best_source}")
    else:
        final_supplier = candidates[0][1]
        supplier_code = None
        markup = None
        
        logger.warning(f"Nenhum fornecedor encontrado no reference_data, usando: '{final_supplier}'")
    
    return final_supplier, supplier_code, markup or 2.73

def assign_supplier_to_products(products: List[Dict[str, Any]], 
                               supplier_name: str, 
                               markup: float) -> List[Dict[str, Any]]:
    if not products:
        return products
    
    logger.info(f"Atribuindo fornecedor '{supplier_name}' a {len(products)} produtos")
    
    for product in products:
        product["supplier"] = supplier_name

        for color in product.get("colors", []):
            color["supplier"] = supplier_name
            
            if not color.get("sales_price") and color.get("unit_price"):
                color["sales_price"] = round(color.get("unit_price", 0) * markup, 2)
                
            if color.get("unit_price") and color.get("sizes"):
                total_quantity = sum(size.get("quantity", 0) for size in color["sizes"])
                if not color.get("subtotal") and total_quantity > 0:
                    color["subtotal"] = round(color["unit_price"] * total_quantity, 2)
    
        for reference in product.get("references", []):
                reference["supplier"] = supplier_name
        
    logger.info(f"Fornecedor atribuído com sucesso a todos os produtos")
    return products


