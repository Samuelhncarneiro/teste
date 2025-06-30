# app/utils/category_mapper.py
import re
import logging
from difflib import get_close_matches
from typing import Dict, List, Optional

from app.data.reference_data import CATEGORIES, get_category

logger = logging.getLogger(__name__)

# Mapeamento de categorias em inglês para português
ENGLISH_TO_PORTUGUESE = {
    # Camisas e similares
    "SHIRT": "CAMISAS",
    "SHIRTS": "CAMISAS",
    "CASUAL SHIRT": "CAMISAS",
    "DRESS SHIRT": "CAMISAS",
    "FORMAL SHIRT": "CAMISAS",
    
    # Casacos e similares
    "COAT": "CASACOS",
    "COATS": "CASACOS",
    "JACKET": "CASACOS",
    "JACKETS": "CASACOS",
    "OVERCOAT": "CASACOS",
    "RAINCOAT": "CASACOS",
    
    # Blusões e parkas
    "PARKA": "BLUSÕES E PARKAS",
    "PARKAS": "BLUSÕES E PARKAS",
    "WIND JACKET": "BLUSÕES E PARKAS",
    "WINDBREAKER": "BLUSÕES E PARKAS",
    
    # Vestidos
    "DRESS": "VESTIDOS",
    "DRESSES": "VESTIDOS",
    "EVENING DRESS": "VESTIDOS",
    "COCKTAIL DRESS": "VESTIDOS",
    
    # Blusas
    "BLOUSE": "BLUSAS",
    "BLOUSES": "BLUSAS",
    "TOP": "BLUSAS",
    "TOPS": "BLUSAS",
    
    # Calças
    "PANTS": "CALÇAS",
    "TROUSERS": "CALÇAS",
    "CHINOS": "CALÇAS",
    "SLACKS": "CALÇAS",
    
    # Malhas
    "KNITWEAR": "MALHAS",
    "KNIT": "MALHAS",
    "PULLOVER": "MALHAS",
    "SWEATER": "MALHAS",
    "CARDIGAN": "MALHAS",
    
    # Saias
    "SKIRT": "SAIAS",
    "SKIRTS": "SAIAS",
    "MIDI SKIRT": "SAIAS",
    "MAXI SKIRT": "SAIAS",
    
    # T-shirts
    "T-SHIRT": "T-SHIRTS",
    "T SHIRT": "T-SHIRTS",
    "TEE": "T-SHIRTS",
    "TSHIRT": "T-SHIRTS",
    
    # Polos
    "POLO": "POLOS",
    "POLO SHIRT": "POLOS",
    "JERSEY": "POLOS",
    "JERSEYS": "POLOS",
    
    # Jeans
    "JEAN": "JEANS",
    "DENIM": "JEANS",
    "DENIM PANTS": "JEANS",
    "JEANS PANTS": "JEANS",
    
    # Sweatshirts
    "SWEATSHIRT": "SWEATSHIRTS",
    "HOODIE": "SWEATSHIRTS",
    "HOODED SWEAT": "SWEATSHIRTS",
    "SWEAT": "SWEATSHIRTS",
    
    # Blazers e fatos
    "BLAZER": "BLAZERS E FATOS",
    "SUIT": "BLAZERS E FATOS",
    "FORMAL SUIT": "BLAZERS E FATOS",
    "TUXEDO": "BLAZERS E FATOS",
    
    # Calçado
    "SHOES": "CALÇADO",
    "FOOTWEAR": "CALÇADO",
    "BOOTS": "CALÇADO",
    "SNEAKERS": "CALÇADO",
    "LOAFERS": "CALÇADO",
    
    # Acessórios
    "ACCESSORIES": "ACESSÓRIOS",
    "BELT": "ACESSÓRIOS",
    "TIE": "ACESSÓRIOS",
    "SCARF": "ACESSÓRIOS",
    "HAT": "ACESSÓRIOS",
    "BAG": "ACESSÓRIOS",
    "WALLET": "ACESSÓRIOS",
    "JEWELRY": "ACESSÓRIOS",
    "WATCH": "ACESSÓRIOS",
    "SUNGLASSES": "ACESSÓRIOS",
    "ACCESSORY": "ACESSÓRIOS"
}

# Mapeamento especial para produtos HUGO BOSS Polo e Jersey
BOSS_POLO_PATTERNS = [
    r"PADDY", r"PAUL", r"POLO", r"JERSEY", r"PIMA", r"PARLAY", r"PALLAS", r"PROUT",
    r"PLAYER", r"PERCY", r"PAULE", r"PIRO", r"PASSERBY", r"PACELLO", r"PHILLIPSON",
    r"PLISY", r"PRIDE", r"PENROSE", r"PALLAS"
]

def get_best_category_match(category: str) -> str:
    """
    Obtém a correspondência mais próxima na lista de categorias
    
    Args:
        category: Categoria a ser comparada
        
    Returns:
        str: Categoria mais próxima da lista ou "ACESSÓRIOS" como padrão
    """
    if not category:
        return "ACESSÓRIOS"
    
    # Tentar encontrar correspondências diretas ou próximas
    matches = get_close_matches(category.upper(), CATEGORIES, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    
    # Se não encontrar, retornar categoria padrão
    return "ACESSÓRIOS"

def map_category(
    category: str, 
    product_name: Optional[str] = None, 
    brand: Optional[str] = None
) -> str:
    """
    Mapeia uma categoria de qualquer idioma para a categoria em português
    de acordo com a lista de CATEGORIES.
    
    Considera o nome do produto e a marca para fazer mapeamentos específicos
    para casos especiais (ex: produtos HUGO BOSS).
    
    Args:
        category: Categoria original
        product_name: Nome do produto (opcional)
        brand: Nome da marca (opcional)
        
    Returns:
        str: Categoria em português da lista CATEGORIES
    """
    if not category:
        category = ""
    
    # 1. Conversão para maiúsculas para facilitar a comparação
    category_upper = category.upper().strip()
    
    # 2. Verificar se a categoria já está na lista oficial
    if category_upper in CATEGORIES:
        return category_upper
    
    # 3. Tentar usar a função existente get_category
    normalized = get_category(category)
    if normalized:
        return normalized
    
    # 4. Verificar casos especiais baseados no nome do produto e marca
    if product_name and brand:
        product_upper = product_name.upper()
        brand_upper = brand.upper()
        
        # Caso especial: Polos/Jerseys da Hugo Boss
        if "HUGO BOSS" in brand_upper or "BOSS" in brand_upper:
            # Verificar se o nome do produto contém padrões de polos
            for pattern in BOSS_POLO_PATTERNS:
                if re.search(pattern, product_upper):
                    logger.info(f"Produto HUGO BOSS mapeado como POLO: {product_name}")
                    return "POLOS"
            
            # Verificar palavras específicas no nome do produto
            if any(word in product_upper for word in ["POLO", "JERSEY", "KNIT SHIRT"]):
                return "POLOS"
    
    # 5. Tentar traduzir do inglês para português
    if category_upper in ENGLISH_TO_PORTUGUESE:
        translated = ENGLISH_TO_PORTUGUESE[category_upper]
        logger.info(f"Traduzida categoria '{category}' para '{translated}'")
        return translated
    
    # 6. Tentar encontrar correspondência parcial no dicionário
    for eng, pt in ENGLISH_TO_PORTUGUESE.items():
        if eng in category_upper or category_upper in eng:
            logger.info(f"Correspondência parcial para '{category}': '{pt}'")
            return pt
    
    # 7. Procurar por palavras-chave no texto da categoria
    keywords = {
        "SHIRT": "CAMISAS",
        "COAT": "CASACOS",
        "JACKET": "CASACOS",
        "DRESS": "VESTIDOS",
        "BLOUSE": "BLUSAS",
        "TOP": "BLUSAS",
        "PANT": "CALÇAS",
        "TROUSER": "CALÇAS",
        "KNIT": "MALHAS",
        "SWEATER": "MALHAS",
        "SKIRT": "SAIAS",
        "TEE": "T-SHIRTS",
        "POLO": "POLOS",
        "JEAN": "JEANS",
        "DENIM": "JEANS",
        "SWEAT": "SWEATSHIRTS",
        "HOODIE": "SWEATSHIRTS",
        "BLAZER": "BLAZERS E FATOS",
        "SUIT": "BLAZERS E FATOS",
        "SHOE": "CALÇADO",
        "BOOT": "CALÇADO",
        "SNEAKER": "CALÇADO"
    }
    
    for keyword, cat in keywords.items():
        if keyword in category_upper:
            logger.info(f"Palavra-chave '{keyword}' encontrada em '{category}': mapeado para '{cat}'")
            return cat
    
    best_match = get_best_category_match(category)
    logger.info(f"Nenhuma correspondência encontrada para '{category}', usando melhor aproximação: '{best_match}'")
    return best_match