# app/utils/supplier_utils.py
import logging
import re
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any, Tuple
from app.data.reference_data import SUPPLIER_MAP, SUPPLIER_DATA, get_supplier_code

logger = logging.getLogger(__name__)

def normalize_supplier_name(name: str) -> str:
    """
    Normaliza o nome do fornecedor para comparação, removendo sufixos de empresas e caracteres especiais.
    
    Args:
        name: Nome do fornecedor
        
    Returns:
        str: Nome normalizado
    """
    if not name:
        return ""
    
    # Converter para maiúsculas
    result = name.upper()
    
    # Remover sufixos comuns de empresas
    suffixes = [
        r'\bS\.p\.A\.?\b', r'\bS\.A\.?\b', r'\bS\.L\.?\b', r'\bLtd\.?\b', 
        r'\bLtda\.?\b', r'\bInc\.?\b', r'\bLLC\.?\b', r'\bGmbH\.?\b',
        r'\bCo\.?\b', r'\bCorp\.?\b', r'\bB\.V\.?\b', r'\bA\.G\.?\b'
    ]
    
    for suffix in suffixes:
        result = re.sub(suffix, '', result, flags=re.IGNORECASE)
    
    # Remover caracteres especiais e converter para espaço
    result = re.sub(r'[^\w\s]', ' ', result)
    
    # Remover espaços extras
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def calculate_similarity_score(str1: str, str2: str) -> float:
    """
    Calcula uma pontuação de similaridade entre duas strings usando diferentes métricas.
    
    Args:
        str1: Primeira string
        str2: Segunda string
        
    Returns:
        float: Pontuação de similaridade entre 0 e 1
    """
    # Similaridade de sequência (considera ordem dos caracteres)
    seq_similarity = SequenceMatcher(None, str1, str2).ratio()
    
    # Similaridade de conjunto (tokens em comum)
    tokens1 = set(str1.split())
    tokens2 = set(str2.split())
    
    # Evitar divisão por zero
    if not tokens1 or not tokens2:
        set_similarity = 0
    else:
        common_tokens = tokens1.intersection(tokens2)
        set_similarity = len(common_tokens) / max(len(tokens1), len(tokens2))
    
    # Verificar se há tokens significativos em comum
    significant_tokens = False
    for t1 in tokens1:
        if len(t1) >= 4:  # token com pelo menos 4 caracteres
            for t2 in tokens2:
                if t1 == t2 or (len(t1) >= 4 and t1 in t2) or (len(t2) >= 4 and t2 in t1):
                    significant_tokens = True
                    break
    
    # Ponderar as diferentes métricas
    token_bonus = 0.2 if significant_tokens else 0.0
    final_score = (seq_similarity * 0.4) + (set_similarity * 0.4) + token_bonus
    
    return min(final_score, 1.0)  # Garantir que não exceda 1.0

def find_most_similar_supplier(normalized_supplier: str) -> Tuple[Optional[str], float]:
    if not normalized_supplier:
        return None, 0.0
    
    # Verificar correspondência exata primeiro
    for supplier in SUPPLIER_MAP.values():
        if normalize_supplier_name(supplier) == normalized_supplier:
            return supplier, 1.0
    
    # Preparar lista de fornecedores conhecidos
    known_suppliers = list(SUPPLIER_MAP.values())
    normalized_suppliers = [normalize_supplier_name(s) for s in known_suppliers]
    
    # Calcular similaridade com cada fornecedor conhecido
    best_match = None
    best_score = 0.0
    
    # Log para depuração
    all_scores = []
    
    for i, supplier in enumerate(normalized_suppliers):
        if not supplier:  # Pular fornecedores com nome vazio após normalização
            continue
            
        # Calcular pontuação de similaridade
        score = calculate_similarity_score(normalized_supplier, supplier)
        all_scores.append((known_suppliers[i], score))
        
        # Verificar token-a-token também
        tokens1 = normalized_supplier.split()
        tokens2 = supplier.split()
        
        # Verificar se há um token muito específico em comum
        for token in tokens1:
            if len(token) >= 4 and token in tokens2:
                score = max(score, 0.7)  # Aumentar pontuação se há um token significativo em comum
        
        if score > best_score:
            best_score = score
            best_match = known_suppliers[i]
    
    # Registrar todos os scores para depuração
    sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
    log_scores = sorted_scores[:3]  # mostrar apenas os 3 melhores para evitar log muito grande
    logger.debug(f"Top 3 correspondências para '{normalized_supplier}': {log_scores}")
    
    return best_match, best_score

def match_supplier_name(extracted_supplier: str) -> str:
    if not extracted_supplier or extracted_supplier.strip() == "":
        logger.warning("Nome de fornecedor vazio fornecido para correspondência")
        return "Fornecedor não identificado"
    
    try:
        # Verificar correspondência exata primeiro
        if extracted_supplier in SUPPLIER_MAP.values():
            return extracted_supplier
        
        # Normalizar o fornecedor extraído
        normalized_extracted = normalize_supplier_name(extracted_supplier)
        
        if not normalized_extracted:
            return extracted_supplier
        
        best_match, similarity = find_most_similar_supplier(normalized_extracted)
        
        if best_match and similarity > 0.3:
            if similarity == 1.0:
                logger.info(f"Fornecedor '{extracted_supplier}' encontrado por correspondência exata")
            else:
                logger.info(f"Fornecedor '{extracted_supplier}' correspondido com '{best_match}' (similaridade: {similarity:.2f})")
            return best_match
        else:
            logger.warning(f"Nenhuma correspondência encontrada para o fornecedor '{extracted_supplier}' acima do limite mínimo")
            return extracted_supplier
            
    except Exception as e:
        logger.exception(f"Erro ao tentar corresponder fornecedor '{extracted_supplier}': {str(e)}")
        return extracted_supplier

def get_normalized_supplier(supplier_name: str) -> tuple[str, Optional[str]]:  
    normalized_name = match_supplier_name(supplier_name)
    
    supplier_code = None
    
    for code, data in SUPPLIER_DATA.items():
        if data["nome"] == normalized_name:
            supplier_code = code
            break
    
    if not supplier_code:
        supplier_code = get_supplier_code(normalized_name)
    
    return normalized_name, supplier_code

def get_supplier_info(supplier_name_or_code: str) -> Dict[str, Any]:
    if supplier_name_or_code in SUPPLIER_DATA:
        return {
            "code": supplier_name_or_code,
            "name": SUPPLIER_DATA[supplier_name_or_code]["nome"],
            "markup": SUPPLIER_DATA[supplier_name_or_code]["marcacao"]
        }
    
    
    # Se for um nome, primeiro normalizar
    normalized_name = match_supplier_name(supplier_name_or_code)
    
    # Buscar o código
    for code, data in SUPPLIER_DATA.items():
        if data["nome"] == normalized_name:
            return {
                "code": code,
                "name": data["nome"],
                "markup": data["marcacao"]
            }
    
    return {}