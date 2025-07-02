# app/extractors/size_detection_agent.py - VERSÃO CORRIGIDA

import logging
from typing import Dict, Any, List, Optional, Set
import re

logger = logging.getLogger(__name__)

class SizeDetectionAgent:
    def __init__(self):
        # CORREÇÃO: Sistema de tamanhos mais específico para LIUJO
        self.size_systems = {
            'clothing_numeric_eu': {
                'sizes': ['34', '36', '38', '40', '42', '44', '46', '48', '50', '52', '54', '56'],
                'categories': ['VESTIDOS', 'BLUSAS', 'SAIAS', 'CASACOS', 'CAMISAS'],
                'priority': 1
            },
            'clothing_letters': {
                'sizes': ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL'],
                'categories': ['T-SHIRTS', 'MALHAS', 'SWEATSHIRTS', 'POLOS'],
                'priority': 2
            },
            'pants_numeric': {
                'sizes': ['24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35'],
                'categories': ['CALÇAS', 'JEANS'],
                'priority': 3
            },
            'combined_sizes': {
                'sizes': ['38/XS', '40/S', '42/M', '44/L', '46/XL', '48/XXL'],
                'categories': [],  # Aplicável a todas
                'priority': 4
            },
            'children_sizes': {
                'sizes': ['2', '4', '6', '8', '10', '12', '14', '16'],
                'categories': ['INFANTIL'],
                'priority': 5
            },
            'universal_size': {
                'sizes': ['TU', 'UNICA', 'ÚNICO', 'ONE SIZE'],
                'categories': ['ACESSÓRIOS'],
                'priority': 6
            }
        }
        
        # CORREÇÃO: Mapeamento de limpeza de tamanhos
        self.size_normalization = {
            'EXTRA SMALL': 'XS',
            'SMALL': 'S',
            'MEDIUM': 'M',
            'LARGE': 'L',
            'EXTRA LARGE': 'XL',
            'XX LARGE': 'XXL',
            '2XL': 'XXL',
            '3XL': 'XXXL',
            'XXX LARGE': 'XXXL',
        }
        
        # CORREÇÃO: Tamanhos inválidos que devem ser ignorados
        self.invalid_sizes = {
            'QTY', 'PRICE', 'TOTAL', 'SUBTOTAL', 
            'COLOR', 'MODEL', 'CODE', 'NAME',
            'EUR', 'USD', '$', '€', '%',
            'PCES', 'PCS', 'PIECES',
            'NULL', 'NONE', 'N/A', '-', '_', '.',
            # Números que não são tamanhos
            '1', '11', '111', '0', '00'
        }
    
    def normalize_size_extraction(self, extracted_sizes: List[Dict], category: str = None) -> List[Dict]:
        """
        CORREÇÃO: Normalização mais rigorosa com validação por sistema
        """
        if not extracted_sizes:
            return []
        
        # Identificar sistema de tamanhos mais provável
        detected_systems = self._detect_size_systems(extracted_sizes, category)
        
        normalized = []
        invalid_count = 0
        
        for size_info in extracted_sizes:
            cleaned_size = self._clean_and_validate_size(size_info, detected_systems)
            if cleaned_size:
                normalized.append(cleaned_size)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            logger.debug(f"Sistema de tamanhos: {[s for s in detected_systems.keys()]}, "
                        f"Válidos: {len(normalized)}, Inválidos: {invalid_count}")
        
        return normalized
    
    def _detect_size_systems(self, extracted_sizes: List[Dict], category: str = None) -> Dict[str, float]:
        """
        CORREÇÃO: Detecção automática do sistema de tamanhos usado
        """
        system_scores = {}
        
        # Extrair tamanhos únicos
        unique_sizes = set()
        for size_info in extracted_sizes:
            size = str(size_info.get('size', '')).strip().upper()
            if size:
                unique_sizes.add(size)
        
        # Calcular scores para cada sistema
        for system_name, system_info in self.size_systems.items():
            score = 0.0
            system_sizes = set(system_info['sizes'])
            
            # Score baseado na correspondência de tamanhos
            matching_sizes = unique_sizes.intersection(system_sizes)
            if system_sizes:
                size_match_ratio = len(matching_sizes) / len(system_sizes)
                score += size_match_ratio * 0.7
            
            # Score baseado na categoria
            if category and system_info['categories']:
                if category in system_info['categories']:
                    score += 0.3
            
            # Bonus para sistemas prioritários se houver correspondência
            if score > 0:
                priority_bonus = (6 - system_info['priority']) * 0.05
                score += priority_bonus
            
            system_scores[system_name] = score
            
            if score > 0:
                logger.info(f"Sistema de tamanhos detectado: {system_name} (score: {score:.2f})")
        
        # Retornar apenas sistemas com score > 0
        return {k: v for k, v in system_scores.items() if v > 0}
    
    def _clean_and_validate_size(self, size_info: Dict, detected_systems: Dict[str, float]) -> Optional[Dict]:
        """
        CORREÇÃO: Limpeza e validação rigorosa de tamanho individual
        """
        if not isinstance(size_info, dict):
            return None
        
        size = str(size_info.get('size', '')).strip().upper()
        quantity = size_info.get('quantity', 0)
        
        # 1. Limpeza básica do tamanho
        size = self._normalize_size_string(size)
        
        # 2. Verificar se é tamanho inválido
        if not size or size in self.invalid_sizes:
            return None
        
        # 3. Validar quantidade
        try:
            qty_num = float(quantity) if quantity is not None else 0
            if qty_num <= 0:
                return None
            # Converter para int se for número inteiro
            if qty_num.is_integer():
                qty_num = int(qty_num)
        except (ValueError, TypeError):
            return None
        
        # 4. Validar tamanho contra sistemas detectados
        if not self._is_size_valid_for_systems(size, detected_systems):
            logger.debug(f"Tamanho '{size}' não válido para sistemas detectados")
            return None
        
        return {
            'size': size,
            'quantity': qty_num
        }
    
    def _normalize_size_string(self, size: str) -> str:
        """
        CORREÇÃO: Normalização específica de strings de tamanho
        """
        if not size:
            return ""
        
        # Remover caracteres especiais
        size = re.sub(r'[^\w\s/]', '', size)
        size = size.strip().upper()
        
        # Aplicar mapeamento de normalização
        if size in self.size_normalization:
            size = self.size_normalization[size]
        
        # Tratar tamanhos numéricos - remover zeros à esquerda
        if size.isdigit():
            size = str(int(size))
        
        # Tratar casos especiais
        if size.startswith('0') and len(size) > 1 and size[1:].isdigit():
            size = str(int(size))  # Remove zeros à esquerda
        
        return size
    
    def _is_size_valid_for_systems(self, size: str, detected_systems: Dict[str, float]) -> bool:
        """
        CORREÇÃO: Validação específica contra sistemas detectados
        """
        if not detected_systems:
            # Se não detectou sistema específico, usar validação geral
            return self._is_generally_valid_size(size)
        
        # Verificar se o tamanho é válido em pelo menos um sistema detectado
        for system_name, score in detected_systems.items():
            if score > 0:  # Sistema foi detectado
                system_info = self.size_systems[system_name]
                if size in system_info['sizes']:
                    return True
        
        return False
    
    def _is_generally_valid_size(self, size: str) -> bool:
        """
        CORREÇÃO: Validação geral para quando não há sistema específico detectado
        """
        # Consolidar todos os tamanhos válidos
        all_valid_sizes = set()
        for system_info in self.size_systems.values():
            all_valid_sizes.update(system_info['sizes'])
        
        return size in all_valid_sizes