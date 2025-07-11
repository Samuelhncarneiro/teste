# app/utils/size_detection.py

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class SizeDetectionAgent:
    def __init__(self):
        self.size_patterns = {
            'clothing_numeric_eu': {
                'sizes': ['32', '34', '36', '38', '40', '42', '44', '46', '48', '50', '52', '54', '56', '58'],
                'categories': ['VESTIDOS', 'BLUSAS', 'SAIAS', 'CASACOS', 'BLAZERS E FATOS'],
                'pattern': r'^(3[2-9]|4[0-9]|5[0-8])$'
            },
            'clothing_letters': {
                'sizes': ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL', '2XL', '3XL'],
                'categories': ['T-SHIRTS', 'MALHAS', 'SWEATSHIRTS', 'POLOS'],
                'pattern': r'^(XS|S|M|L|XL|XXL|XXXL|2XL|3XL)$'
            },
            'pants_numeric': {
                'sizes': ['24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36'],
                'categories': ['CALÇAS', 'JEANS'],
                'pattern': r'^(2[4-9]|3[0-6])$'
            },
            'mixed_sizes': {
                'sizes': ['38/XS', '40/S', '42/M', '44/L', '46/XL', '48/XXL'],
                'categories': ['UNIVERSAL'],
                'pattern': r'^(38/XS|40/S|42/M|44/L|46/XL|48/XXL)$'
            }
        }
    
    def detect_size_system(self, sizes_list: List[str]) -> str:
        """
        Detecta qual sistema de tamanhos está sendo usado
        """
        if not sizes_list:
            return 'unknown'
        
        # Limpar e normalizar tamanhos
        clean_sizes = [self._normalize_size(size) for size in sizes_list if size and size.strip()]
        
        # Contar matches para cada sistema
        system_scores = {}
        
        for system_name, system_info in self.size_patterns.items():
            score = 0
            total_sizes = len(clean_sizes)
            
            for size in clean_sizes:
                if re.match(system_info['pattern'], size):
                    score += 1
            
            if total_sizes > 0:
                system_scores[system_name] = score / total_sizes
        
        # Retornar o sistema com maior score
        if system_scores:
            best_system = max(system_scores, key=system_scores.get)
            if system_scores[best_system] > 0.6:  # Pelo menos 60% de match
                logger.info(f"Sistema de tamanhos detectado: {best_system} (score: {system_scores[best_system]:.2f})")
                return best_system
        
        return 'unknown'
    
    def _normalize_size(self, size: str) -> str:
        """
        Normaliza um tamanho para comparação
        """
        if not size:
            return ""
        
        # Remover espaços e converter para uppercase
        normalized = str(size).strip().upper()
        
        # Remover zeros à esquerda para tamanhos numéricos
        if normalized.isdigit():
            normalized = str(int(normalized))
        
        return normalized
    
    def validate_size_quantity_mapping(
        self, 
        size_quantity_pairs: List[Dict[str, Any]], 
        category: str = None
    ) -> List[Dict[str, Any]]:
        """
        Valida e corrige o mapeamento entre tamanhos e quantidades
        """
        if not size_quantity_pairs:
            return []
        
        # Detectar sistema de tamanhos
        sizes = [pair.get('size', '') for pair in size_quantity_pairs]
        detected_system = self.detect_size_system(sizes)
        
        validated_pairs = []
        
        for pair in size_quantity_pairs:
            size = self._normalize_size(pair.get('size', ''))
            quantity = pair.get('quantity', 0)
            
            # Validar se é um tamanho válido
            if self._is_valid_size_for_system(size, detected_system):
                # Validar quantidade
                try:
                    qty_num = float(quantity) if quantity is not None else 0
                    if qty_num > 0:
                        validated_pairs.append({
                            'size': size,
                            'quantity': int(qty_num) if qty_num.is_integer() else qty_num
                        })
                except (ValueError, TypeError):
                    logger.warning(f"Quantidade inválida para tamanho {size}: {quantity}")
                    continue
            else:
                logger.warning(f"Tamanho inválido para sistema {detected_system}: {size}")
        
        return validated_pairs
    
    def _is_valid_size_for_system(self, size: str, system: str) -> bool:
        """
        Verifica se um tamanho é válido para um sistema específico
        """
        if system == 'unknown':
            # Se não conseguimos detectar o sistema, aceitar tamanhos comuns
            all_valid_sizes = []
            for pattern_info in self.size_patterns.values():
                all_valid_sizes.extend(pattern_info['sizes'])
            return size in all_valid_sizes
        
        if system in self.size_patterns:
            pattern = self.size_patterns[system]['pattern']
            return bool(re.match(pattern, size))
        
        return False
    
    def extract_sizes_from_table_row(
        self, 
        headers: List[str], 
        quantities: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Extrai tamanhos e quantidades de uma linha de tabela com base nos cabeçalhos
        """
        if len(headers) != len(quantities):
            logger.warning(f"Número de cabeçalhos ({len(headers)}) não coincide com quantidades ({len(quantities)})")
            return []
        
        size_quantity_pairs = []
        
        for i, (header, quantity) in enumerate(zip(headers, quantities)):
            # Tentar extrair tamanho do cabeçalho
            potential_size = self._extract_size_from_header(header)
            
            if potential_size:
                try:
                    qty_num = float(quantity) if quantity not in [None, '', 0, '0'] else 0
                    if qty_num > 0:
                        size_quantity_pairs.append({
                            'size': potential_size,
                            'quantity': int(qty_num) if qty_num.is_integer() else qty_num
                        })
                except (ValueError, TypeError):
                    continue
        
        # Validar o mapeamento
        return self.validate_size_quantity_mapping(size_quantity_pairs)
    
    def _extract_size_from_header(self, header: str) -> Optional[str]:
        """
        Extrai o tamanho de um cabeçalho de coluna
        """
        if not header:
            return None
        
        header_clean = str(header).strip().upper()
        
        # Padrões para extrair tamanhos de cabeçalhos
        size_patterns = [
            r'^(XS|S|M|L|XL|XXL|XXXL|2XL|3XL)$',  # Tamanhos por letra
            r'^(3[2-9]|4[0-9]|5[0-8])$',  # Tamanhos numéricos europeus
            r'^(2[4-9]|3[0-6])$',  # Tamanhos de calças
            r'(38/XS|40/S|42/M|44/L|46/XL|48/XXL)',  # Tamanhos mistos
            r'(\d{2})',  # Qualquer número de 2 dígitos
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, header_clean)
            if match:
                return match.group(1)
        
        return None
    
    def normalize_size_extraction(
        self, 
        extracted_sizes: List[Dict], 
        category: str = None
    ) -> List[Dict]:
        """
        Normaliza uma extração de tamanhos, garantindo consistência
        """
        if not extracted_sizes:
            return []
        
        # Primeiro, validar o mapeamento
        validated_sizes = self.validate_size_quantity_mapping(extracted_sizes, category)
        
        # Ordenar por tamanho (se possível)
        try:
            validated_sizes.sort(key=lambda x: self._get_size_sort_key(x['size']))
        except:
            pass  # Se não conseguir ordenar, manter ordem original
        
        return validated_sizes
    
    def _get_size_sort_key(self, size: str) -> Tuple:
        """
        Gera uma chave para ordenação de tamanhos
        """
        # Ordem para tamanhos por letra
        letter_order = {'XS': 1, 'S': 2, 'M': 3, 'L': 4, 'XL': 5, 'XXL': 6, 'XXXL': 7, '2XL': 5, '3XL': 7}
        
        if size in letter_order:
            return (0, letter_order[size])
        
        # Para tamanhos numéricos
        if size.isdigit():
            return (1, int(size))
        
        # Para tamanhos mistos (38/XS)
        if '/' in size:
            parts = size.split('/')
            if parts[0].isdigit():
                return (2, int(parts[0]))
        
        # Fallback: ordenação alfabética
        return (3, size)
    
    def debug_size_extraction(self, headers: List[str], quantities: List[Any]) -> Dict[str, Any]:
        """
        Função de debug para entender problemas na extração de tamanhos
        """
        debug_info = {
            'headers': headers,
            'quantities': quantities,
            'header_quantity_pairs': list(zip(headers, quantities)),
            'detected_sizes': [],
            'valid_pairs': [],
            'issues': []
        }
        
        # Analisar cada cabeçalho
        for i, header in enumerate(headers):
            potential_size = self._extract_size_from_header(header)
            debug_info['detected_sizes'].append({
                'index': i,
                'header': header,
                'detected_size': potential_size,
                'quantity': quantities[i] if i < len(quantities) else None
            })
        
        # Extrair pares válidos
        valid_pairs = self.extract_sizes_from_table_row(headers, quantities)
        debug_info['valid_pairs'] = valid_pairs
        
        # Identificar problemas
        if len(headers) != len(quantities):
            debug_info['issues'].append(f"Mismatch: {len(headers)} headers vs {len(quantities)} quantities")
        
        if not valid_pairs:
            debug_info['issues'].append("Nenhum par tamanho-quantidade válido encontrado")
        
        return debug_info