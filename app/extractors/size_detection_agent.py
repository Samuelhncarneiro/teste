#apps/extractors/size_detection_agent.py

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)
class SizeDetectionAgent:
    def __init__(self):
        self.size_patterns = {
            'clothing_numeric': {
                'sizes': ['34', '36', '38', '40', '42', '44', '46', '48', '50', '52'],
                'categories': ['VESTIDOS', 'BLUSAS', 'SAIAS', 'CASACOS']
            },
            'clothing_letters': {
                'sizes': ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL'],
                'categories': ['CAMISETAS', 'MALHAS', 'TOPS']
            },
            'pants_numeric': {
                'sizes': ['24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34'],
                'categories': ['CALÇAS', 'JEANS']
            }
        }
    
    def normalize_size_extraction(self, extracted_sizes: List[Dict], category: str = None) -> List[Dict]:
        if not extracted_sizes:
            return []
        
        normalized = []
        for size_info in extracted_sizes:
            size = str(size_info.get('size', '')).strip()
            quantity = size_info.get('quantity', 0)
            
            # Validar se é um tamanho válido
            if self._is_valid_size(size, category):
                try:
                    qty_num = float(quantity) if quantity is not None else 0
                    if qty_num > 0:
                        normalized.append({
                            'size': size,
                            'quantity': int(qty_num) if qty_num.is_integer() else qty_num
                        })
                except (ValueError, TypeError):
                    continue
        
        return normalized
    
    def _is_valid_size(self, size: str, category: str = None) -> bool:
        all_valid_sizes = []
        for pattern_info in self.size_patterns.values():
            all_valid_sizes.extend(pattern_info['sizes'])
        return size in all_valid_sizes