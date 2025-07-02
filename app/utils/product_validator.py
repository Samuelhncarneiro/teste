# app/utils/product_validator.py
import re
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class ProductValidator:
    def __init__(self):
        self.valid_material_code_patterns = [
            r'^CF\d{4}[A-Z]{2,6}\d*$',
            r'^[A-Z]{2,4}\d{4,}[A-Z]*\d*$', 
        ]
        
        self.required_product_fields = [
            'name', 'material_code', 'colors'
        ]
        
        self.required_color_fields = [
            'color_code', 'sizes'
        ]
        
        self.required_size_fields = [
            'size', 'quantity'
        ]
    
    def validate_product(self, product: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Valida um produto completo e retorna (é_válido, lista_de_erros)
        """
        errors = []
        
        # Validar estrutura básica
        if not isinstance(product, dict):
            return False, ["Produto deve ser um dicionário"]
        
        # Validar campos obrigatórios
        for field in self.required_product_fields:
            if field not in product or not product[field]:
                errors.append(f"Campo obrigatório ausente: {field}")
        
        # Validar material_code específico
        material_code = product.get('material_code', '')
        if material_code and not self._is_valid_material_code(material_code):
            errors.append(f"Código de material inválido: {material_code}")
        
        # Validar cores
        colors = product.get('colors', [])
        if not isinstance(colors, list) or len(colors) == 0:
            errors.append("Produto deve ter pelo menos uma cor")
        else:
            for i, color in enumerate(colors):
                color_valid, color_errors = self._validate_color(color)
                if not color_valid:
                    errors.extend([f"Cor {i+1}: {err}" for err in color_errors])
        
        return len(errors) == 0, errors
    
    def _is_valid_material_code(self, code: str) -> bool:
        """Validação específica para códigos de material"""
        if not code or len(code) < 6:
            return False
        
        for pattern in self.valid_material_code_patterns:
            if re.match(pattern, code):
                return True
        
        return False
    
    def _validate_color(self, color: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida uma cor específica"""
        errors = []
        
        if not isinstance(color, dict):
            return False, ["Cor deve ser um dicionário"]
        
        # Validar campos obrigatórios da cor
        for field in self.required_color_fields:
            if field not in color or not color[field]:
                if field == 'sizes':
                    errors.append("Cor deve ter pelo menos um tamanho")
                else:
                    errors.append(f"Campo obrigatório da cor ausente: {field}")
        
        # Validar tamanhos
        sizes = color.get('sizes', [])
        if isinstance(sizes, list) and len(sizes) > 0:
            for i, size in enumerate(sizes):
                size_valid, size_errors = self._validate_size(size)
                if not size_valid:
                    errors.extend([f"Tamanho {i+1}: {err}" for err in size_errors])
        
        return len(errors) == 0, errors
    
    def _validate_size(self, size: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida um tamanho específico"""
        errors = []
        
        if not isinstance(size, dict):
            return False, ["Tamanho deve ser um dicionário"]
        
        # Validar campos obrigatórios do tamanho
        for field in self.required_size_fields:
            if field not in size or size[field] is None:
                errors.append(f"Campo obrigatório do tamanho ausente: {field}")
        
        # Validar quantidade
        quantity = size.get('quantity', 0)
        try:
            qty_num = float(quantity)
            if qty_num <= 0:
                errors.append("Quantidade deve ser maior que zero")
        except (ValueError, TypeError):
            errors.append("Quantidade deve ser um número válido")
        
        return len(errors) == 0, errors