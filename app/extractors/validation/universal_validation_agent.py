# app/extractors/validation/universal_validation_agent.py

import logging
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Níveis de severidade para problemas de validação"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Representa um problema encontrado durante a validação"""
    code: str
    severity: ValidationSeverity
    message: str
    field_path: str
    suggested_fix: Optional[str] = None
    confidence: float = 1.0

class UniversalValidationAgent:
    """
    Agente de validação universal que verifica a qualidade e consistência
    dos dados extraídos usando padrões genéricos da indústria.
    """
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.price_ranges = self._initialize_price_ranges()
        self.size_patterns = self._initialize_size_patterns()
        self.color_patterns = self._initialize_color_patterns()
        
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Inicializa regras de validação genéricas"""
        return {
            "required_fields": {
                "product": ["name", "material_code", "category"],
                "color": ["color_code", "sizes"],
                "size": ["size", "quantity"],
                "order_info": ["supplier"]
            },
            "field_formats": {
                "material_code": r"^[A-Z0-9]{3,20}$",
                "color_code": r"^\d{3}$",
                "quantity": r"^\d+$",
                "unit_price": r"^\d+(\.\d{2})?$",
                "sales_price": r"^\d+(\.\d{2})?$"
            },
            "business_rules": {
                "max_products_per_document": 1000,
                "max_colors_per_product": 50,
                "max_sizes_per_color": 20,
                "min_quantity": 1,
                "max_quantity": 10000,
                "min_price": 0.01,
                "max_price": 10000.00
            }
        }
    
    def _initialize_price_ranges(self) -> Dict[str, float]:
        """
        Inicializa faixas de preços esperadas por categoria.
        Valores em euros para validação de consistência.
        """
        # Mapeamento corrigido: cada categoria tem seu próprio valor
        return {
            'CAMISAS': 45.0,
            'CASACOS': 120.0,
            'VESTIDOS': 80.0,
            'BLUSAS': 35.0,
            'CALÇAS': 60.0,
            'CALÇÃO': 25.0,
            'MALHAS': 70.0,
            'MAGLIA': 70.0,
            'KNIT': 70.0,
            'SWEATER': 70.0,
            'SAIAS': 40.0,
            'T-SHIRTS': 20.0,
            'POLOS': 50.0,
            'JEANS': 75.0,
            'SWEATSHIRTS': 65.0,
            'BLAZERS E FATOS': 150.0,
            'BLUSÕES E PARKAS': 100.0,
            'CALÇADO': 90.0,
            'TOP': 30.0,
            'ACESSÓRIOS': 25.0
        }
    
    def _initialize_size_patterns(self) -> Dict[str, List[str]]:
        """Inicializa padrões de tamanhos válidos por categoria"""
        return {
            "clothing_letters": ["XS", "S", "M", "L", "XL", "XXL", "XXXL"],
            "clothing_numeric": ["34", "36", "38", "40", "42", "44", "46", "48", "50", "52", "54"],
            "pants_numeric": ["24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "36"],
            "shoes_numeric": ["35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46"],
            "children_numeric": ["2", "4", "6", "8", "10", "12", "14", "16"],
            "universal": ["TU", "ONE SIZE", "ÚNICA"]
        }
    
    def _initialize_color_patterns(self) -> Dict[str, str]:
        """Inicializa padrões de validação para cores"""
        return {
            "numeric_code": r"^\d{3}$",
            "alpha_code": r"^[A-Z]{2,4}$",
            "alphanumeric_code": r"^[A-Z0-9]{2,5}$"
        }
    
    def validate_extraction_result(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida um resultado completo de extração
        
        Args:
            extraction_result: Resultado da extração a ser validado
            
        Returns:
            Dict: Relatório de validação com problemas encontrados e métricas
        """
        issues = []
        metrics = {
            "total_products": 0,
            "valid_products": 0,
            "total_colors": 0,
            "valid_colors": 0,
            "total_sizes": 0,
            "valid_sizes": 0,
            "validation_score": 0.0
        }
        
        try:
            # Validar estrutura geral
            issues.extend(self._validate_overall_structure(extraction_result))
            
            # Validar order_info
            if "order_info" in extraction_result:
                issues.extend(self._validate_order_info(extraction_result["order_info"]))
            
            # Validar produtos
            if "products" in extraction_result:
                product_issues, product_metrics = self._validate_products(extraction_result["products"])
                issues.extend(product_issues)
                metrics.update(product_metrics)
            
            # Calcular score geral de validação
            metrics["validation_score"] = self._calculate_validation_score(issues, metrics)
            
            # Agrupar issues por severidade
            issues_by_severity = self._group_issues_by_severity(issues)
            
            return {
                "validation_status": "passed" if len([i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0 else "failed",
                "total_issues": len(issues),
                "issues_by_severity": issues_by_severity,
                "detailed_issues": [self._issue_to_dict(issue) for issue in issues],
                "metrics": metrics,
                "recommendations": self._generate_recommendations(issues, metrics)
            }
            
        except Exception as e:
            logger.exception(f"Erro durante validação: {str(e)}")
            return {
                "validation_status": "error",
                "error": str(e),
                "total_issues": 0,
                "metrics": metrics
            }
    
    def _validate_overall_structure(self, extraction_result: Dict[str, Any]) -> List[ValidationIssue]:
        """Valida a estrutura geral do resultado"""
        issues = []
        
        # Verificar campos obrigatórios principais
        required_top_level = ["products"]
        for field in required_top_level:
            if field not in extraction_result:
                issues.append(ValidationIssue(
                    code="MISSING_TOP_LEVEL_FIELD",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Campo obrigatório '{field}' não encontrado",
                    field_path=field,
                    suggested_fix=f"Adicionar campo '{field}' ao resultado"
                ))
        
        # Verificar se products é uma lista
        if "products" in extraction_result and not isinstance(extraction_result["products"], list):
            issues.append(ValidationIssue(
                code="INVALID_PRODUCTS_TYPE",
                severity=ValidationSeverity.CRITICAL,
                message="Campo 'products' deve ser uma lista",
                field_path="products",
                suggested_fix="Converter 'products' para lista"
            ))
        
        # Verificar se há produtos
        if "products" in extraction_result and len(extraction_result["products"]) == 0:
            issues.append(ValidationIssue(
                code="NO_PRODUCTS_FOUND",
                severity=ValidationSeverity.WARNING,
                message="Nenhum produto encontrado no documento",
                field_path="products",
                suggested_fix="Verificar se o documento contém produtos ou revisar processo de extração"
            ))
        
        return issues
    
    def _validate_order_info(self, order_info: Dict[str, Any]) -> List[ValidationIssue]:
        """Valida informações do pedido"""
        issues = []
        
        # Verificar campos obrigatórios
        required_fields = self.validation_rules["required_fields"]["order_info"]
        for field in required_fields:
            if field not in order_info or not order_info[field]:
                issues.append(ValidationIssue(
                    code="MISSING_ORDER_FIELD",
                    severity=ValidationSeverity.WARNING,
                    message=f"Campo '{field}' não encontrado em order_info",
                    field_path=f"order_info.{field}",
                    suggested_fix=f"Adicionar informação de '{field}' ao pedido"
                ))
        
        # Validar formato de data se presente
        if "date" in order_info and order_info["date"]:
            date_value = order_info["date"]
            if not self._is_valid_date_format(date_value):
                issues.append(ValidationIssue(
                    code="INVALID_DATE_FORMAT",
                    severity=ValidationSeverity.INFO,
                    message=f"Formato de data pode estar incorreto: '{date_value}'",
                    field_path="order_info.date",
                    suggested_fix="Usar formato DD/MM/YYYY ou YYYY-MM-DD"
                ))
        
        return issues
    
    def _validate_products(self, products: List[Dict[str, Any]]) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Valida lista de produtos"""
        issues = []
        metrics = {
            "total_products": len(products),
            "valid_products": 0,
            "total_colors": 0,
            "valid_colors": 0,
            "total_sizes": 0,
            "valid_sizes": 0
        }
        
        seen_material_codes = set()
        
        for i, product in enumerate(products):
            product_issues = self._validate_single_product(product, i, seen_material_codes)
            issues.extend(product_issues)
            
            # Contar métricas
            if len([issue for issue in product_issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0:
                metrics["valid_products"] += 1
            
            # Contar cores e tamanhos
            if "colors" in product and isinstance(product["colors"], list):
                for color in product["colors"]:
                    metrics["total_colors"] += 1
                    if "sizes" in color and isinstance(color["sizes"], list):
                        for size in color["sizes"]:
                            metrics["total_sizes"] += 1
        
        return issues, metrics
    
    def _validate_single_product(self, product: Dict[str, Any], index: int, seen_codes: Set[str]) -> List[ValidationIssue]:
        """Valida um produto individual"""
        issues = []
        base_path = f"products[{index}]"
        
        # Verificar campos obrigatórios
        required_fields = self.validation_rules["required_fields"]["product"]
        for field in required_fields:
            if field not in product or not product[field]:
                issues.append(ValidationIssue(
                    code="MISSING_PRODUCT_FIELD",
                    severity=ValidationSeverity.ERROR,
                    message=f"Campo obrigatório '{field}' não encontrado no produto",
                    field_path=f"{base_path}.{field}",
                    suggested_fix=f"Adicionar campo '{field}' ao produto"
                ))
        
        # Validar código de material
        if "material_code" in product:
            material_code = product["material_code"]
            
            # Verificar formato
            if not re.match(self.validation_rules["field_formats"]["material_code"], str(material_code)):
                issues.append(ValidationIssue(
                    code="INVALID_MATERIAL_CODE_FORMAT",
                    severity=ValidationSeverity.WARNING,
                    message=f"Formato de código de material pode estar incorreto: '{material_code}'",
                    field_path=f"{base_path}.material_code",
                    suggested_fix="Verificar se o código segue o padrão alpanumérico"
                ))
            
            # Verificar duplicação
            if material_code in seen_codes:
                issues.append(ValidationIssue(
                    code="DUPLICATE_MATERIAL_CODE",
                    severity=ValidationSeverity.WARNING,
                    message=f"Código de material duplicado: '{material_code}'",
                    field_path=f"{base_path}.material_code",
                    suggested_fix="Verificar se produtos com mesmo código são realmente duplicatas"
                ))
            else:
                seen_codes.add(material_code)
        
        # Validar categoria
        if "category" in product:
            category = product["category"]
            if not self._is_valid_category(category):
                issues.append(ValidationIssue(
                    code="INVALID_CATEGORY",
                    severity=ValidationSeverity.WARNING,
                    message=f"Categoria não reconhecida: '{category}'",
                    field_path=f"{base_path}.category",
                    suggested_fix="Usar uma das categorias padrão do sistema"
                ))
        
        # Validar cores
        if "colors" in product and isinstance(product["colors"], list):
            color_issues = self._validate_product_colors(product["colors"], base_path, product.get("category"))
            issues.extend(color_issues)
        
        return issues
    
    def _validate_product_colors(self, colors: List[Dict[str, Any]], base_path: str, category: Optional[str] = None) -> List[ValidationIssue]:
        """Valida cores de um produto"""
        issues = []
        
        seen_color_codes = set()
        
        for i, color in enumerate(colors):
            color_path = f"{base_path}.colors[{i}]"
            
            # Verificar campos obrigatórios
            required_fields = self.validation_rules["required_fields"]["color"]
            for field in required_fields:
                if field not in color or not color[field]:
                    issues.append(ValidationIssue(
                        code="MISSING_COLOR_FIELD",
                        severity=ValidationSeverity.ERROR,
                        message=f"Campo obrigatório '{field}' não encontrado na cor",
                        field_path=f"{color_path}.{field}",
                        suggested_fix=f"Adicionar campo '{field}' à cor"
                    ))
            
            # Validar código da cor
            if "color_code" in color:
                color_code = color["color_code"]
                
                # Verificar formato
                if not re.match(self.color_patterns["numeric_code"], str(color_code)):
                    issues.append(ValidationIssue(
                        code="INVALID_COLOR_CODE_FORMAT",
                        severity=ValidationSeverity.WARNING,
                        message=f"Formato de código de cor pode estar incorreto: '{color_code}'",
                        field_path=f"{color_path}.color_code",
                        suggested_fix="Usar código numérico de 3 dígitos"
                    ))
                
                # Verificar duplicação
                if color_code in seen_color_codes:
                    issues.append(ValidationIssue(
                        code="DUPLICATE_COLOR_CODE",
                        severity=ValidationSeverity.WARNING,
                        message=f"Código de cor duplicado: '{color_code}'",
                        field_path=f"{color_path}.color_code",
                        suggested_fix="Verificar se cores com mesmo código são realmente duplicatas"
                    ))
                else:
                    seen_color_codes.add(color_code)
            
            # Validar preços
            for price_field in ["unit_price", "sales_price"]:
                if price_field in color and color[price_field] is not None:
                    price_issues = self._validate_price(color[price_field], f"{color_path}.{price_field}", category)
                    issues.extend(price_issues)
            
            # Validar tamanhos
            if "sizes" in color and isinstance(color["sizes"], list):
                size_issues = self._validate_color_sizes(color["sizes"], color_path, category)
                issues.extend(size_issues)
        
        return issues
    
    def _validate_color_sizes(self, sizes: List[Dict[str, Any]], base_path: str, category: Optional[str] = None) -> List[ValidationIssue]:
        """Valida tamanhos de uma cor"""
        issues = []
        
        seen_sizes = set()
        
        for i, size in enumerate(sizes):
            size_path = f"{base_path}.sizes[{i}]"
            
            # Verificar campos obrigatórios
            required_fields = self.validation_rules["required_fields"]["size"]
            for field in required_fields:
                if field not in size or size[field] is None:
                    issues.append(ValidationIssue(
                        code="MISSING_SIZE_FIELD",
                        severity=ValidationSeverity.ERROR,
                        message=f"Campo obrigatório '{field}' não encontrado no tamanho",
                        field_path=f"{size_path}.{field}",
                        suggested_fix=f"Adicionar campo '{field}' ao tamanho"
                    ))
            
            # Validar tamanho
            if "size" in size:
                size_value = str(size["size"]).upper()
                
                if not self._is_valid_size(size_value, category):
                    issues.append(ValidationIssue(
                        code="INVALID_SIZE",
                        severity=ValidationSeverity.WARNING,
                        message=f"Tamanho não reconhecido: '{size_value}'",
                        field_path=f"{size_path}.size",
                        suggested_fix="Verificar se o tamanho está correto"
                    ))
                
                # Verificar duplicação
                if size_value in seen_sizes:
                    issues.append(ValidationIssue(
                        code="DUPLICATE_SIZE",
                        severity=ValidationSeverity.WARNING,
                        message=f"Tamanho duplicado: '{size_value}'",
                        field_path=f"{size_path}.size",
                        suggested_fix="Remover tamanhos duplicados"
                    ))
                else:
                    seen_sizes.add(size_value)
            
            # Validar quantidade
            if "quantity" in size:
                quantity = size["quantity"]
                
                try:
                    qty_num = float(quantity)
                    
                    if qty_num < self.validation_rules["business_rules"]["min_quantity"]:
                        issues.append(ValidationIssue(
                            code="INVALID_QUANTITY_LOW",
                            severity=ValidationSeverity.WARNING,
                            message=f"Quantidade muito baixa: {quantity}",
                            field_path=f"{size_path}.quantity",
                            suggested_fix="Verificar se a quantidade está correta"
                        ))
                    
                    if qty_num > self.validation_rules["business_rules"]["max_quantity"]:
                        issues.append(ValidationIssue(
                            code="INVALID_QUANTITY_HIGH",
                            severity=ValidationSeverity.WARNING,
                            message=f"Quantidade muito alta: {quantity}",
                            field_path=f"{size_path}.quantity",
                            suggested_fix="Verificar se a quantidade está correta"
                        ))
                
                except (ValueError, TypeError):
                    issues.append(ValidationIssue(
                        code="INVALID_QUANTITY_FORMAT",
                        severity=ValidationSeverity.ERROR,
                        message=f"Formato de quantidade inválido: '{quantity}'",
                        field_path=f"{size_path}.quantity",
                        suggested_fix="Usar apenas números para quantidade"
                    ))
        
        return issues
    
    def _validate_price(self, price: Any, field_path: str, category: Optional[str] = None) -> List[ValidationIssue]:
        """Valida um preço"""
        issues = []
        
        try:
            price_num = float(price)
            
            # Verificar limites gerais
            if price_num < self.validation_rules["business_rules"]["min_price"]:
                issues.append(ValidationIssue(
                    code="PRICE_TOO_LOW",
                    severity=ValidationSeverity.WARNING,
                    message=f"Preço muito baixo: {price}",
                    field_path=field_path,
                    suggested_fix="Verificar se o preço está correto"
                ))
            
            if price_num > self.validation_rules["business_rules"]["max_price"]:
                issues.append(ValidationIssue(
                    code="PRICE_TOO_HIGH",
                    severity=ValidationSeverity.WARNING,
                    message=f"Preço muito alto: {price}",
                    field_path=field_path,
                    suggested_fix="Verificar se o preço está correto"
                ))
            
            # Verificar faixa esperada por categoria
            if category and category.upper() in self.price_ranges:
                expected_price = self.price_ranges[category.upper()]
                ratio = price_num / expected_price
                
                # Se o preço estiver muito fora da faixa esperada
                if ratio < 0.3 or ratio > 3.0:
                    issues.append(ValidationIssue(
                        code="PRICE_OUTSIDE_EXPECTED_RANGE",
                        severity=ValidationSeverity.INFO,
                        message=f"Preço fora da faixa esperada para categoria '{category}': {price} (esperado: ~{expected_price})",
                        field_path=field_path,
                        suggested_fix="Verificar se categoria e preço estão corretos",
                        confidence=0.7
                    ))
        
        except (ValueError, TypeError):
            issues.append(ValidationIssue(
                code="INVALID_PRICE_FORMAT",
                severity=ValidationSeverity.ERROR,
                message=f"Formato de preço inválido: '{price}'",
                field_path=field_path,
                suggested_fix="Usar formato numérico para preços"
            ))
        
        return issues
    
    def _is_valid_date_format(self, date_str: str) -> bool:
        """Verifica se uma string tem formato de data válido"""
        date_patterns = [
            r"\d{2}/\d{2}/\d{4}",      # DD/MM/YYYY
            r"\d{4}-\d{2}-\d{2}",      # YYYY-MM-DD
            r"\d{2}-\d{2}-\d{4}",      # DD-MM-YYYY
            r"\d{2}\.\d{2}\.\d{4}"     # DD.MM.YYYY
        ]
        
        return any(re.match(pattern, date_str) for pattern in date_patterns)
    
    def _is_valid_category(self, category: str) -> bool:
        """Verifica se uma categoria é válida"""
        from app.data.reference_data import CATEGORIES
        
        category_upper = category.upper()
        return category_upper in CATEGORIES
    
    def _is_valid_size(self, size: str, category: Optional[str] = None) -> bool:
        """Verifica se um tamanho é válido"""
        size_upper = size.upper()
        
        # Verificar em todos os padrões conhecidos
        for pattern_list in self.size_patterns.values():
            if size_upper in pattern_list:
                return True
        
        # Verificar padrões numéricos comuns
        if re.match(r"^\d{2}$", size):
            return True
        
        return False
    
    def _calculate_validation_score(self, issues: List[ValidationIssue], metrics: Dict[str, Any]) -> float:
        """Calcula score de validação baseado nos problemas encontrados"""
        if metrics["total_products"] == 0:
            return 0.0
        
        # Peso por severidade
        severity_weights = {
            ValidationSeverity.INFO: 0.1,
            ValidationSeverity.WARNING: 0.3,
            ValidationSeverity.ERROR: 0.7,
            ValidationSeverity.CRITICAL: 1.0
        }
        
        # Calcular penalidades
        total_penalty = 0.0
        for issue in issues:
            total_penalty += severity_weights[issue.severity] * issue.confidence
        
        # Score baseado em produtos válidos e penalidades
        base_score = metrics["valid_products"] / metrics["total_products"]
        penalty_factor = min(1.0, total_penalty / metrics["total_products"])
        
        final_score = max(0.0, base_score - penalty_factor)
        
        return round(final_score, 3)
    
    def _group_issues_by_severity(self, issues: List[ValidationIssue]) -> Dict[str, int]:
        """Agrupa issues por severidade"""
        counts = {severity.value: 0 for severity in ValidationSeverity}
        
        for issue in issues:
            counts[issue.severity.value] += 1
        
        return counts
    
    def _issue_to_dict(self, issue: ValidationIssue) -> Dict[str, Any]:
        """Converte ValidationIssue para dicionário"""
        return {
            "code": issue.code,
            "severity": issue.severity.value,
            "message": issue.message,
            "field_path": issue.field_path,
            "suggested_fix": issue.suggested_fix,
            "confidence": issue.confidence
        }
    
    def _generate_recommendations(self, issues: List[ValidationIssue], metrics: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas nos problemas encontrados"""
        recommendations = []
        
        # Contar tipos de problemas
        issue_counts = {}
        for issue in issues:
            issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1
        
        # Recomendações baseadas em padrões
        if issue_counts.get("MISSING_PRODUCT_FIELD", 0) > 0:
            recommendations.append("Verificar processo de extração para garantir captura de campos obrigatórios")
        
        if issue_counts.get("INVALID_PRICE_FORMAT", 0) > 0:
            recommendations.append("Revisar detecção de preços para garantir formato numérico correto")
        
        if issue_counts.get("DUPLICATE_MATERIAL_CODE", 0) > 0:
            recommendations.append("Implementar verificação de duplicatas durante extração")
        
        if metrics["valid_products"] / max(1, metrics["total_products"]) < 0.8:
            recommendations.append("Taxa de produtos válidos baixa - revisar processo de extração")
        
        if len(issues) == 0:
            recommendations.append("Dados extraídos têm boa qualidade - nenhum problema crítico encontrado")
        
        return recommendations