# app/extractors/universal_validation_agent.py
import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import google.generativeai as genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.data.reference_data import CATEGORIES

logger = logging.getLogger(__name__)

@dataclass
class ValidationScore:
    """Pontuação de validação para um produto"""
    material_code_score: float = 0.0
    category_score: float = 0.0
    colors_score: float = 0.0
    sizes_score: float = 0.0
    prices_score: float = 0.0
    overall_score: float = 0.0
    details: Dict[str, Any] = None

@dataclass
class ProductStats:
    """Estatísticas de um produto"""
    material_code: str
    total_colors: int
    total_sizes: int
    total_quantity: int
    price_range: Tuple[float, float]
    size_distribution: Dict[str, int]
    confidence_score: float

class UniversalValidationAgent:
    """
    Agente universal de validação para qualquer tipo de nota de encomenda.
    Calcula percentagens de acerto e valida dados automaticamente.
    """
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Padrões universais para validação
        self.material_code_patterns = {
            'alphanumeric_long': r'^[A-Z]{2,6}\d{4,12}[A-Z]{0,6}$',  # CF5015E0624
            'numeric_long': r'^\d{8,15}$',                           # 50243521
            'mixed_format': r'^[A-Z0-9]{6,20}$',                    # Formato misto
            'short_code': r'^[A-Z]{2,4}\d{2,6}$'                    # AB1234
        }
        
        # Padrões de tamanhos universais
        self.size_patterns = {
            'letters': r'^(XXS|XS|S|M|L|XL|XXL|XXXL)$',
            'numeric_clothing': r'^(3[4-9]|[4-5][0-9]|6[0-4])$',    # 34-64
            'numeric_pants': r'^(2[4-9]|3[0-6])$',                  # 24-36
            'numeric_shoes': r'^(3[5-9]|4[0-6])$',                  # 35-46
            'mixed': r'^(X{0,3}[SML]{1,2}|[0-9]{2,3})$'
        }
        
        # Padrões de cores
        self.color_code_patterns = {
            'numeric': r'^\d{3,8}$',
            'alphanumeric': r'^[A-Z0-9]{3,8}$',
            'hex_like': r'^[A-Fa-f0-9]{6}$'
        }
        
        # Estatísticas globais
        self.validation_stats = {
            "total_products": 0,
            "average_confidence": 0.0,
            "products_by_confidence": {
                "high": 0,    # >80%
                "medium": 0,  # 50-80%
                "low": 0      # <50%
            },
            "common_issues": Counter(),
            "material_code_stats": {},
            "validation_details": []
        }
    
    async def validate_products_with_confidence(
        self, 
        products: List[Dict[str, Any]], 
        original_images: List[str] = None
    ) -> Dict[str, Any]:
        """
        Valida produtos e calcula percentagens de confiança para cada um
        
        Returns:
            Dict com produtos validados e relatório detalhado
        """
        if not products:
            return {"validated_products": [], "validation_report": {}}
        
        logger.info(f"🔍 Iniciando validação universal de {len(products)} produtos...")
        
        validated_products = []
        product_scores = []
        material_code_stats = {}
        
        for i, product in enumerate(products):
            logger.debug(f"Validando produto {i+1}/{len(products)}")
            
            # Calcular pontuação de confiança
            validation_score = self._calculate_product_confidence(product, i+1)
            
            # Gerar estatísticas do produto
            product_stats = self._generate_product_stats(product)
            
            # Armazenar estatísticas por material code
            if product_stats.material_code:
                material_code_stats[product_stats.material_code] = {
                    "colors_found": product_stats.total_colors,
                    "sizes_found": product_stats.total_sizes,
                    "total_quantity": product_stats.total_quantity,
                    "confidence": product_stats.confidence_score,
                    "size_distribution": product_stats.size_distribution,
                    "price_range": product_stats.price_range
                }
            
            validated_product = product.copy()
            validated_product["_validation"] = {
                "confidence_score": validation_score.overall_score,
                "confidence_percentage": f"{validation_score.overall_score:.1f}%",
                "scores_breakdown": {
                    "material_code": f"{validation_score.material_code_score:.1f}%",
                    "category": f"{validation_score.category_score:.1f}%",
                    "colors": f"{validation_score.colors_score:.1f}%",
                    "sizes": f"{validation_score.sizes_score:.1f}%",
                    "prices": f"{validation_score.prices_score:.1f}%"
                },
                "product_stats": {
                    "total_colors": product_stats.total_colors,
                    "total_sizes": product_stats.total_sizes,
                    "total_quantity": product_stats.total_quantity,
                    "size_distribution": product_stats.size_distribution
                },
                "issues_found": validation_score.details.get("issues", []),
                "validation_level": self._get_confidence_level(validation_score.overall_score)
            }
            
            validated_products.append(validated_product)
            product_scores.append(validation_score.overall_score)
        
        # Gerar relatório de validação
        validation_report = self._generate_validation_report(
            validated_products, product_scores, material_code_stats
        )
        
        # Atualizar estatísticas globais
        self.validation_stats["total_products"] = len(products)
        self.validation_stats["average_confidence"] = sum(product_scores) / len(product_scores)
        self.validation_stats["material_code_stats"] = material_code_stats
        
        # Log do relatório
        self._log_validation_summary(validation_report)
        
        return {
            "validated_products": validated_products,
            "validation_report": validation_report
        }
    
    def _calculate_product_confidence(self, product: Dict[str, Any], product_number: int) -> ValidationScore:
        """
        Calcula pontuação de confiança detalhada para um produto
        """
        scores = ValidationScore()
        issues = []
        
        # 1. VALIDAÇÃO MATERIAL CODE (25%)
        material_code = product.get("material_code", "").strip()
        scores.material_code_score = self._validate_material_code_confidence(material_code)
        
        if scores.material_code_score < 50:
            issues.append(f"Material code suspeito: '{material_code}'")
        
        # 2. VALIDAÇÃO CATEGORIA (15%)
        category = product.get("category", "")
        scores.category_score = self._validate_category_confidence(category)
        
        if scores.category_score < 80:
            issues.append(f"Categoria inválida ou suspeita: '{category}'")
        
        # 3. VALIDAÇÃO CORES (25%)
        colors = product.get("colors", [])
        scores.colors_score = self._validate_colors_confidence(colors)
        
        if scores.colors_score < 70:
            issues.append(f"Problemas nas cores: {len(colors)} cores encontradas")
        
        # 4. VALIDAÇÃO TAMANHOS (25%)
        scores.sizes_score = self._validate_sizes_confidence(colors)
        
        if scores.sizes_score < 70:
            issues.append("Problemas nos tamanhos")
        
        # 5. VALIDAÇÃO PREÇOS (10%)
        scores.prices_score = self._validate_prices_confidence(colors)
        
        if scores.prices_score < 50:
            issues.append("Preços suspeitos ou em falta")
        
        # PONTUAÇÃO GERAL (média ponderada)
        scores.overall_score = (
            scores.material_code_score * 0.25 +
            scores.category_score * 0.15 +
            scores.colors_score * 0.25 +
            scores.sizes_score * 0.25 +
            scores.prices_score * 0.10
        )
        
        scores.details = {
            "issues": issues,
            "product_number": product_number,
            "material_code": material_code,
            "colors_count": len(colors)
        }
        
        return scores
    
    def _validate_material_code_confidence(self, material_code: str) -> float:
        """Valida confiança do material code"""
        if not material_code or len(material_code) < 3:
            return 0.0
        
        # Verificar padrões conhecidos
        for pattern_name, pattern in self.material_code_patterns.items():
            if re.match(pattern, material_code.upper()):
                if pattern_name == 'alphanumeric_long':
                    return 95.0  # Padrão mais confiável
                elif pattern_name == 'numeric_long':
                    return 85.0
                elif pattern_name == 'mixed_format':
                    return 75.0
                else:
                    return 60.0
        
        # Verificar se tem caracteres válidos
        if re.match(r'^[A-Z0-9]{4,}$', material_code.upper()):
            return 40.0
        
        return 10.0  # Muito suspeito
    
    def _validate_category_confidence(self, category: str) -> float:
        """Valida confiança da categoria"""
        if not category:
            return 0.0
        
        category_upper = category.upper().strip()
        
        # Categoria válida exata
        if category_upper in [cat.upper() for cat in CATEGORIES]:
            return 100.0
        
        # Verificar semelhanças
        for valid_cat in CATEGORIES:
            if category_upper in valid_cat.upper() or valid_cat.upper() in category_upper:
                return 80.0
        
        # Categorias que podem ser mapeadas
        mappable_keywords = {
            'MAGLIA', 'KNIT', 'SWEATER': 70.0,
            'ABITO', 'DRESS': 70.0,
            'GIACCONE', 'CAPPOTTO', 'COAT', 'JACKET': 70.0,
            'PANTALONE', 'PANT', 'TROUSER': 70.0,
            'CAMICIA', 'SHIRT': 70.0,
            'GONNA', 'SKIRT': 70.0
        }
        
        for keywords, score in mappable_keywords.items():
            if any(keyword in category_upper for keyword in keywords.split()):
                return score
        
        return 20.0  # Categoria desconhecida
    
    def _validate_colors_confidence(self, colors: List[Dict[str, Any]]) -> float:
        """Valida confiança das cores"""
        if not colors:
            return 0.0
        
        total_score = 0.0
        valid_colors = 0
        
        for color in colors:
            color_score = 0.0
            
            # Verificar código da cor
            color_code = str(color.get("color_code", "")).strip()
            if color_code:
                for pattern in self.color_code_patterns.values():
                    if re.match(pattern, color_code.upper()):
                        color_score += 40.0
                        break
                else:
                    color_score += 20.0  # Código presente mas formato desconhecido
            
            # Verificar nome da cor
            color_name = color.get("color_name", "").strip()
            if color_name and len(color_name) > 1:
                color_score += 30.0
            
            # Verificar tamanhos
            sizes = color.get("sizes", [])
            if sizes:
                color_score += 30.0
            
            if color_score > 50:
                valid_colors += 1
                total_score += color_score
        
        if valid_colors == 0:
            return 0.0
        
        return min(100.0, total_score / valid_colors)
    
    def _validate_sizes_confidence(self, colors: List[Dict[str, Any]]) -> float:
        """Valida confiança dos tamanhos"""
        if not colors:
            return 0.0
        
        total_sizes = 0
        valid_sizes = 0
        
        for color in colors:
            for size_info in color.get("sizes", []):
                total_sizes += 1
                size = str(size_info.get("size", "")).strip().upper()
                quantity = size_info.get("quantity", 0)
                
                # Verificar se é um tamanho válido
                size_valid = False
                for pattern in self.size_patterns.values():
                    if re.match(pattern, size):
                        size_valid = True
                        break
                
                # Verificar quantidade
                quantity_valid = isinstance(quantity, (int, float)) and quantity > 0
                
                if size_valid and quantity_valid:
                    valid_sizes += 1
                elif size_valid or quantity_valid:
                    valid_sizes += 0.5
        
        if total_sizes == 0:
            return 0.0
        
        return min(100.0, (valid_sizes / total_sizes) * 100)
    
    def _validate_prices_confidence(self, colors: List[Dict[str, Any]]) -> float:
        """Valida confiança dos preços"""
        if not colors:
            return 0.0
        
        total_prices = 0
        valid_prices = 0
        
        for color in colors:
            # Verificar preço unitário
            unit_price = color.get("unit_price")
            if isinstance(unit_price, (int, float)) and unit_price > 0:
                valid_prices += 1
            total_prices += 1
            
            # Verificar preço de venda (opcional)
            sales_price = color.get("sales_price")
            if isinstance(sales_price, (int, float)) and sales_price > 0:
                valid_prices += 0.5
            total_prices += 0.5
        
        if total_prices == 0:
            return 50.0  # Neutro se não há preços para verificar
        
        return min(100.0, (valid_prices / total_prices) * 100)
    
    def _generate_product_stats(self, product: Dict[str, Any]) -> ProductStats:
        """Gera estatísticas detalhadas de um produto"""
        material_code = product.get("material_code", "N/A")
        colors = product.get("colors", [])
        
        total_colors = len(colors)
        total_sizes = 0
        total_quantity = 0
        size_distribution = Counter()
        prices = []
        
        for color in colors:
            sizes = color.get("sizes", [])
            total_sizes += len(sizes)
            
            for size_info in sizes:
                quantity = size_info.get("quantity", 0)
                if isinstance(quantity, (int, float)):
                    total_quantity += quantity
                    size = size_info.get("size", "")
                    size_distribution[size] += quantity
            
            # Coletar preços
            unit_price = color.get("unit_price", 0)
            if isinstance(unit_price, (int, float)) and unit_price > 0:
                prices.append(unit_price)
        
        price_range = (min(prices), max(prices)) if prices else (0.0, 0.0)
        
        # Calcular confiança baseada nas estatísticas
        confidence = 100.0
        if total_colors == 0:
            confidence -= 30
        if total_sizes == 0:
            confidence -= 30
        if total_quantity == 0:
            confidence -= 20
        if not prices:
            confidence -= 20
        
        return ProductStats(
            material_code=material_code,
            total_colors=total_colors,
            total_sizes=total_sizes,
            total_quantity=int(total_quantity),
            price_range=price_range,
            size_distribution=dict(size_distribution),
            confidence_score=max(0.0, confidence)
        )
    
    def _get_confidence_level(self, score: float) -> str:
        """Determina nível de confiança baseado na pontuação"""
        if score >= 80:
            return "ALTO"
        elif score >= 50:
            return "MÉDIO"
        else:
            return "BAIXO"
    
    def _generate_validation_report(
        self, 
        validated_products: List[Dict[str, Any]], 
        scores: List[float],
        material_code_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gera relatório detalhado de validação"""
        
        # Classificar produtos por confiança
        high_confidence = sum(1 for score in scores if score >= 80)
        medium_confidence = sum(1 for score in scores if 50 <= score < 80)
        low_confidence = sum(1 for score in scores if score < 50)
        
        # Problemas mais comuns
        all_issues = []
        for product in validated_products:
            issues = product.get("_validation", {}).get("issues_found", [])
            all_issues.extend(issues)
        
        common_issues = Counter(all_issues).most_common(5)
        
        # Estatísticas de material codes
        material_stats_summary = {}
        for code, stats in material_code_stats.items():
            material_stats_summary[code] = {
                "colors": stats["colors_found"],
                "sizes": stats["sizes_found"], 
                "total_qty": stats["total_quantity"],
                "confidence": f"{stats['confidence']:.1f}%",
                "top_sizes": dict(Counter(stats["size_distribution"]).most_common(3))
            }
        
        return {
            "summary": {
                "total_products": len(validated_products),
                "average_confidence": f"{sum(scores)/len(scores):.1f}%",
                "confidence_distribution": {
                    "high_confidence": f"{high_confidence} ({high_confidence/len(scores)*100:.1f}%)",
                    "medium_confidence": f"{medium_confidence} ({medium_confidence/len(scores)*100:.1f}%)",
                    "low_confidence": f"{low_confidence} ({low_confidence/len(scores)*100:.1f}%)"
                }
            },
            "material_code_analysis": material_stats_summary,
            "common_issues": [{"issue": issue, "count": count} for issue, count in common_issues],
            "validation_method": "universal_confidence_scoring",
            "recommendations": self._generate_recommendations(scores, common_issues)
        }
    
    def _generate_recommendations(
        self, 
        scores: List[float], 
        common_issues: List[Tuple[str, int]]
    ) -> List[str]:
        """Gera recomendações baseadas na validação"""
        recommendations = []
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score < 70:
            recommendations.append("Confiança geral baixa - revisar extração manual")
        
        if len([s for s in scores if s < 50]) > len(scores) * 0.3:
            recommendations.append("Muitos produtos com baixa confiança - verificar qualidade do PDF")
        
        # Recomendações baseadas em problemas comuns
        for issue, count in common_issues[:3]:
            if "Material code" in issue and count > 2:
                recommendations.append("Verificar padrões de material codes no documento")
            elif "Categoria" in issue and count > 2:
                recommendations.append("Melhorar mapeamento de categorias")
            elif "Preços" in issue and count > 2:
                recommendations.append("Verificar formatação de preços no documento")
        
        if not recommendations:
            recommendations.append("Qualidade de extração boa - nenhuma ação necessária")
        
        return recommendations
    
    def _log_validation_summary(self, report: Dict[str, Any]):
        """Log resumido dos resultados de validação"""
        summary = report["summary"]
        
        logger.info("=" * 70)
        logger.info("📊 RELATÓRIO DE VALIDAÇÃO UNIVERSAL")
        logger.info("=" * 70)
        logger.info(f"   Total de produtos: {summary['total_products']}")
        logger.info(f"   Confiança média: {summary['average_confidence']}")
        logger.info("   Distribuição de confiança:")
        logger.info(f"     🟢 Alta: {summary['confidence_distribution']['high_confidence']}")
        logger.info(f"     🟡 Média: {summary['confidence_distribution']['medium_confidence']}")
        logger.info(f"     🔴 Baixa: {summary['confidence_distribution']['low_confidence']}")
        
        if report["material_code_analysis"]:
            logger.info(f"\n📋 ANÁLISE POR MATERIAL CODE ({len(report['material_code_analysis'])} códigos):")
            for code, stats in list(report["material_code_analysis"].items())[:5]:
                logger.info(f"   • {code}: {stats['colors']} cores, {stats['sizes']} tamanhos, "
                           f"{stats['total_qty']} peças (confiança: {stats['confidence']})")
        
        if report["common_issues"]:
            logger.info(f"\n⚠️  PROBLEMAS MAIS COMUNS:")
            for issue in report["common_issues"][:3]:
                logger.info(f"   • {issue['issue']} ({issue['count']}x)")
        
        if report["recommendations"]:
            logger.info(f"\n💡 RECOMENDAÇÕES:")
            for rec in report["recommendations"]:
                logger.info(f"   • {rec}")
        
        logger.info("=" * 70)
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Retorna relatório detalhado de todas as validações"""
        return {
            "global_statistics": self.validation_stats,
            "validation_patterns": {
                "material_codes": self.material_code_patterns,
                "sizes": self.size_patterns,
                "colors": self.color_code_patterns
            }
        }