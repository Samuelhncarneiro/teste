# app/extractors/validation_agent.py
import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import google.generativeai as genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.data.reference_data import CATEGORIES

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Resultado da validação com métricas detalhadas"""
    products: List[Dict[str, Any]]
    confidence_score: float
    missing_fields: List[str]
    validation_errors: List[str]
    total_pages_processed: int
    extraction_method: str
    completeness_score: float
    consistency_score: float
    visual_completeness_score: float
    density_score: float
    recommendations: List[str]

class ValidationAgent:
    """Agent responsável por validar e corrigir extrações de produtos"""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    async def validate_extraction(self, 
                                extracted_products: List[Dict], 
                                original_context: Dict,
                                pdf_pages: List[Image.Image],
                                layout_analysis: Dict = None) -> ValidationResult:
        """
        Valida a extração e tenta corrigir problemas identificados
        
        Args:
            extracted_products: Lista de produtos extraídos
            original_context: Contexto original do documento
            pdf_pages: Lista de imagens das páginas do PDF
            layout_analysis: Análise de layout do documento
            
        Returns:
            ValidationResult: Resultado completo da validação
        """
        logger.info("Iniciando validação da extração...")
        
        validation_errors = []
        missing_fields = []
        recommendations = []
        
        # 1. Validar estrutura dos produtos
        for i, product in enumerate(extracted_products):
            if not self._validate_product_structure(product):
                validation_errors.append(f"Produto {i+1}: Estrutura inválida")
            
            # Verificar campos obrigatórios
            required_fields = ['product_name', 'material_code', 'colors']
            for field in required_fields:
                if not product.get(field):
                    missing_fields.append(f"Produto {i+1}: Campo '{field}' em falta")
        
        # 2. Calcular métricas de qualidade
        completeness_score = self._calculate_completeness_score(extracted_products)
        consistency_score = self._calculate_consistency_score(extracted_products)
        visual_completeness_score = await self._analyze_visual_completeness(
            extracted_products, pdf_pages, original_context
        )
        density_score = self._calculate_density_score(extracted_products, original_context)
        
        # 3. Calcular score de confiança global
        confidence_score = self._calculate_overall_confidence(
            completeness_score, consistency_score, visual_completeness_score, density_score
        )
        
        logger.info(f"Scores de validação - Completude: {completeness_score:.2f}, "
                   f"Consistência: {consistency_score:.2f}, "
                   f"Visual: {visual_completeness_score:.2f}, "
                   f"Densidade: {density_score:.2f}")
        
        # 4. Gerar recomendações
        recommendations = self._generate_recommendations(
            completeness_score, consistency_score, visual_completeness_score, 
            density_score, validation_errors, missing_fields
        )
        
        # 5. Tentar correção se confiança baixa
        corrected_products = extracted_products
        extraction_method = "original"
        
        if confidence_score < 0.7:
            logger.warning(f"Confiança baixa ({confidence_score:.2f}), tentando correção...")
            corrected_products = await self._attempt_correction(
                extracted_products, original_context, pdf_pages, layout_analysis
            )
            
            # Recalcular métricas após correção
            completeness_score = self._calculate_completeness_score(corrected_products)
            consistency_score = self._calculate_consistency_score(corrected_products)
            visual_completeness_score = await self._analyze_visual_completeness(
                corrected_products, pdf_pages, original_context
            )
            density_score = self._calculate_density_score(corrected_products, original_context)
            
            confidence_score = self._calculate_overall_confidence(
                completeness_score, consistency_score, visual_completeness_score, density_score
            )
            
            extraction_method = "corrected"
            logger.info(f"Após correção - Nova confiança: {confidence_score:.2f}")
        
        return ValidationResult(
            products=corrected_products,
            confidence_score=confidence_score,
            missing_fields=missing_fields,
            validation_errors=validation_errors,
            total_pages_processed=len(pdf_pages),
            extraction_method=extraction_method,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            visual_completeness_score=visual_completeness_score,
            density_score=density_score,
            recommendations=recommendations
        )
    
    def _validate_product_structure(self, product: Dict) -> bool:
        """Valida se o produto tem estrutura mínima necessária"""
        if not isinstance(product, dict):
            return False
        
        # Verificar campos essenciais
        if not product.get('product_name') or not product.get('colors'):
            return False
        
        # Verificar estrutura das cores
        colors = product.get('colors', [])
        if not isinstance(colors, list) or len(colors) == 0:
            return False
        
        for color in colors:
            if not isinstance(color, dict):
                return False
            if not color.get('sizes') or not isinstance(color.get('sizes'), list):
                return False
        
        return True
    
    def _calculate_completeness_score(self, products: List[Dict]) -> float:
        """Calcula score de completude dos dados"""
        if not products:
            return 0.0
        
        complete_products = sum(1 for p in products if self._is_product_complete(p))
        return complete_products / len(products)
    
    def _is_product_complete(self, product: Dict) -> bool:
        """Verifica se um produto tem todos os campos importantes"""
        required_fields = ['product_name', 'material_code', 'colors']
        
        for field in required_fields:
            if not product.get(field):
                return False
        
        # Verificar se cores têm tamanhos e quantidades
        colors = product.get('colors', [])
        for color in colors:
            sizes = color.get('sizes', [])
            if not sizes:
                return False
            
            # Pelo menos um tamanho deve ter quantidade
            has_quantity = any(s.get('quantity', 0) > 0 for s in sizes)
            if not has_quantity:
                return False
        
        return True
    
    def _calculate_consistency_score(self, products: List[Dict]) -> float:
        """Verifica consistência dos códigos e estruturas"""
        if not products:
            return 0.0
        
        # Analisar padrões de códigos
        codes = [p.get('material_code', '') for p in products if p.get('material_code')]
        if not codes:
            return 0.0
        
        # Verificar se códigos seguem padrão similar
        pattern_matches = 0
        for code in codes:
            # Padrões comuns: CF5015E0624, 251204, T3216, etc.
            if (re.match(r'^[A-Z]{2}\d{3,4}[A-Z]*\d*$', code) or  # CF5015E0624
                re.match(r'^\d{6,}$', code) or                      # 251204
                re.match(r'^[A-Z]\d{4}$', code)):                   # T3216
                pattern_matches += 1
        
        return pattern_matches / len(codes) if codes else 0.0
    
    async def _analyze_visual_completeness(self, 
                                         products: List[Dict], 
                                         pages: List[Image.Image],
                                         context: Dict) -> float:
        """Analisa visualmente se há produtos não capturados"""
        if not pages:
            return 0.5  # Score neutro se não há páginas
        
        try:
            # Análise visual da primeira página como referência
            prompt = f"""
            Analise esta imagem de documento comercial e faça uma avaliação:
            
            1. Quantos produtos/itens únicos você consegue identificar visualmente?
            2. Há tabelas ou seções que parecem conter produtos não processados?
            3. Existem linhas de produtos que parecem incompletas ou cortadas?
            4. A densidade de produtos na imagem condiz com o número extraído?
            
            Informações do contexto:
            - Tipo de documento: {context.get('document_type', 'N/A')}
            - Fornecedor: {context.get('supplier', 'N/A')}
            - Produtos extraídos: {len(products)}
            
            Responda APENAS com um número decimal entre 0.0 e 1.0 indicando a completude:
            - 1.0: Todos os produtos visíveis foram capturados adequadamente
            - 0.8: A maioria foi capturada, algumas omissões menores
            - 0.6: Aproximadamente metade foi capturada adequadamente
            - 0.4: Muitos produtos visíveis parecem não ter sido capturados
            - 0.0: A maioria dos produtos visíveis não foi capturada
            
            Resposta (apenas o número decimal):
            """
            
            response = self.model.generate_content([prompt, pages[0]])
            score_text = response.text.strip()
            
            # Extrair número da resposta
            score_match = re.search(r'(\d*\.?\d+)', score_text)
            
            if score_match:
                score = float(score_match.group(1))
                return min(1.0, max(0.0, score))
            
            return 0.5  # Score neutro se não conseguir analisar
            
        except Exception as e:
            logger.warning(f"Erro na análise visual: {e}")
            return 0.5
    
    def _calculate_density_score(self, products: List[Dict], context: Dict) -> float:
        """Calcula densidade de informação baseada no contexto"""
        if not products:
            return 0.0
        
        # Estimar densidade baseada no tipo de documento
        doc_type = context.get('document_type', '').lower()
        
        expected_density = {
            'nota de encomenda': 0.8,
            'pedido': 0.7,
            'orçamento': 0.6,
            'fatura': 0.9
        }
        
        base_density = expected_density.get(doc_type, 0.7)
        
        # Calcular densidade real
        complete_fields = 0
        total_fields = 0
        
        for product in products:
            # Campos por produto
            product_fields = ['product_name', 'material_code', 'category', 'brand']
            for field in product_fields:
                total_fields += 1
                if product.get(field):
                    complete_fields += 1
            
            # Campos por cor
            colors = product.get('colors', [])
            for color in colors:
                color_fields = ['color_code', 'color_name', 'unit_price']
                for field in color_fields:
                    total_fields += 1
                    if color.get(field):
                        complete_fields += 1
                
                # Tamanhos
                sizes = color.get('sizes', [])
                for size in sizes:
                    size_fields = ['size', 'quantity']
                    for field in size_fields:
                        total_fields += 1
                        if size.get(field):
                            complete_fields += 1
        
        actual_density = complete_fields / total_fields if total_fields > 0 else 0.0
        density_ratio = actual_density / base_density
        
        return min(1.0, density_ratio)
    
    def _calculate_overall_confidence(self, completeness: float, consistency: float, 
                                    visual: float, density: float) -> float:
        """Calcula score de confiança global ponderado"""
        weights = {
            'completeness': 0.3,
            'consistency': 0.2,
            'visual': 0.3,
            'density': 0.2
        }
        
        overall = (completeness * weights['completeness'] + 
                  consistency * weights['consistency'] + 
                  visual * weights['visual'] + 
                  density * weights['density'])
        
        return overall
    
    def _generate_recommendations(self, completeness: float, consistency: float,
                                visual: float, density: float, 
                                errors: List[str], missing: List[str]) -> List[str]:
        """Gera recomendações baseadas nas métricas"""
        recommendations = []
        
        if completeness < 0.6:
            recommendations.append("Produtos com dados incompletos - verificar campos obrigatórios")
        
        if consistency < 0.5:
            recommendations.append("Inconsistência nos códigos de produto - verificar padrões")
        
        if visual < 0.6:
            recommendations.append("Possíveis produtos não extraídos - verificar visualmente o documento")
        
        if density < 0.5:
            recommendations.append("Baixa densidade de informação - pode haver campos não capturados")
        
        if len(errors) > len(missing):
            recommendations.append("Muitos erros de estrutura - considerar ajustar estratégia de extração")
        
        if not recommendations:
            recommendations.append("Extração com boa qualidade - proceder com confiança")
        
        return recommendations
    
    async def _attempt_correction(self, 
                                products: List[Dict], 
                                context: Dict,
                                pages: List[Image.Image],
                                layout_analysis: Dict = None) -> List[Dict]:
        """Tenta corrigir problemas identificados na extração"""
        corrected_products = products.copy()
        
        try:
            # Estratégia de correção baseada no layout
            layout_info = layout_analysis or {}
            layout_type = layout_info.get('layout_type', 'UNKNOWN')
            
            correction_prompt = f"""
            CORREÇÃO ESPECIALIZADA DE EXTRAÇÃO DE PRODUTOS
            
            A extração inicial pode ter perdido produtos ou dados. Analise esta imagem com foco em:
            
            1. Produtos que podem ter sido perdidos completamente
            2. Campos de produtos existentes que podem estar vazios
            3. Códigos de produto isolados sem dados associados
            4. Tamanhos/cores que podem não ter sido mapeados corretamente
            
            CONTEXTO DO DOCUMENTO:
            - Tipo: {context.get('document_type', 'N/A')}
            - Fornecedor: {context.get('supplier', 'N/A')}
            - Layout detectado: {layout_type}
            - Produtos já extraídos: {len(products)}
            
            ESTRATÉGIA DE CORREÇÃO:
            - Foque em áreas que podem ter sido negligenciadas
            - Procure padrões de dados não capturados
            - Identifique produtos com estrutura incompleta
            
            Se encontrar produtos adicionais ou correções, retorne em formato JSON:
            {{
                "additional_products": [...],
                "corrections": [
                    {{
                        "product_index": 0,
                        "field": "campo_a_corrigir",
                        "corrected_value": "valor_corrigido"
                    }}
                ]
            }}
            
            Se não encontrar melhorias, retorne:
            {{"additional_products": [], "corrections": []}}
            """
            
            response = self.model.generate_content([correction_prompt, pages[0]])
            
            # Extrair correções
            correction_data = self._extract_json_from_text(response.text)
            
            if correction_data:
                # Aplicar produtos adicionais
                additional_products = correction_data.get('additional_products', [])
                if additional_products:
                    logger.info(f"Adicionando {len(additional_products)} produtos encontrados na correção")
                    corrected_products.extend(additional_products)
                
                # Aplicar correções
                corrections = correction_data.get('corrections', [])
                for correction in corrections:
                    try:
                        idx = correction.get('product_index', -1)
                        field = correction.get('field', '')
                        value = correction.get('corrected_value', '')
                        
                        if 0 <= idx < len(corrected_products) and field and value:
                            corrected_products[idx][field] = value
                            logger.info(f"Correção aplicada ao produto {idx}: {field} = {value}")
                    except Exception as e:
                        logger.warning(f"Erro ao aplicar correção: {e}")
            
        except Exception as e:
            logger.warning(f"Erro na tentativa de correção: {e}")
        
        return corrected_products
    
    def _extract_json_from_text(self, text: str) -> Dict:
        """Extrai JSON do texto de resposta"""
        try:
            # Procurar por JSON com markdown
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Procurar por JSON sem markdown
            json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            return {}
        except Exception as e:
            logger.warning(f"Erro ao extrair JSON: {e}")
            return {}