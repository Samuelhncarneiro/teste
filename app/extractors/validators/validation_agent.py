# app/extractors/validation_agent.py (VERSÃO MELHORADA)

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
    # NOVOS CAMPOS ESPECÍFICOS
    sizes_corrected: int
    quantities_corrected: int
    products_merged: int
    corrections_applied: List[str]

class ValidationAgent:
    """Agent responsável por validar e corrigir extrações de produtos com foco em problemas específicos"""
    
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
        Valida a extração com foco em problemas específicos:
        1. Tamanhos incompletos ou incorretos
        2. Quantidades todas iguais a 1
        3. Produtos duplicados por cor
        4. Alinhamento incorreto de colunas
        """
        logger.info("🔍 Iniciando validação específica...")
        
        validation_errors = []
        missing_fields = []
        recommendations = []
        corrections_applied = []
        sizes_corrected = 0
        quantities_corrected = 0
        products_merged = 0
        
        # 1. CORREÇÃO ESPECÍFICA: Agrupar produtos por cor
        logger.info("🎨 Verificando agrupamento de produtos por cor...")
        color_grouped_products, merge_corrections = await self._fix_color_grouping(
            extracted_products, pdf_pages
        )
        corrections_applied.extend(merge_corrections)
        products_merged = len(extracted_products) - len(color_grouped_products)
        
        # 2. CORREÇÃO ESPECÍFICA: Tamanhos e quantidades
        logger.info("📏 Corrigindo tamanhos e quantidades...")
        size_corrected_products, size_corrections = await self._fix_sizes_and_quantities(
            color_grouped_products, pdf_pages
        )
        corrections_applied.extend(size_corrections)
        sizes_corrected = len([c for c in size_corrections if 'tamanho' in c.lower()])
        quantities_corrected = len([c for c in size_corrections if 'quantidade' in c.lower()])
        
        # 3. Validações originais
        completeness_score = self._calculate_completeness_score(size_corrected_products)
        consistency_score = self._calculate_consistency_score(size_corrected_products)
        visual_completeness_score = await self._analyze_visual_completeness(
            size_corrected_products, pdf_pages, original_context
        )
        density_score = self._calculate_density_score(size_corrected_products, original_context)
        
        confidence_score = self._calculate_overall_confidence(
            completeness_score, consistency_score, visual_completeness_score, density_score
        )
        
        # 4. Gerar recomendações específicas
        recommendations = self._generate_specific_recommendations(
            extracted_products, size_corrected_products, corrections_applied
        )
        
        logger.info(f"✅ Validação concluída:")
        logger.info(f"   - Produtos originais: {len(extracted_products)}")
        logger.info(f"   - Produtos finais: {len(size_corrected_products)}")
        logger.info(f"   - Produtos agrupados: {products_merged}")
        logger.info(f"   - Tamanhos corrigidos: {sizes_corrected}")
        logger.info(f"   - Quantidades corrigidas: {quantities_corrected}")
        logger.info(f"   - Confiança: {confidence_score:.2f}")
        
        return ValidationResult(
            products=size_corrected_products,
            confidence_score=confidence_score,
            missing_fields=missing_fields,
            validation_errors=validation_errors,
            total_pages_processed=len(pdf_pages),
            extraction_method="corrected",
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            visual_completeness_score=visual_completeness_score,
            density_score=density_score,
            recommendations=recommendations,
            sizes_corrected=sizes_corrected,
            quantities_corrected=quantities_corrected,
            products_merged=products_merged,
            corrections_applied=corrections_applied
        )
    
    async def _fix_color_grouping(self, 
                                products: List[Dict], 
                                images: List[Image.Image]) -> Tuple[List[Dict], List[str]]:
        """
        Corrige agrupamento de produtos por cor (ex: CF5271MA96E.1 e CF5271MA96E.2)
        """
        corrections = []
        
        # Agrupar por código base
        product_groups = {}
        for product in products:
            material_code = product.get('material_code', '')
            base_code = re.sub(r'\.\d+$', '', material_code)  # Remove .1, .2, etc.
            
            if base_code not in product_groups:
                product_groups[base_code] = []
            product_groups[base_code].append(product)
        
        merged_products = []
        
        for base_code, group in product_groups.items():
            if len(group) == 1:
                merged_products.append(group[0])
            else:
                # Verificar se devem ser agrupados
                merged = await self._merge_product_variants(base_code, group, images)
                if merged:
                    merged_products.append(merged)
                    corrections.append(f"Agrupadas {len(group)} variantes de {base_code} por cor")
                else:
                    merged_products.extend(group)
        
        return merged_products, corrections
    
    async def _merge_product_variants(self, 
                                    base_code: str, 
                                    variants: List[Dict],
                                    images: List[Image.Image]) -> Optional[Dict]:
        """
        Faz merge de variantes de produto por cor
        """
        try:
            if not images or len(variants) < 2:
                return None
            
            # Extrair todas as cores das variantes
            all_colors = []
            base_product = variants[0].copy()
            
            for variant in variants:
                variant_colors = variant.get('colors', [])
                all_colors.extend(variant_colors)
            
            # Verificar visualmente se é mesmo o mesmo produto
            merge_prompt = f"""
            VERIFICAÇÃO DE AGRUPAMENTO - {base_code}
            
            Encontradas {len(variants)} variantes com códigos similares:
            {[v.get('material_code') for v in variants]}
            
            Analise a imagem e determine se estas variantes representam o MESMO produto com cores diferentes.
            
            Critérios para AGRUPAR:
            1. Mesmo nome de produto
            2. Mesmo formato/tipo de item
            3. Apenas cores diferentes
            4. Código base idêntico
            
            Responda apenas: "AGRUPAR" ou "SEPARAR"
            """
            
            response = self.model.generate_content([merge_prompt, images[0]])
            
            if "AGRUPAR" in response.text.upper():
                # Fazer merge
                merged_product = base_product
                merged_product['material_code'] = base_code
                merged_product['colors'] = all_colors
                
                logger.info(f"✅ Agrupando {len(variants)} variantes de {base_code}")
                return merged_product
            
            return None
            
        except Exception as e:
            logger.warning(f"Erro no merge de {base_code}: {e}")
            return None
    
    async def _fix_sizes_and_quantities(self, 
                                       products: List[Dict],
                                       images: List[Image.Image]) -> Tuple[List[Dict], List[str]]:
        """
        Corrige tamanhos incompletos e quantidades incorretas
        """
        corrections = []
        corrected_products = []
        
        for product in products:
            material_code = product.get('material_code', '')
            logger.debug(f"🔍 Validando tamanhos/quantidades: {material_code}")
            
            corrected_product, product_corrections = await self._fix_single_product(
                product, images
            )
            
            corrected_products.append(corrected_product)
            corrections.extend(product_corrections)
        
        return corrected_products, corrections
    
    async def _fix_single_product(self, 
                                product: Dict,
                                images: List[Image.Image]) -> Tuple[Dict, List[str]]:
        """
        Corrige um produto específico analisando a imagem
        """
        corrections = []
        material_code = product.get('material_code', '')
        
        try:
            if not images:
                return product, corrections
            
            # Análise específica para este produto
            fix_prompt = f"""
            CORREÇÃO ESPECÍFICA DE PRODUTO - {material_code}
            
            Produto atual:
            {json.dumps(product, indent=2)}
            
            PROBLEMAS A VERIFICAR:
            1. TAMANHOS INCOMPLETOS: Faltam XS ou XL que estão visíveis?
            2. QUANTIDADES INCORRETAS: Todas as quantidades são 1 quando deveriam ser diferentes?
            3. ALINHAMENTO: Os tamanhos correspondem às colunas corretas?
            
            TAREFA:
            1. Localize este código ({material_code}) na imagem
            2. Leia TODOS os tamanhos da linha (XS, S, M, L, XL, etc.)
            3. Leia as quantidades EXATAS (incluindo 0 para tamanhos sem stock)
            4. Verifique o alinhamento posicional
            
            RESPOSTA EM JSON:
            {{
                "needs_correction": true/false,
                "corrections": {{
                    "material_code": "{material_code}",
                    "colors": [
                        {{
                            "color_code": "código_da_cor",
                            "color_name": "nome_da_cor",
                            "unit_price": 0.0,
                            "sizes": [
                                {{"size": "XS", "quantity": 0}},
                                {{"size": "S", "quantity": 1}},
                                {{"size": "M", "quantity": 1}},
                                {{"size": "L", "quantity": 1}},
                                {{"size": "XL", "quantity": 0}}
                            ]
                        }}
                    ]
                }},
                "changes_made": [
                    "Lista de mudanças específicas feitas"
                ]
            }}
            
            IMPORTANTE: 
            - Incluir TODOS os tamanhos visíveis (mesmo quantidade 0)
            - Ler quantidades EXATAS da imagem
            - Só corrigir se tiver certeza
            """
            
            # Tentar com múltiplas imagens
            for image in images:
                try:
                    response = self.model.generate_content([fix_prompt, image])
                    analysis = self._extract_json_safely(response.text)
                    
                    if analysis and analysis.get('needs_correction'):
                        corrections_data = analysis.get('corrections', {})
                        changes_made = analysis.get('changes_made', [])
                        
                        if corrections_data and changes_made:
                            # Aplicar correções
                            corrected_product = product.copy()
                            corrected_product.update(corrections_data)
                            
                            for change in changes_made:
                                corrections.append(f"{material_code}: {change}")
                                logger.info(f"🔧 {material_code}: {change}")
                            
                            return corrected_product, corrections
                    
                except Exception as e:
                    logger.warning(f"Erro ao analisar {material_code} na imagem: {e}")
                    continue
            
            return product, corrections
            
        except Exception as e:
            logger.warning(f"Erro na correção de {material_code}: {e}")
            return product, corrections
    
    def _generate_specific_recommendations(self, 
                                         original: List[Dict],
                                         corrected: List[Dict],
                                         corrections: List[str]) -> List[str]:
        """
        Gera recomendações específicas baseadas nas correções aplicadas
        """
        recommendations = []
        
        # Analisar produtos agrupados
        if len(corrected) < len(original):
            merged_count = len(original) - len(corrected)
            recommendations.append(f"Agrupados {merged_count} produtos duplicados por cor")
        
        # Analisar tipos de correções
        size_corrections = [c for c in corrections if 'tamanho' in c.lower()]
        quantity_corrections = [c for c in corrections if 'quantidade' in c.lower()]
        
        if size_corrections:
            recommendations.append(f"Corrigidos tamanhos em {len(size_corrections)} produtos")
        
        if quantity_corrections:
            recommendations.append(f"Corrigidas quantidades em {len(quantity_corrections)} produtos")
        
        # Verificar se ainda há problemas
        uniform_quantities = 0
        for product in corrected:
            for color in product.get('colors', []):
                sizes = color.get('sizes', [])
                quantities = [s.get('quantity', 0) for s in sizes]
                if len(set(quantities)) == 1 and quantities[0] == 1:
                    uniform_quantities += 1
        
        if uniform_quantities > 0:
            recommendations.append(f"ATENÇÃO: {uniform_quantities} cores ainda têm quantidades uniformes (possível erro)")
        
        if not recommendations:
            recommendations.append("Extração parece estar correta - sem correções necessárias")
        
        return recommendations
    
    # Métodos auxiliares originais (mantidos)
    def _calculate_completeness_score(self, products: List[Dict]) -> float:
        if not products:
            return 0.0
        complete_products = sum(1 for p in products if self._is_product_complete(p))
        return complete_products / len(products)
    
    def _is_product_complete(self, product: Dict) -> bool:
        required_fields = ['product_name', 'material_code', 'colors']
        for field in required_fields:
            if not product.get(field):
                return False
        
        colors = product.get('colors', [])
        for color in colors:
            sizes = color.get('sizes', [])
            if not sizes:
                return False
            has_quantity = any(s.get('quantity', 0) > 0 for s in sizes)
            if not has_quantity:
                return False
        return True
    
    def _calculate_consistency_score(self, products: List[Dict]) -> float:
        if not products:
            return 0.0
        codes = [p.get('material_code', '') for p in products if p.get('material_code')]
        if not codes:
            return 0.0
        
        pattern_matches = 0
        for code in codes:
            if (re.match(r'^[A-Z]{2}\d{3,4}[A-Z]*\d*$', code) or
                re.match(r'^\d{6,}$', code) or
                re.match(r'^[A-Z]\d{4}$', code)):
                pattern_matches += 1
        
        return pattern_matches / len(codes) if codes else 0.0
    
    async def _analyze_visual_completeness(self, products: List[Dict], 
                                         pages: List[Image.Image],
                                         context: Dict) -> float:
        if not pages:
            return 0.5
        
        try:
            prompt = f"""
            Analise esta imagem e avalie a qualidade da extração:
            
            Produtos extraídos: {len(products)}
            
            1. Todos os produtos visíveis foram capturados?
            2. Os tamanhos parecem completos (XS, S, M, L, XL)?
            3. As quantidades parecem variadas (não todas iguais a 1)?
            
            Responda com score 0.0-1.0:
            - 1.0: Extração parece completa e precisa
            - 0.8: Boa qualidade, pequenos problemas
            - 0.6: Qualidade média
            - 0.4: Problemas significativos
            - 0.0: Muitos problemas
            
            Resposta (apenas número):
            """
            
            response = self.model.generate_content([prompt, pages[0]])
            score_text = response.text.strip()
            score_match = re.search(r'(\d*\.?\d+)', score_text)
            
            if score_match:
                return min(1.0, max(0.0, float(score_match.group(1))))
            return 0.5
            
        except Exception as e:
            logger.warning(f"Erro na análise visual: {e}")
            return 0.5
    
    def _calculate_density_score(self, products: List[Dict], context: Dict) -> float:
        if not products:
            return 0.0
        
        doc_type = context.get('document_type', '').lower()
        expected_density = {
            'nota de encomenda': 0.8,
            'pedido': 0.7,
            'orçamento': 0.6,
            'fatura': 0.9
        }
        
        base_density = expected_density.get(doc_type, 0.7)
        
        complete_fields = 0
        total_fields = 0
        
        for product in products:
            product_fields = ['product_name', 'material_code', 'category', 'brand']
            for field in product_fields:
                total_fields += 1
                if product.get(field):
                    complete_fields += 1
            
            colors = product.get('colors', [])
            for color in colors:
                color_fields = ['color_code', 'color_name', 'unit_price']
                for field in color_fields:
                    total_fields += 1
                    if color.get(field):
                        complete_fields += 1
                
                sizes = color.get('sizes', [])
                for size in sizes:
                    size_fields = ['size', 'quantity']
                    for field in size_fields:
                        total_fields += 1
                        if size.get(field) is not None:  # Importante: aceitar 0
                            complete_fields += 1
        
        actual_density = complete_fields / total_fields if total_fields > 0 else 0.0
        density_ratio = actual_density / base_density
        
        return min(1.0, density_ratio)
    
    def _calculate_overall_confidence(self, completeness: float, consistency: float, 
                                    visual: float, density: float) -> float:
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
    
    def _extract_json_safely(self, text: str) -> Optional[Dict]:
        try:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            return None
            
        except Exception as e:
            logger.warning(f"Erro ao extrair JSON: {e}")
            return None