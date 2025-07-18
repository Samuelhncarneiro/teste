# app/extractors/size_color_validation_agent.py

import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import google.generativeai as genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

@dataclass
class SizeColorValidationResult:
    """Resultado da validação de tamanhos e cores"""
    corrected_products: List[Dict[str, Any]]
    corrections_made: List[str]
    confidence_score: float
    validation_errors: List[str]
    size_alignment_issues: List[str]
    color_grouping_issues: List[str]

class SizeColorValidationAgent:
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    async def validate_and_correct(self, 
                                 products: List[Dict], 
                                 page_images: List[Image.Image],
                                 confidence_threshold: float = 0.7) -> SizeColorValidationResult:
        """
        Valida e corrige problemas de tamanhos e cores
        
        Args:
            products: Lista de produtos extraídos
            page_images: Imagens das páginas para análise visual
            confidence_threshold: Limite de confiança para acionar correções
            
        Returns:
            SizeColorValidationResult: Resultado com correções aplicadas
        """
        logger.info("🔍 Iniciando validação de tamanhos e cores...")
        
        corrections_made = []
        validation_errors = []
        size_alignment_issues = []
        color_grouping_issues = []
        
        # 1. Detectar problemas de agrupamento de cores
        color_grouped_products = await self._group_products_by_color_issues(products, page_images)
        
        # 2. Detectar problemas de alinhamento de tamanhos
        size_corrected_products = await self._correct_size_alignment_issues(
            color_grouped_products, page_images
        )
        
        # 3. Validar consistência final
        final_products, final_corrections = await self._final_consistency_check(
            size_corrected_products, page_images
        )
        
        # 4. Calcular confiança
        confidence_score = self._calculate_correction_confidence(
            products, final_products, corrections_made
        )
        
        logger.info(f"✅ Validação concluída:")
        logger.info(f"   - Produtos originais: {len(products)}")
        logger.info(f"   - Produtos finais: {len(final_products)}")
        logger.info(f"   - Correções aplicadas: {len(corrections_made)}")
        logger.info(f"   - Confiança: {confidence_score:.2f}")
        
        return SizeColorValidationResult(
            corrected_products=final_products,
            corrections_made=corrections_made,
            confidence_score=confidence_score,
            validation_errors=validation_errors,
            size_alignment_issues=size_alignment_issues,
            color_grouping_issues=color_grouping_issues
        )
    
    async def _group_products_by_color_issues(self, 
                                            products: List[Dict], 
                                            images: List[Image.Image]) -> List[Dict]:
        """
        Detecta e corrige problemas de agrupamento de cores
        Exemplo: CF5271MA96E com duas cores diferentes deve ser um produto só
        """
        logger.info("🎨 Analisando agrupamento de cores...")
        
        # Agrupar por código base de produto
        product_groups = {}
        for product in products:
            base_code = product.get('material_code', '')
            
            # Remover sufixos que podem indicar variantes (.1, .2, etc)
            clean_code = re.sub(r'\.\d+$', '', base_code)
            
            if clean_code not in product_groups:
                product_groups[clean_code] = []
            product_groups[clean_code].append(product)
        
        corrected_products = []
        
        for base_code, group_products in product_groups.items():
            if len(group_products) == 1:
                # Produto único, sem problemas de agrupamento
                corrected_products.append(group_products[0])
            else:
                # Múltiplos produtos com mesmo código base - analisar se devem ser agrupados
                logger.info(f"🔍 Analisando agrupamento para código {base_code}: {len(group_products)} variantes")
                
                merged_product = await self._analyze_and_merge_product_variants(
                    base_code, group_products, images
                )
                
                if merged_product:
                    corrected_products.append(merged_product)
                else:
                    # Se não conseguir merge, manter produtos separados
                    corrected_products.extend(group_products)
        
        return corrected_products
    
    async def _analyze_and_merge_product_variants(self, 
                                                base_code: str,
                                                variants: List[Dict],
                                                images: List[Image.Image]) -> Optional[Dict]:
        """
        Analisa se variantes de produto devem ser agrupadas e faz o merge
        """
        try:
            if not images:
                return None
            
            # Extrair informações das variantes
            variant_info = []
            for variant in variants:
                colors = variant.get('colors', [])
                for color in colors:
                    variant_info.append({
                        'color_code': color.get('color_code', ''),
                        'color_name': color.get('color_name', ''),
                        'sizes': color.get('sizes', []),
                        'unit_price': color.get('unit_price', 0)
                    })
            
            # Análise visual para confirmar se é o mesmo produto
            analysis_prompt = f"""
            ANÁLISE DE AGRUPAMENTO DE PRODUTO - CÓDIGO {base_code}
            
            Analise esta imagem e determine se as seguintes variantes representam o MESMO produto com cores diferentes:
            
            Variantes encontradas:
            {json.dumps(variant_info, indent=2)}
            
            Critérios para AGRUPAR:
            1. Mesmo código de produto base
            2. Mesmo nome/descrição de produto
            3. Diferem apenas na cor
            4. Tamanhos podem ser diferentes por cor
            
            Responda em JSON:
            {{
                "should_merge": true/false,
                "reasoning": "Explicação da decisão",
                "merged_product": {{
                    "product_name": "Nome do produto",
                    "material_code": "{base_code}",
                    "colors": [
                        {{
                            "color_code": "código",
                            "color_name": "nome",
                            "sizes": [...],
                            "unit_price": 0.0
                        }}
                    ]
                }}
            }}
            """
            
            # Usar primeira imagem como referência
            response = self.model.generate_content([analysis_prompt, images[0]])
            analysis = self._extract_json_safely(response.text)
            
            if analysis and analysis.get('should_merge'):
                logger.info(f"✅ Agrupando variantes de {base_code}: {analysis.get('reasoning', '')}")
                
                merged = analysis.get('merged_product', {})
                if merged:
                    # Copiar outros campos do primeiro produto
                    first_variant = variants[0]
                    merged.update({
                        'category': first_variant.get('category'),
                        'composition': first_variant.get('composition'),
                        'model': first_variant.get('model')
                    })
                    
                    return merged
            
            return None
            
        except Exception as e:
            logger.warning(f"Erro na análise de agrupamento para {base_code}: {e}")
            return None
    
    async def _correct_size_alignment_issues(self, 
                                           products: List[Dict],
                                           images: List[Image.Image]) -> List[Dict]:
        """
        Corrige problemas de alinhamento de tamanhos com quantidades
        """
        logger.info("📏 Corrigindo alinhamento de tamanhos...")
        
        corrected_products = []
        
        for product in products:
            corrected_product = await self._analyze_product_size_alignment(product, images)
            corrected_products.append(corrected_product)
        
        return corrected_products
    
    async def _analyze_product_size_alignment(self, 
                                            product: Dict,
                                            images: List[Image.Image]) -> Dict:
        """
        Analisa e corrige alinhamento de tamanhos para um produto específico
        """
        try:
            material_code = product.get('material_code', '')
            colors = product.get('colors', [])
            
            if not colors or not images:
                return product
            
            # Analisar cada cor do produto
            corrected_colors = []
            
            for color in colors:
                sizes = color.get('sizes', [])
                if not sizes:
                    corrected_colors.append(color)
                    continue
                
                # Análise visual para verificar alinhamento correto
                size_analysis_prompt = f"""
                CORREÇÃO DE ALINHAMENTO DE TAMANHOS - {material_code}
                
                Produto: {product.get('product_name', '')}
                Cor: {color.get('color_name', '')} ({color.get('color_code', '')})
                
                Tamanhos extraídos atualmente:
                {json.dumps(sizes, indent=2)}
                
                Analise esta imagem e verifique se os tamanhos estão corretos:
                
                1. Procure por este código de produto na imagem
                2. Identifique a linha/seção correspondente
                3. Verifique o alinhamento entre colunas de tamanhos e quantidades
                4. Identifique tamanhos que têm quantidade > 0
                
                PROBLEMAS COMUNS:
                - Tamanhos desalinhados (ex: M,L,XL extraído mas na imagem é S,M,L)
                - Tamanhos numéricos incorretos (ex: 25,26,27 extraído mas na imagem é 26,27,28)
                - Tamanhos de roupas vs. sapatos confundidos
                
                Responda em JSON:
                {{
                    "alignment_correct": true/false,
                    "corrected_sizes": [
                        {{"size": "S", "quantity": 1}},
                        {{"size": "M", "quantity": 2}}
                    ],
                    "correction_reasoning": "Explicação da correção feita"
                }}
                """
                
                # Analisar com primeira imagem que provavelmente contém o produto
                response = self.model.generate_content([size_analysis_prompt, images[0]])
                size_analysis = self._extract_json_safely(response.text)
                
                if size_analysis and not size_analysis.get('alignment_correct', True):
                    # Aplicar correção
                    corrected_sizes = size_analysis.get('corrected_sizes', sizes)
                    
                    logger.info(f"🔧 Corrigindo tamanhos para {material_code} cor {color.get('color_code')}")
                    logger.info(f"   Original: {[s.get('size') for s in sizes]}")
                    logger.info(f"   Corrigido: {[s.get('size') for s in corrected_sizes]}")
                    logger.info(f"   Razão: {size_analysis.get('correction_reasoning', '')}")
                    
                    # Atualizar cor com tamanhos corrigidos
                    corrected_color = color.copy()
                    corrected_color['sizes'] = corrected_sizes
                    corrected_colors.append(corrected_color)
                else:
                    corrected_colors.append(color)
            
            # Atualizar produto com cores corrigidas
            corrected_product = product.copy()
            corrected_product['colors'] = corrected_colors
            
            return corrected_product
            
        except Exception as e:
            logger.warning(f"Erro na correção de tamanhos para {product.get('material_code', '')}: {e}")
            return product
    
    async def _final_consistency_check(self, 
                                     products: List[Dict],
                                     images: List[Image.Image]) -> Tuple[List[Dict], List[str]]:
        """
        Verificação final de consistência
        """
        logger.info("✅ Verificação final de consistência...")
        
        final_corrections = []
        final_products = []
        
        for product in products:
            # Verificar se produto tem dados mínimos necessários
            if self._validate_product_completeness(product):
                final_products.append(product)
            else:
                # Tentar recuperar dados em falta
                recovered_product = await self._attempt_data_recovery(product, images)
                if recovered_product:
                    final_products.append(recovered_product)
                    final_corrections.append(f"Dados recuperados para {product.get('material_code', 'produto')}")
                else:
                    # Manter produto original mesmo com dados incompletos
                    final_products.append(product)
                    final_corrections.append(f"Produto {product.get('material_code', 'produto')} com dados incompletos mantido")
        
        return final_products, final_corrections
    
    def _validate_product_completeness(self, product: Dict) -> bool:
        """Valida se produto tem dados mínimos necessários"""
        
        # Verificar campos essenciais
        if not product.get('material_code') or not product.get('product_name'):
            return False
        
        # Verificar cores
        colors = product.get('colors', [])
        if not colors:
            return False
        
        # Verificar se pelo menos uma cor tem tamanhos com quantidade
        for color in colors:
            sizes = color.get('sizes', [])
            if sizes:
                # Verificar se há pelo menos um tamanho com quantidade > 0
                has_quantity = any(s.get('quantity', 0) > 0 for s in sizes)
                if has_quantity:
                    return True
        
        return False
    
    async def _attempt_data_recovery(self, 
                                   product: Dict,
                                   images: List[Image.Image]) -> Optional[Dict]:
        """
        Tenta recuperar dados em falta para um produto
        """
        try:
            if not images:
                return None
            
            material_code = product.get('material_code', '')
            
            recovery_prompt = f"""
            RECUPERAÇÃO DE DADOS - {material_code}
            
            Este produto tem dados incompletos:
            {json.dumps(product, indent=2)}
            
            Analise a imagem e tente recuperar informações em falta:
            
            1. Nome do produto (se em falta)
            2. Cores e seus códigos
            3. Tamanhos disponíveis com quantidades
            4. Preços
            
            Responda em JSON com APENAS os dados que conseguir ver claramente:
            {{
                "recovered_data": {{
                    "product_name": "nome se visível",
                    "colors": [
                        {{
                            "color_code": "código se visível",
                            "color_name": "nome se visível", 
                            "sizes": [...],
                            "unit_price": 0.0
                        }}
                    ]
                }},
                "recovery_confidence": <0.0-1.0>
            }}
            """
            
            response = self.model.generate_content([recovery_prompt, images[0]])
            recovery_result = self._extract_json_safely(response.text)
            
            if recovery_result and recovery_result.get('recovery_confidence', 0) > 0.6:
                # Aplicar dados recuperados
                recovered_product = product.copy()
                recovered_data = recovery_result.get('recovered_data', {})
                
                for key, value in recovered_data.items():
                    if value:  # Só aplicar se há valor
                        recovered_product[key] = value
                
                return recovered_product
            
            return None
            
        except Exception as e:
            logger.warning(f"Erro na recuperação de dados para {product.get('material_code', '')}: {e}")
            return None
    
    def _calculate_correction_confidence(self, 
                                       original_products: List[Dict],
                                       corrected_products: List[Dict],
                                       corrections: List[str]) -> float:
        """
        Calcula confiança das correções aplicadas
        """
        if not original_products:
            return 0.0
        
        factors = []
        
        # Fator 1: Proporção de produtos mantidos vs alterados
        if len(corrected_products) > 0:
            stability_factor = min(1.0, len(corrected_products) / len(original_products))
            factors.append(stability_factor * 0.3)
        
        # Fator 2: Qualidade das correções (baseado no número vs qualidade)
        correction_quality = 1.0 - min(0.5, len(corrections) / len(original_products))
        factors.append(correction_quality * 0.3)
        
        # Fator 3: Completude dos dados finais
        complete_products = sum(1 for p in corrected_products if self._validate_product_completeness(p))
        completeness_factor = complete_products / len(corrected_products) if corrected_products else 0
        factors.append(completeness_factor * 0.4)
        
        return sum(factors)
    
    def _extract_json_safely(self, text: str) -> Optional[Dict]:
        """Extração segura de JSON"""
        try:
            # Procurar JSON com markdown
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Procurar JSON sem markdown
            json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            return None
            
        except Exception as e:
            logger.warning(f"Erro ao extrair JSON: {e}")
            return None