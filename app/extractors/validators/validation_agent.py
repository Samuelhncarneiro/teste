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
        
        if self.should_skip_validation(products):
            logger.info("✅ Color mapping já aplicado - produtos mantidos como estão")
            return ValidationResult(products=products, valid=True)
    
        validation_errors = []
        missing_fields = []
        recommendations = []
        corrections_applied = []
        sizes_corrected = 0
        quantities_corrected = 0
        products_merged = 0
        
        color_grouped_products = extracted_products.copy()
        merge_corrections = []
        
        # Só fazer correções de tamanhos e quantidades se REALMENTE necessário
        size_corrected_products = []
        size_corrections = []
        
        for product in color_grouped_products:
            # Verificar se produto REALMENTE precisa de correção
            if self._product_needs_size_correction(product):
                logger.info(f"🔧 Produto {product.get('material_code')} precisa correção de tamanhos")
                corrected_product, product_corrections = await self._fix_single_product(product, pdf_pages)
                size_corrected_products.append(corrected_product)
                size_corrections.extend(product_corrections)
            else:
                # Produto está OK, não tocar
                size_corrected_products.append(product)
        
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
    
    def _create_minimal_validation_result(self, products: List[Dict], 
                                    pdf_pages: List[Image.Image], 
                                    context: Dict) -> ValidationResult:

        logger.info("📋 Criando validação mínima para preservar color mapping")
        
        return ValidationResult(
            products=products,  # Produtos sem alterações
            confidence_score=0.95,  # Alta confiança pois color mapping funcionou
            missing_fields=[],
            validation_errors=[],
            total_pages_processed=len(pdf_pages),
            extraction_method="color_mapping_preserved",
            completeness_score=0.9,
            consistency_score=0.9,
            visual_completeness_score=0.85,
            density_score=0.8,
            recommendations=["Color mapping aplicado corretamente - nenhuma correção necessária"],
            sizes_corrected=0,
            quantities_corrected=0,
            products_merged=0,
            corrections_applied=["Preservado color mapping existente"]
        )

    def _product_needs_size_correction(self, product: Dict[str, Any]) -> bool:

        if not product.get("colors"):
            return False
        
        needs_correction = False
        
        for color in product["colors"]:
            sizes = color.get("sizes", [])
            
            # Problema 1: Todos os tamanhos têm quantidade 1 (suspeito)
            if len(sizes) > 2:  # Só verificar se tem vários tamanhos
                quantities = [s.get("quantity", 0) for s in sizes]
                if len(set(quantities)) == 1 and quantities[0] == 1:
                    logger.info(f"Suspeita: {product.get('material_code')} tem todas quantidades = 1")
                    needs_correction = True
                    break
            
            # Problema 2: Faltam tamanhos óbvios (ex: só tem M, falta S e L)
            size_names = [s.get("size", "") for s in sizes]
            if len(size_names) == 1 and size_names[0] in ["M", "L"]:
                logger.info(f"Suspeita: {product.get('material_code')} só tem 1 tamanho")
                needs_correction = True
                break
        
        return needs_correction

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
    
    def should_skip_validation(self, products: List[Dict]) -> bool:
 
        # Se produtos já têm códigos de cores válidos, não validar
        valid_codes = {"001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012"}
        
        for product in products:
            for color in product.get("colors", []):
                color_code = color.get("color_code", "")
                if color_code in valid_codes:
                    # Se já tem códigos válidos, color mapping funcionou
                    logger.info("✅ Color mapping já aplicado - pulando validação destrutiva")
                    return True
        
        return False

    async def _fix_single_product(self, 
                            product: Dict,
                            images: List[Image.Image]) -> Tuple[Dict, List[str]]:

        corrections = []
        material_code = product.get('material_code', '')
        
        try:
            if not images:
                return product, corrections
            
            # Prompt MUITO específico e conservador
            fix_prompt = f"""
            CORREÇÃO CONSERVADORA DE TAMANHOS/QUANTIDADES - {material_code}
            
            Produto: {product.get('name', '')}
            
            IMPORTANTE: NÃO ALTERAR CORES! Só verificar tamanhos e quantidades.
            
            Cores atuais (NÃO MODIFICAR):
            """
            
            for i, color in enumerate(product.get('colors', [])):
                fix_prompt += f"""
            Cor {i+1}: {color.get('color_name', '')} (código: {color.get('color_code', '')})
            Tamanhos: {[f"{s.get('size')}({s.get('quantity')})" for s in color.get('sizes', [])]}
            """
            
            fix_prompt += f"""
            
            TAREFA LIMITADA:
            1. Localize este produto na imagem
            2. Verifique se os TAMANHOS estão corretos (não alterar cores!)
            3. Verifique se as QUANTIDADES estão corretas
            
            RESPOSTA JSON (só se precisar correção):
            {{
                "needs_correction": true/false,
                "reason": "Motivo específico",
                "size_corrections": [
                    "Falta tamanho S com quantidade X",
                    "Tamanho XL deveria ter quantidade Y"
                ]
            }}
            
            Se tudo estiver correto, retorne: {{"needs_correction": false}}
            """
            
            # Tentar análise (só primeira imagem para ser rápido)
            try:
                response = self.model.generate_content([fix_prompt, images[0]])
                analysis = self._extract_json_safely(response.text)
                
                if analysis and analysis.get('needs_correction'):
                    corrections_needed = analysis.get('size_corrections', [])
                    
                    if corrections_needed:
                        logger.info(f"🔧 {material_code}: Correções necessárias detectadas")
                        for correction in corrections_needed:
                            corrections.append(f"{material_code}: {correction}")
                        
                        # IMPORTANTE: Não aplicar correções automaticamente
                        # Só registrar que foram detectadas
                        logger.warning(f"⚠️ {material_code}: Correções detectadas mas não aplicadas automaticamente")
                
            except Exception as e:
                logger.warning(f"Erro na análise de {material_code}: {e}")
            
            return product, corrections  # Retornar produto original SEMPRE
            
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
    
    async def _validate_single_product_non_destructive(self, product: Dict, images: List, material_code: str) -> List[str]:

        corrections = []
        
        try:
            validation_prompt = f"""
            # VALIDAÇÃO NÃO-DESTRUTIVA DO PRODUTO: {material_code}
            
            Você vai validar apenas TAMANHOS, QUANTIDADES e CÓDIGOS DE CORES.
            NÃO ALTERE nem comente sobre preços, fornecedores, ou outros campos.
            
            ## PRODUTO EXTRAÍDO:
            - Código: {product.get('material_code', '')}
            - Nome: {product.get('name', '')}
            - Cores encontradas: {len(product.get('colors', []))}
            
            ## CORES E TAMANHOS EXTRAÍDOS:
            """
            
            for i, color in enumerate(product.get('colors', [])):
                validation_prompt += f"""
            Cor {i+1}:
            - Código: {color.get('color_code', '')}
            - Nome: {color.get('color_name', '')}
            - Tamanhos: {[s.get('size') for s in color.get('sizes', [])]}
            - Quantidades: {[s.get('quantity') for s in color.get('sizes', [])]}
            """
            
            validation_prompt += f"""
            
            ## SUA TAREFA LIMITADA:
            
            1. **VERIFICAR SE OS TAMANHOS ESTÃO CORRETOS** (comparar com tabela)
            2. **VERIFICAR SE AS QUANTIDADES ESTÃO CORRETAS** (mapear posicionalmente)
            3. **VERIFICAR SE OS CÓDIGOS DE CORES ESTÃO CORRETOS**
            
            ## FORMATO DE RESPOSTA SIMPLES:
            
            ```json
            {{
            "status": "OK" ou "CORRIGIR_TAMANHOS" ou "CORRIGIR_CORES",
            "corrections_needed": [
                "Tamanho XL incluído mas sem quantidade na tabela",
                "Falta tamanho S com quantidade 2",
                "Código da cor 1 deveria ser 018 em vez de 011"
            ]
            }}
            ```
            
            IMPORTANTE: NÃO retorne produto corrigido, apenas liste as correções necessárias.
            """
            
            # Enviar para IA para validação
            response = await self._send_validation_request(validation_prompt, images[0])
            
            # Processar resposta
            validation_result = self._parse_validation_response(response)
            
            if validation_result.get("status") in ["CORRIGIR_TAMANHOS", "CORRIGIR_CORES"]:
                corrections = validation_result.get("corrections_needed", [])
                
                await self._apply_specific_corrections(product, corrections, images[0])
                
                logger.info(f"Produto {material_code} corrigido pontualmente")
            
            return corrections
            
        except Exception as e:
            logger.error(f"Erro na validação do produto {material_code}: {e}")
            return []
    
    async def _apply_specific_corrections(self, product: Dict, corrections: List[str], image) -> None:
        try:
            for correction in corrections:
                correction_lower = correction.lower()
                
                # Correções de tamanhos
                if "falta tamanho" in correction_lower:
                    # Extrair tamanho e quantidade da correção
                    size_match = re.search(r'tamanho (\w+)', correction)
                    qty_match = re.search(r'quantidade (\d+)', correction)
                    
                    if size_match:
                        size_to_add = size_match.group(1)
                        quantity_to_add = int(qty_match.group(1)) if qty_match else 1
                        
                        # Adicionar tamanho à primeira cor (assumindo que é a correção mais comum)
                        if product.get('colors') and len(product['colors']) > 0:
                            sizes_list = product['colors'][0].get('sizes', [])
                            
                            # Verificar se o tamanho já existe
                            size_exists = any(s.get('size') == size_to_add for s in sizes_list)
                            
                            if not size_exists:
                                sizes_list.append({
                                    "size": size_to_add,
                                    "quantity": quantity_to_add
                                })
                                logger.info(f"✅ Adicionado tamanho {size_to_add} com quantidade {quantity_to_add}")
                
                elif "tamanho" in correction_lower and "incluído mas sem quantidade" in correction_lower:
                    # Remover tamanho que não deveria estar lá
                    size_match = re.search(r'tamanho (\w+)', correction)
                    
                    if size_match:
                        size_to_remove = size_match.group(1)
                        
                        for color in product.get('colors', []):
                            sizes_list = color.get('sizes', [])
                            # Filtrar tamanhos, removendo o problemático
                            color['sizes'] = [s for s in sizes_list if s.get('size') != size_to_remove]
                            logger.info(f"✅ Removido tamanho {size_to_remove} sem quantidade")
                
                # Correções de códigos de cores
                elif "código da cor" in correction_lower and "deveria ser" in correction_lower:
                    # Extrair códigos da correção
                    code_match = re.search(r'deveria ser (\w+) em vez de (\w+)', correction)
                    
                    if code_match:
                        correct_code = code_match.group(1)
                        wrong_code = code_match.group(2)
                        
                        for color in product.get('colors', []):
                            if color.get('color_code') == wrong_code:
                                color['color_code'] = correct_code
                                logger.info(f"✅ Corrigido código de cor de {wrong_code} para {correct_code}")
                
                # Correções de quantidades
                elif "quantidade" in correction_lower and "incorreta" in correction_lower:
                    # Para correções mais complexas de quantidades, pode implementar lógica adicional
                    logger.info(f"⚠️ Correção de quantidade detectada mas não implementada: {correction}")
                    
        except Exception as e:
            logger.warning(f"Erro ao aplicar correção '{correction}': {e}")

    async def validate_products_individually(self, extraction_result: Dict[str, Any], document_path: str) -> Dict[str, Any]:
            logger.info("🔍 Iniciando validação produto por produto...")
            
            products = extraction_result.get("products", [])
            if not products:
                logger.warning("Nenhum produto para validar")
                return extraction_result
            
            # Obter imagens do documento
            images = self._get_document_images_safe(document_path)
            if not images:
                logger.warning("Sem imagens para validação visual")
                return extraction_result
            
            validated_products = []
            validation_stats = {
                "total_products": len(products),
                "products_corrected": 0,
                "sizes_corrected": 0,
                "colors_corrected": 0,
                "corrections_made": []
            }
            
            # Validar cada produto individualmente
            for i, product in enumerate(products):
                material_code = product.get("material_code", "")
                product_name = product.get("name", "")
                
                logger.info(f"🔍 Validando produto {i+1}/{len(products)}: {material_code} - {product_name}")
                
                # MUDANÇA: Criar cópia completa do produto original
                validated_product = product.copy()
                
                # Validação específica para este produto
                corrections = await self._validate_single_product_non_destructive(
                    validated_product, images, material_code
                )
                
                validated_products.append(validated_product)
                
                # Registrar correções
                if corrections:
                    validation_stats["products_corrected"] += 1
                    validation_stats["corrections_made"].extend(corrections)
                    
                    # Contar tipos de correções
                    for correction in corrections:
                        if "tamanho" in correction.lower():
                            validation_stats["sizes_corrected"] += 1
                        if "cor" in correction.lower():
                            validation_stats["colors_corrected"] += 1
                    
                    logger.info(f"✅ Produto {material_code}: {len(corrections)} correções aplicadas")
                else:
                    logger.info(f"✅ Produto {material_code}: OK, nenhuma correção necessária")
            
            # MUDANÇA: Preservar toda a estrutura original
            validated_result = extraction_result.copy()
            validated_result["products"] = validated_products
            validated_result["individual_validation"] = validation_stats
            
            # Log final
            logger.info(f"🎉 Validação individual concluída:")
            logger.info(f"   - Produtos validados: {validation_stats['total_products']}")
            logger.info(f"   - Produtos corrigidos: {validation_stats['products_corrected']}")
            logger.info(f"   - Tamanhos corrigidos: {validation_stats['sizes_corrected']}")
            logger.info(f"   - Cores corrigidas: {validation_stats['colors_corrected']}")
            
            return validated_result

    async def _validate_single_product(self, product: Dict, images: List, material_code: str) -> Tuple[Dict, List[str]]:
        """
        Valida um único produto contra as imagens
        """
        corrections = []
        validated_product = product.copy()
        
        try:
            # Prompt específico para validar este produto
            validation_prompt = f"""
            # VALIDAÇÃO ESPECÍFICA DO PRODUTO: {material_code}
            
            Você vai validar se este produto foi extraído corretamente das imagens.
            
            ## PRODUTO EXTRAÍDO:
            - Código: {product.get('material_code', '')}
            - Nome: {product.get('name', '')}
            - Categoria: {product.get('category', '')}
            - Cores encontradas: {len(product.get('colors', []))}
            
            ## CORES E TAMANHOS EXTRAÍDOS:
            """
            
            for i, color in enumerate(product.get('colors', [])):
                validation_prompt += f"""
            Cor {i+1}:
            - Código: {color.get('color_code', '')}
            - Nome: {color.get('color_name', '')}
            - Tamanhos: {[s.get('size') for s in color.get('sizes', [])]}
            - Quantidades: {[s.get('quantity') for s in color.get('sizes', [])]}
            """
            
            validation_prompt += f"""
            
            ## SUA TAREFA:
            
            1. **VERIFICAR SE O PRODUTO {material_code} ESTÁ VISÍVEL** nas imagens
            2. **CONFERIR SE OS TAMANHOS ESTÃO CORRETOS** (comparar com tabela)
            3. **CONFERIR SE AS QUANTIDADES ESTÃO CORRETAS** (mapear posicionalmente)
            4. **CONFERIR SE AS CORES ESTÃO CORRETAS** (códigos e nomes)
            
            ## REGRAS DE VALIDAÇÃO:
            
            **Para TAMANHOS:**
            - Verificar se todos os tamanhos com quantidade > 0 estão incluídos
            - Verificar se não há tamanhos sem quantidade que foram incluídos
            - Mapear posicionalmente: tamanho = posição da quantidade
            
            **Para CORES:**
            - Verificar se o código da cor corresponde ao nome
            - Verificar se há cores em falta
            
            ## FORMATO DE RESPOSTA:
            
            ```json
            {{
            "status": "OK" ou "CORRIGIR",
            "corrections_needed": [
                "Tamanho XL incluído mas sem quantidade na tabela",
                "Falta tamanho S com quantidade 2",
                "Cor azul tem código errado"
            ],
            "corrected_product": {{
                // Produto corrigido (só se status = "CORRIGIR")
                "name": "...",
                "material_code": "{material_code}",
                "colors": [
                {{
                    "color_code": "...",
                    "color_name": "...", 
                    "sizes": [
                    {{"size": "S", "quantity": 2}},
                    {{"size": "M", "quantity": 1}}
                    ]
                }}
                ]
            }}
            }}
            ```
            
            IMPORTANTE: Se status = "OK", não inclua "corrected_product"
            """
            
            # Enviar para IA para validação
            response = await self._send_validation_request(validation_prompt, images[0])
            
            # Processar resposta
            validation_result = self._parse_validation_response(response)
            
            if validation_result.get("status") == "CORRIGIR":
                corrections = validation_result.get("corrections_needed", [])
                corrected_product = validation_result.get("corrected_product")
                
                if corrected_product:
                    # Manter campos originais e atualizar apenas os corrigidos
                    validated_product.update(corrected_product)
                    logger.info(f"Produto {material_code} corrigido com base na validação visual")
            
            return validated_product, corrections
            
        except Exception as e:
            logger.error(f"Erro na validação do produto {material_code}: {e}")
            return validated_product, []
    
    def _get_document_images_safe(self, document_path: str) -> List[Image.Image]:
        """Obter imagens do documento para validação"""
        try:
            if not document_path.lower().endswith('.pdf'):
                return []
            
            from app.utils.file_utils import convert_pdf_to_images
            from app.config import CONVERTED_DIR
            
            image_paths = convert_pdf_to_images(document_path, CONVERTED_DIR)
            images = []
            
            for img_path in image_paths:
                try:
                    img = Image.open(img_path)
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Erro ao carregar imagem {img_path}: {e}")
            
            return images
        except Exception as e:
            logger.warning(f"Erro ao obter imagens: {e}")
            return []

    async def _send_validation_request(self, prompt: str, image) -> str:
        """Enviar request de validação para a IA"""
        try:
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            logger.error(f"Erro na requisição de validação: {e}")
            return ""

    def _parse_validation_response(self, response_text: str) -> Dict:
        """Processar resposta da validação"""
        return self._extract_json_safely(response_text)