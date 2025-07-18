# app/extractors/gemini_extractor.py
import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
import re 
import math
import numpy as np
from PIL import Image 

from app.config import GEMINI_API_KEY, GEMINI_MODEL, CONVERTED_DIR
from app.extractors.base import BaseExtractor
from app.extractors.context_agent import ContextAgent
from app.extractors.extraction_agent import ExtractionAgent
from app.extractors.color_mapping_agent import ColorMappingAgent
from app.extractors.layout_detetion_agent import LayoutDetetionAgent
from app.extractors.generic_strategy_agent import GenericStrategyAgent
from app.extractors.validators.size_color_validation_agent import SizeColorValidationAgent, SizeColorValidationResult
from app.extractors.validators.validation_agent import ValidationAgent, ValidationResult

from app.utils.file_utils import convert_pdf_to_images
from app.utils.barcode_generator import add_barcodes_to_extraction_result, add_barcodes_to_products
from app.data.reference_data import (get_supplier_code, get_markup, get_category,SUPPLIER_MAP, COLOR_MAP, SIZE_MAP,CATEGORIES)
from app.utils.json_utils import safe_json_dump, fix_nan_in_products, sanitize_for_json
from app.utils.supplier_assignment import determine_best_supplier, assign_supplier_to_products
from app.data.reference_data import determine_gender_by_brand

logger = logging.getLogger(__name__)

try:
    HAS_VALIDATION = True
    logger.info("✅ Sistema de validação melhorado carregado")
except ImportError:
    HAS_VALIDATION = False
    logger.warning("⚠️ Sistema de validação não disponível")

try:
    HAS_SIZE_VALIDATION = True
    logger.info("✅ Sistema de validação de tamanhos carregado")
except ImportError:
    HAS_SIZE_VALIDATION = False
    logger.warning("⚠️ Sistema de validação de tamanhos não disponível")

try:
    from app.utils.json_utils import safe_json_dump, fix_nan_in_products, sanitize_for_json
    has_json_utils = True
except ImportError:
    has_json_utils = False
    logger.warning("Módulo json_utils não encontrado, usar serialização padrão")

class GeminiExtractor(BaseExtractor):
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        self.context_agent = ContextAgent(api_key)
        self.extraction_agent = ExtractionAgent(api_key)
        self.ai_color_mapping_agent = ColorMappingAgent()
        
        self.layout_detector = LayoutDetetionAgent(api_key)
        self.strategy_agent = GenericStrategyAgent()

        self.validation_agent = ValidationAgent(api_key)
        
        if HAS_SIZE_VALIDATION:
            try:
                self.size_validator = SizeColorValidationAgent(api_key)
                logger.info("✅ Validador de tamanhos e cores inicializado")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao inicializar validador de tamanhos: {e}")
                self.size_validator = None
            else:
                self.size_validator = None

        self.current_layout_analysis = {}
        self.current_strategy = None
        self.page_results_history = []

    async def analyze_context(self, document_path: str) -> str:
        logger.info("🔧 Usando análise clássica")
        
        context_info = await self.context_agent.analyze_document(document_path)
        
        logger.info("Detectando layout do documento...")
        layout_analysis = await self.layout_detector.analyze_document_structure(document_path)
        
        layout_type = layout_analysis.get('layout_type', 'UNKNOWN')
        confidence = layout_analysis.get('confidence', 0.0)
        logger.info(f"Layout detectado: {layout_type} (confiança: {confidence:.2f})")
        
        logger.info("Selecionando estratégia de extração...")
        strategy = self.strategy_agent.select_strategy(
            layout_analysis=layout_analysis,
            page_number=1
        )
        logger.info(f"Estratégia selecionada: {strategy.name}")
        
        self.current_context_info = context_info
        self.current_layout_analysis = layout_analysis
        self.current_strategy = strategy
        
        enhanced_context = self._enhance_context_with_layout_and_strategy(
            context_info, layout_analysis, strategy
        )
        
        return enhanced_context

    async def _retry_extraction_with_different_strategy(self, 
                                                   document_path: str,
                                                   recommendations: List[str]) -> List[Dict]:
        try:
            logger.info("🔄 Tentando re-extração com estratégia alternativa...")
            
            # Analisar recomendações para escolher estratégia
            retry_focus = []
            
            if any("tamanho" in rec.lower() for rec in recommendations):
                retry_focus.append("sizes")
            if any("quantidade" in rec.lower() for rec in recommendations):
                retry_focus.append("quantities")
            if any("agrup" in rec.lower() for rec in recommendations):
                retry_focus.append("grouping")
            
            # Executar re-extração focada
            if "sizes" in retry_focus or "quantities" in retry_focus:
                return await self._focused_size_quantity_extraction(document_path)
            elif "grouping" in retry_focus:
                return await self._focused_grouping_extraction(document_path)
            else:
                return await self._generic_retry_extraction(document_path)
                
        except Exception as e:
            logger.warning(f"Erro na re-extração: {e}")
            return []

    async def _focused_size_quantity_extraction(self, document_path: str) -> List[Dict]:
        """
        Re-extração focada em tamanhos e quantidades corretos
        """
        try:
            images = self._get_document_images_safe(document_path)
            if not images:
                return []
            
            context = self.current_context_info or {}
            
            # Prompt especializado em tamanhos e quantidades
            focused_prompt = f"""
            RE-EXTRAÇÃO FOCADA EM TAMANHOS E QUANTIDADES
            
            PROBLEMA DETECTADO: Tamanhos incompletos ou quantidades incorretas
            
            INSTRUÇÕES ESPECÍFICAS:
            1. Para cada produto, leia TODOS os tamanhos da linha (XS, S, M, L, XL, XXL)
            2. Leia as quantidades EXATAS de cada tamanho (incluindo 0)
            3. NÃO assuma quantidade 1 para todos
            4. Verifique alinhamento posicional entre colunas
            
            FORMATO DE RESPOSTA:
            {{
                "products": [
                    {{
                        "product_name": "...",
                        "material_code": "...",
                        "colors": [
                            {{
                                "color_code": "...",
                                "color_name": "...",
                                "unit_price": 0.0,
                                "sizes": [
                                    {{"size": "XS", "quantity": 0}},
                                    {{"size": "S", "quantity": 1}},
                                    {{"size": "M", "quantity": 2}},
                                    {{"size": "L", "quantity": 1}},
                                    {{"size": "XL", "quantity": 0}}
                                ]
                            }}
                        ]
                    }}
                ]
            }}
            
            CRÍTICO: Incluir TODOS os tamanhos visíveis, mesmo com quantidade 0
            """
            
            result = await self.extraction_agent.extract_from_page(
                images[0], focused_prompt, 1, len(images), []
            )
            
            return result.get('products', [])
            
        except Exception as e:
            logger.warning(f"Erro na re-extração focada: {e}")
            return []

    async def _focused_grouping_extraction(self, document_path: str) -> List[Dict]:
        """
        Re-extração focada em agrupamento correto de produtos
        """
        try:
            images = self._get_document_images_safe(document_path)
            if not images:
                return []
            
            grouping_prompt = f"""
            RE-EXTRAÇÃO FOCADA EM AGRUPAMENTO DE PRODUTOS
            
            PROBLEMA DETECTADO: Produtos duplicados por cor
            
            INSTRUÇÕES:
            1. Identifique produtos com códigos similares (ex: CF5271MA96E.1, CF5271MA96E.2)
            2. Agrupe-os em UM produto com múltiplas cores
            3. NÃO crie produtos separados para cada cor
            
            EXEMPLO CORRETO:
            {{
                "products": [
                    {{
                        "material_code": "CF5271MA96E",
                        "product_name": "Malha Gola Redonda",
                        "colors": [
                            {{
                                "color_code": "M9799",
                                "color_name": "Nero",
                                "sizes": [...tamanhos para esta cor...]
                            }},
                            {{
                                "color_code": "012",
                                "color_name": "Bege", 
                                "sizes": [...tamanhos para esta cor...]
                            }}
                        ]
                    }}
                ]
            }}
            """
            
            result = await self.extraction_agent.extract_from_page(
                images[0], grouping_prompt, 1, len(images), []
            )
            
            return result.get('products', [])
            
        except Exception as e:
            logger.warning(f"Erro na re-extração de agrupamento: {e}")
            return []

    async def _generic_retry_extraction(self, document_path: str) -> List[Dict]:
        """
        Re-extração genérica com prompt mais rigoroso
        """
        try:
            images = self._get_document_images_safe(document_path)
            if not images:
                return []
            
            context = self.current_context_info.copy() if self.current_context_info else {}
            context.update({
                'extraction_mode': 'retry_conservative',
                'focus': 'accuracy_over_speed',
                'validation': 'strict'
            })
            
            result = await self.extraction_agent.extract_from_page(
                images[0], context, 1, len(images), []
            )
            
            return result.get('products', [])
            
        except Exception as e:
            logger.warning(f"Erro no retry genérico: {e}")
            return []
    
    async def extract_with_validation(self, document_path: str, 
                                enable_validation: bool = True,
                                max_retries: int = 1) -> Dict[str, Any]:

        logger.info(f"🔍 Iniciando extração com validação MELHORADA para: {os.path.basename(document_path)}")
        
        try:
            # 1. Extração inicial (método existente)
            logger.info("🚀 Executando extração inicial...")
            extraction_result = await self.extract(document_path)
            initial_products = extraction_result.get('products', [])
            
            logger.info(f"📊 Extração inicial: {len(initial_products)} produtos")
            
            # 2. Se validação desabilitada ou não disponível
            if not enable_validation or not HAS_VALIDATION or not self.validation_agent:
                logger.info("📄 Executando sem validação")
                extraction_result['validation'] = {
                    'enabled': False,
                    'message': 'Validação desabilitada ou não disponível'
                }
                return extraction_result
            
            # 3. Detectar páginas que falharam (método existente)
            failed_pages = []
            if hasattr(self, 'page_results_history'):
                for i, page_result in enumerate(self.page_results_history):
                    if page_result.get('error') or page_result.get('products_count', 0) == 0:
                        failed_pages.append(i + 1)
            
            if failed_pages:
                logger.info(f"🔍 Páginas com falha detectadas: {failed_pages}")
            
            # 4. Obter imagens para validação
            images = self._get_document_images_safe(document_path)
            if not images:
                logger.warning("⚠️ Sem imagens para validação visual")
                extraction_result['validation'] = {
                    'enabled': False,
                    'message': 'Sem imagens para análise visual'
                }
                return extraction_result
            
            logger.info(f"🖼️ Carregadas {len(images)} imagens para validação")
            
            # 5. Executar validação MELHORADA
            logger.info("🔍 Executando validação com correções específicas...")
            
            context = {
                'document_type': extraction_result.get('document_type', ''),
                'supplier': extraction_result.get('supplier', ''),
                'brand': extraction_result.get('brand', ''),
                'file_name': os.path.basename(document_path)
            }
            
            validation_result = await self.validation_agent.validate_extraction(
                extracted_products=initial_products,
                original_context=context,
                pdf_pages=images,
                layout_analysis=self.current_layout_analysis
            )
            
            # 6. Log detalhado das correções
            logger.info(f"📈 Validação concluída:")
            logger.info(f"   - Produtos iniciais: {len(initial_products)}")
            logger.info(f"   - Produtos finais: {len(validation_result.products)}")
            logger.info(f"   - Produtos agrupados: {validation_result.products_merged}")
            logger.info(f"   - Tamanhos corrigidos: {validation_result.sizes_corrected}")
            logger.info(f"   - Quantidades corrigidas: {validation_result.quantities_corrected}")
            logger.info(f"   - Confiança: {validation_result.confidence_score:.2f}")
            
            # 7. Log das correções aplicadas
            if validation_result.corrections_applied:
                logger.info("🔧 Correções aplicadas:")
                for correction in validation_result.corrections_applied[:5]:
                    logger.info(f"   - {correction}")
                if len(validation_result.corrections_applied) > 5:
                    logger.info(f"   - ... e mais {len(validation_result.corrections_applied) - 5}")
            
            # 8. Log das recomendações
            if validation_result.recommendations:
                logger.info("💡 Recomendações:")
                for rec in validation_result.recommendations:
                    logger.info(f"   - {rec}")
            
            # 9. Retry se confiança muito baixa
            retry_count = 0
            while (validation_result.confidence_score < 0.5 and 
                retry_count < max_retries):
                
                retry_count += 1
                logger.warning(f"Confiança baixa ({validation_result.confidence_score:.2f}) - Retry {retry_count}")
                
                # Re-extração com estratégia diferente (método existente)
                alternative_products = await self._retry_extraction_with_different_strategy(
                    document_path, validation_result.recommendations
                )
                
                if alternative_products:
                    validation_result = await self.validation_agent.validate_extraction(
                        extracted_products=alternative_products,
                        original_context=context,
                        pdf_pages=images,
                        layout_analysis=self.current_layout_analysis
                    )
            
            # 10. Preparar resultado final melhorado
            enhanced_result = extraction_result.copy()
            enhanced_result.update({
                'products': validation_result.products,
                'validation': {
                    'enabled': True,
                    'confidence_score': validation_result.confidence_score,
                    'completeness_score': validation_result.completeness_score,
                    'consistency_score': validation_result.consistency_score,
                    'visual_completeness_score': validation_result.visual_completeness_score,
                    'density_score': validation_result.density_score,
                    'extraction_method': validation_result.extraction_method,
                    'validation_errors': validation_result.validation_errors,
                    'missing_fields': validation_result.missing_fields,
                    'recommendations': validation_result.recommendations,
                    'total_pages_processed': validation_result.total_pages_processed,
                    'products_initial': len(initial_products),
                    'products_final': len(validation_result.products),
                    'products_merged': validation_result.products_merged,
                    'sizes_corrected': validation_result.sizes_corrected,
                    'quantities_corrected': validation_result.quantities_corrected,
                    'corrections_applied': validation_result.corrections_applied,
                    'failed_pages_detected': failed_pages
                }
            })
            
            # 11. Status final
            if validation_result.confidence_score >= 0.8:
                logger.info(f"✅ EXCELENTE: Alta confiança ({validation_result.confidence_score:.2f})")
            elif validation_result.confidence_score >= 0.6:
                logger.info(f"⚠️ BOM: Confiança média ({validation_result.confidence_score:.2f})")
            else:
                logger.warning(f"❌ ATENÇÃO: Baixa confiança ({validation_result.confidence_score:.2f})")
            
            return enhanced_result
            
        except Exception as e:
            logger.exception(f"❌ Erro na validação: {str(e)}")
            
            # Fallback para resultado original
            extraction_result['validation'] = {
                'enabled': False,
                'error': str(e),
                'message': 'Validação falhou - resultado da extração normal'
            }
            return extraction_result
    
    def _analyze_improvements(self, 
                        initial_products: List[Dict], 
                        corrected_products: List[Dict]) -> List[str]:
        """
        Analisa e reporta melhorias específicas feitas
        """
        improvements = []
        
        # Detectar produtos agrupados
        if len(corrected_products) < len(initial_products):
            merged_count = len(initial_products) - len(corrected_products)
            improvements.append(f"Agrupados {merged_count} produtos duplicados por cor")
        
        # Detectar correções de tamanhos
        initial_sizes = set()
        corrected_sizes = set()
        
        for product in initial_products:
            for color in product.get('colors', []):
                for size in color.get('sizes', []):
                    initial_sizes.add(size.get('size', ''))
        
        for product in corrected_products:
            for color in product.get('colors', []):
                for size in color.get('sizes', []):
                    corrected_sizes.add(size.get('size', ''))
        
        # Verificar se houve mudanças significativas nos tamanhos
        size_changes = corrected_sizes - initial_sizes
        if size_changes:
            improvements.append(f"Corrigidos tamanhos incorretos: {', '.join(list(size_changes)[:3])}")
        
        # Detectar melhorias na completude
        initial_complete = sum(1 for p in initial_products if self._is_product_complete(p))
        corrected_complete = sum(1 for p in corrected_products if self._is_product_complete(p))
        
        if corrected_complete > initial_complete:
            improvement = corrected_complete - initial_complete
            improvements.append(f"Recuperados dados completos para {improvement} produtos")
        
        return improvements

    def _is_product_complete(self, product: Dict) -> bool:
        """Verifica se produto tem dados completos"""
        if not product.get('material_code') or not product.get('product_name'):
            return False
        
        colors = product.get('colors', [])
        if not colors:
            return False
        
        for color in colors:
            sizes = color.get('sizes', [])
            if sizes and any(s.get('quantity', 0) > 0 for s in sizes):
                return True
        
        return False

    def _get_document_images_safe(self, document_path: str) -> List[Image.Image]:
        """Método melhorado para obter imagens"""
        try:
            if not document_path.lower().endswith('.pdf'):
                logger.info("Documento não é PDF - validação visual limitada")
                return []
            
            # Usar método existente de conversão
            image_paths = convert_pdf_to_images(document_path, CONVERTED_DIR)
            
            if not image_paths:
                logger.warning("Não foi possível converter PDF para imagens")
                return []
            
            images = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path)
                    images.append(img)
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao carregar imagem {img_path}: {e}")
            
            logger.info(f"🖼️ Carregadas {len(images)} imagens para validação")
            return images
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao obter imagens do documento: {e}")
            return []
        
    async def extract_with_size_validation(self, document_path: str, confidence_threshold: float = 0.7) -> Dict[str, Any]:
     
        logger.info(f"📏 Iniciando extração com validação de tamanhos para: {os.path.basename(document_path)}")
        
        try:
            # 1. Executar extração normal primeiro
            logger.info("🚀 Executando extração inicial...")
            extraction_result = await self.extract(document_path)
            initial_products = extraction_result.get('products', [])
            
            logger.info(f"📊 Extração inicial: {len(initial_products)} produtos")
            
            # 2. Se não há validação disponível, retornar resultado normal
            if not HAS_SIZE_VALIDATION or not self.size_validator:
                logger.info("📄 Validação de tamanhos não disponível")
                extraction_result['size_validation'] = {
                    'enabled': False,
                    'message': 'Sistema não disponível'
                }
                return extraction_result
            
            # 3. Obter imagens para análise visual
            images = self._get_document_images_safe(document_path)
            if not images:
                logger.warning("⚠️ Sem imagens para validação visual de tamanhos")
                extraction_result['size_validation'] = {
                    'enabled': False,
                    'message': 'Sem imagens para análise'
                }
                return extraction_result
            
            logger.info(f"🖼️ Carregadas {len(images)} imagens para validação")
            
            # 4. Executar validação de tamanhos e cores
            logger.info("🔍 Executando validação de tamanhos e cores...")
            
            validation_result = await self.size_validator.validate_and_correct(
                products=initial_products,
                page_images=images,
                confidence_threshold=confidence_threshold
            )
            
            # 5. Analisar resultados da validação
            corrected_products = validation_result.corrected_products
            corrections_made = validation_result.corrections_made
            
            logger.info(f"📈 Validação de tamanhos concluída:")
            logger.info(f"   - Produtos iniciais: {len(initial_products)}")
            logger.info(f"   - Produtos finais: {len(corrected_products)}")
            logger.info(f"   - Correções aplicadas: {len(corrections_made)}")
            logger.info(f"   - Confiança: {validation_result.confidence_score:.2f}")
            
            # 6. Log das correções aplicadas
            if corrections_made:
                logger.info("🔧 Correções aplicadas:")
                for correction in corrections_made[:5]:  # Primeiras 5
                    logger.info(f"   - {correction}")
                if len(corrections_made) > 5:
                    logger.info(f"   - ... e mais {len(corrections_made) - 5} correções")
            
            # 7. Detectar melhorias específicas
            improvements = self._analyze_improvements(initial_products, corrected_products)
            if improvements:
                logger.info("✨ Melhorias detectadas:")
                for improvement in improvements:
                    logger.info(f"   - {improvement}")
            
            # 8. Preparar resultado final
            enhanced_result = extraction_result.copy()
            enhanced_result.update({
                'products': corrected_products,
                'size_validation': {
                    'enabled': True,
                    'confidence_score': validation_result.confidence_score,
                    'corrections_made': corrections_made,
                    'validation_errors': validation_result.validation_errors,
                    'size_alignment_issues': validation_result.size_alignment_issues,
                    'color_grouping_issues': validation_result.color_grouping_issues,
                    'products_initial': len(initial_products),
                    'products_final': len(corrected_products),
                    'products_merged': len(initial_products) - len(corrected_products),
                    'improvements_detected': improvements,
                    'confidence_threshold_used': confidence_threshold
                }
            })
            
            # 9. Status final
            if validation_result.confidence_score >= 0.8:
                logger.info(f"✅ Alta confiança nas correções ({validation_result.confidence_score:.2f})")
            elif validation_result.confidence_score >= 0.6:
                logger.info(f"⚠️ Confiança média ({validation_result.confidence_score:.2f}) - verificar resultados")
            else:
                logger.warning(f"❌ Baixa confiança ({validation_result.confidence_score:.2f}) - possíveis problemas")
            
            return enhanced_result
            
        except Exception as e:
            logger.exception(f"❌ Erro na validação de tamanhos: {str(e)}")
            
            # Fallback para resultado original
            extraction_result['size_validation'] = {
                'enabled': False,
                'error': str(e),
                'message': 'Validação falhou - resultado da extração normal'
            }
            return extraction_result

    async def _alternative_structure_extraction(self, document_path: str) -> List[Dict]:
        """Extração com abordagem estrutural alternativa"""
        try:
            # Forçar re-análise de layout com estratégia diferente
            alternative_layout = await self.layout_detector.analyze_document_structure(document_path)
            
            # Selecionar estratégia diferente
            alternative_strategy = self.strategy_agent.select_strategy(
                layout_analysis=alternative_layout,
                page_number=1,
                previous_results=self.page_results_history
            )
            
            if alternative_strategy.name != self.current_strategy.name:
                logger.info(f"Usando estratégia alternativa: {alternative_strategy.name}")
                
                # Re-extrair com nova estratégia
                images = self._get_document_images(document_path)
                if images:
                    enhanced_context = self._enhance_context_with_layout_and_strategy(
                        self.current_context_info, alternative_layout, alternative_strategy
                    )
                    
                    result = await self.extraction_agent.extract_from_page(
                        images[0], enhanced_context, 1, len(images), []
                    )
                    
                    return result.get('products', [])
            
            return []
            
        except Exception as e:
            logger.warning(f"Erro na extração estrutural alternativa: {e}")
            return []

    async def _generic_retry_extraction(self, document_path: str) -> List[Dict]:
        """Retry genérico com parâmetros ajustados"""
        try:
            # Extrair novamente mas com prompt modificado para ser mais rigoroso
            images = self._get_document_images(document_path)
            if not images:
                return []
            
            # Criar contexto mais específico para retry
            retry_context = self.current_context_info.copy() if self.current_context_info else {}
            retry_context.update({
                'extraction_mode': 'retry',
                'focus': 'complete_products_only',
                'strictness': 'high'
            })
            
            result = await self.extraction_agent.extract_from_page(
                images[0], retry_context, 1, len(images), []
            )
            
            return result.get('products', [])
            
        except Exception as e:
            logger.warning(f"Erro no retry genérico: {e}")
            return []
        
    def _enhance_context_with_layout_and_strategy(
        self, 
        context_info: Dict[str, Any], 
        layout_analysis: Dict[str, Any], 
        strategy: Any
    ) -> str:

        original_context = self.context_agent.format_context_for_extraction(context_info)
        
        layout_info = [
            "\n## LAYOUT DETECTADO AUTOMATICAMENTE",
            f"**Tipo**: {layout_analysis.get('layout_type', 'UNKNOWN')}",
            f"**Confiança**: {layout_analysis.get('confidence', 0):.2f}",
            f"**Estratégia**: {layout_analysis.get('extraction_strategy', 'adaptive')}"
        ]
        
        technical = layout_analysis.get("technical_analysis", {})
        if not technical.get("error"):
            column_info = technical.get("column_detection", {})
            if column_info.get("column_count", 0) > 0:
                layout_info.append(f"**Colunas detectadas**: {column_info['column_count']}")
        
        strategy_info = [
            f"\n## ESTRATÉGIA SELECIONADA: {strategy.name.upper()}",
            f"**Abordagem**: {strategy.approach}"
        ]
        
        # Instruções específicas da estratégia
        strategy_info.append("\n### INSTRUÇÕES ESPECÍFICAS:")
        for key, instruction in strategy.specific_instructions.items():
            readable_key = key.replace('_', ' ').title()
            strategy_info.append(f"- **{readable_key}**: {instruction}")
        
        # Instruções do layout detectado
        extraction_instructions = layout_analysis.get("extraction_instructions", {})
        if extraction_instructions:
            strategy_info.append("\n### INSTRUÇÕES BASEADAS NO LAYOUT:")
            for key, instruction in extraction_instructions.items():
                if instruction and key != "special_considerations":
                    readable_key = key.replace('_', ' ').title()
                    strategy_info.append(f"- **{readable_key}**: {instruction}")
        
        # Combinar tudo
        return "\n".join([
            original_context,
            "\n".join(layout_info),
            "\n".join(strategy_info),
            "\n## REGRAS CRÍTICAS:",
            "- Seguir rigorosamente a estratégia selecionada",
            "- Adaptar se a estrutura mudar entre páginas",
            "- Extrair apenas dados claramente visíveis"
        ])

    async def process_page(
        self, 
        image_path: str, 
        context: str,
        page_number: int,
        total_pages: int,
        previous_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        logger.info(f"Processando página {page_number}/{total_pages}")
        
        if page_number > 1 and self.page_results_history:
            last_result = self.page_results_history[-1]
            new_strategy = self.strategy_agent.adapt_strategy_for_page(
                self.current_strategy, 
                last_result, 
                page_number - 1,
                self.current_layout_analysis
            )
            
            if new_strategy:
                logger.info(f"ADAPTAÇÃO: {self.current_strategy.name} → {new_strategy.name}")
                self.current_strategy = new_strategy
                
                # Atualizar contexto com nova estratégia
                context = self._update_context_with_new_strategy(context, new_strategy, page_number)
        
        # Processar página (usa o ExtractionAgent original)
        page_result = await self.extraction_agent.process_page(
            image_path, context, page_number, total_pages, previous_result
        )
        
        # NOVA: Armazenar resultado para adaptação
        page_result["_strategy_used"] = self.current_strategy.name if self.current_strategy else "unknown"
        self.page_results_history.append(page_result)
        
        # Log dos resultados
        products_found = len(page_result.get("products", []))
        has_error = "error" in page_result
        strategy_name = self.current_strategy.name if self.current_strategy else "unknown"
        
        logger.info(f"Página {page_number}: {products_found} produtos extraídos "
                   f"(estratégia: {strategy_name}){' COM ERRO' if has_error else ''}")
        
        return page_result
    
    def _update_context_with_new_strategy(
        self, 
        context: str, 
        new_strategy: Any, 
        page_number: int
    ) -> str:

        strategy_update = f"""
            ## ESTRATÉGIA ADAPTADA PARA PÁGINA {page_number}

            ⚠️ **MUDANÇA DE ESTRATÉGIA**
            - Nova estratégia: {new_strategy.name}
            - Motivo: Resultado anterior insatisfatório
            - Aplicar novas instruções:

            ### INSTRUÇÕES ATUALIZADAS:
            """
                    
        for key, instruction in new_strategy.specific_instructions.items():
            readable_key = key.replace('_', ' ').title()
            strategy_update += f"- **{readable_key}**: {instruction}\n"
        
        return context + strategy_update

    async def extract_document(
        self, 
        document_path: str,
        job_id: str,
        jobs_store: Dict[str, Any],
        update_progress_callback: Callable
    ) -> Dict[str, Any]:

        start_time = time.time()
        
        try:
            logger.info(f"🚀 INICIANDO EXTRAÇÃO - Job: {job_id}")  # ADICIONAR
            
            jobs_store[job_id]["model_results"]["gemini"] = {
                "model_name": GEMINI_MODEL,
                "status": "processing",
                "progress": 5.0
            }
            
            # ETAPA 1: Análise melhorada (contexto + layout + estratégia)
            is_pdf = document_path.lower().endswith('.pdf')
            
            if is_pdf:
                logger.info(f"📄 Processando PDF: {document_path}")  # ADICIONAR
                
                jobs_store[job_id]["model_results"]["gemini"]["progress"] = 10.0
                logger.info(f"=== ANÁLISE GENÉRICA INICIADA ===")
                logger.info(f"Documento: {os.path.basename(document_path)}")
                
                # NOVA: Análise completa (contexto + layout + estratégia)
                logger.info("🔍 Iniciando análise de contexto...")  # ADICIONAR
                context_description = await self.analyze_context(document_path)
                logger.info("✅ Análise de contexto concluída")  # ADICIONAR
                
                context_info = self.current_context_info
                
                logger.info(f"Análise completa concluída:")
                logger.info(f"- Layout: {self.current_layout_analysis.get('layout_type', 'UNKNOWN')}")
                logger.info(f"- Estratégia: {self.current_strategy.name if self.current_strategy else 'N/A'}")
            else:
                logger.info("📄 Documento não é PDF, usando configuração básica")  # ADICIONAR
                context_description = "Documento de pedido ou nota de encomenda"
                context_info = {"document_type": "Documento de pedido", "supplier": "", "brand": ""}
            
            # ETAPA 2-4: Mantém-se igual (preparar imagens + extração + cores)
            logger.info("🖼️ Preparando imagens...")  # ADICIONAR
            jobs_store[job_id]["model_results"]["gemini"]["progress"] = 15.0
            
            if is_pdf:
                logger.info("📸 Convertendo PDF para imagens...")  # ADICIONAR
                image_paths = convert_pdf_to_images(document_path, CONVERTED_DIR)
                logger.info(f"✅ {len(image_paths)} imagens criadas")  # ADICIONAR
            else:
                image_paths = [document_path]
                
            total_pages = len(image_paths)
            logger.info(f"📋 Preparadas {total_pages} imagens para processamento")
            
            # Extração com adaptação automática
            logger.info("🤖 Iniciando extração com IA...")  # ADICIONAR
            combined_result = {"products": [], "order_info": {}}
            
            if context_info:
                combined_result["order_info"] = {
                    "supplier": context_info.get("supplier", ""),
                    "document_type": context_info.get("document_type", ""),
                    "order_number": context_info.get("reference_number", ""),
                    "date": context_info.get("date", ""),
                    "customer": context_info.get("customer", ""),
                    "brand": context_info.get("brand", ""),
                    "season": context_info.get("season", "")
                }
            
            progress_per_page = 80.0 / total_pages
            
            for page_num, img_path in enumerate(image_paths, start=1):
                logger.info(f"📄 Processando página {page_num}/{total_pages}: {os.path.basename(img_path)}")  # ADICIONAR
                
                current_progress = 15.0 + (page_num - 1) * progress_per_page
                jobs_store[job_id]["model_results"]["gemini"]["progress"] = current_progress
                
                # NOVA: Processamento com adaptação automática
                logger.info(f"🔍 Enviando página {page_num} para análise IA...")  # ADICIONAR
                page_result = await self.process_page(
                    img_path,
                    context_description,
                    page_num,
                    total_pages,
                    combined_result if page_num > 1 else None
                )
                logger.info(f"✅ Página {page_num} processada")  # ADICIONAR
                
                # Verificar erro (mantém-se igual)
                if "error" in page_result and not page_result.get("products"):
                    logger.error(f"❌ Erro ao processar página {page_num}: {page_result['error']}")
                    if page_num == 1:
                        raise ValueError(f"Falha ao processar a primeira página: {page_result['error']}")
                    continue
                
                # Mesclar resultados (mantém-se igual)
                if "products" in page_result:
                    products_found = len(page_result.get("products", []))
                    logger.info(f"📦 Página {page_num}: {products_found} produtos encontrados")  # ADICIONAR
                    combined_result["products"].extend(page_result.get("products", []))
                
                if "order_info" in page_result and page_result["order_info"]:
                    for key, value in page_result["order_info"].items():
                        if value and (key not in combined_result["order_info"] or not combined_result["order_info"].get(key)):
                            combined_result["order_info"][key] = value
                
                jobs_store[job_id]["model_results"]["gemini"]["progress"] = 15.0 + page_num * progress_per_page
            
            total_products = len(combined_result["products"])
            logger.info(f"🎉 EXTRAÇÃO CONCLUÍDA - Total de produtos: {total_products}")
            
            if combined_result["products"]:
                try:
                    mapped_products = self.ai_color_mapping_agent.map_product_colors(
                        combined_result["products"]
                    )
                    combined_result["products"] = mapped_products
                    
                    mapping_report = self.ai_color_mapping_agent.get_mapping_report()
                    combined_result["_ai_color_mapping"] = mapping_report
                    
                    stats = mapping_report['statistics']
                    if stats['mappings_details']:
                        for change in stats['mappings_details'][:3]:
                            confidence = change.get('confidence', 'unknown')
                            logger.info(f"  '{change['original_name']}' ({change['original_code']}) → '{change['mapped_name']}' ({change['mapped_code']}) [confidence: {confidence}]")
                
                except Exception as e:
                    logger.error(f"Erro no mapeamento AI de cores: {str(e)}")
                    combined_result["_ai_color_mapping"] = {"error": str(e)}
            
            # Pós-processamento (mantém-se igual)
            logger.debug(f"🔍 Antes do pós-processamento: {len(combined_result['products'])} produtos")
            
            try:
                result_tuple = self._post_process_products(combined_result["products"], context_info)
                
                # Verificar se retornou tupla corretamente
                if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                    processed_products, determined_supplier = result_tuple
                    logger.debug(f"✅ Pós-processamento retornou: {len(processed_products) if processed_products else 0} produtos, fornecedor: {determined_supplier}")
                else:
                    logger.error(f"❌ Retorno inesperado do pós-processamento: {type(result_tuple)}")
                    processed_products = []
                    determined_supplier = ""
                    
            except Exception as e:
                logger.error(f"❌ Erro no pós-processamento: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                processed_products = []
                determined_supplier = context_info.get("supplier", "")
            
            combined_result["order_info"]["supplier"] = determined_supplier
            
            # Verificar se processed_products está vazio
            if not processed_products:
                logger.warning(f"⚠️ Nenhum produto após pós-processamento! Verificando produtos originais...")
                # Tentar recuperar produtos originais
                if combined_result.get("products"):
                    logger.info(f"🔧 Tentando usar {len(combined_result['products'])} produtos originais")
                    processed_products = combined_result["products"]
                else:
                    logger.error("❌ Sem produtos para processar!")
            
            if has_json_utils and processed_products:
                supplier = context_info.get("supplier", "")
                supplier_code = get_supplier_code(supplier) if supplier else None
                markup = 2.73
                
                if supplier_code:
                    markup_value = get_markup(supplier_code)
                    if markup_value:
                        markup = markup_value
                
                produtos_antes_fix = len(processed_products)
                processed_products = fix_nan_in_products(processed_products, markup=markup)
                produtos_depois_fix = len(processed_products) if processed_products else 0
                
                logger.info(f"Produtos sanitizados para evitar valores NaN no JSON: {produtos_antes_fix} → {produtos_depois_fix}")

            combined_result["products"] = processed_products if processed_products else []
            logger.info(f"Pós-processamento: {len(combined_result['products'])} produtos únicos identificados")
            
            processing_time = time.time() - start_time
            
            # NOVA: Metadados melhorados
            strategy_adaptations = len(set(r.get("_strategy_used", "") for r in self.page_results_history)) - 1
            
            combined_result["_metadata"] = {
                "pages_processed": total_pages,
                "context_description": context_description,
                "processing_approach": "enhanced_with_generic_system",
                "processing_time_seconds": processing_time,
                "context_info": context_info,
                "layout_analysis": self.current_layout_analysis,
                "final_strategy": self.current_strategy.name if self.current_strategy else "unknown",
                "strategy_adaptations": strategy_adaptations,
                "agents_used": ["ContextAgent", "LayoutDetectionAgent", "GenericStrategyAgent", "ExtractionAgent", "ColorMappingAgent"],
            }
            
            # Log final melhorado
            logger.info(f"=== EXTRAÇÃO CONCLUÍDA ===")
            logger.info(f"Produtos extraídos: {len(combined_result['products'])}")
            logger.info(f"Tempo total: {processing_time:.2f}s")
            logger.info(f"Layout detectado: {self.current_layout_analysis.get('layout_type', 'UNKNOWN')}")
            logger.info(f"Estratégia final: {self.current_strategy.name if self.current_strategy else 'N/A'}")
            logger.info(f"Adaptações: {strategy_adaptations}")
            
            # Atualizar job (mantém-se igual)
            jobs_store[job_id]["model_results"]["gemini"] = {
                "model_name": GEMINI_MODEL,
                "status": "completed",
                "progress": 100.0,
                "result": combined_result,
                "processing_time": processing_time
            }
            
            results_file = os.path.join(os.path.dirname(CONVERTED_DIR), "results", f"{job_id}_gemini.json")
            if has_json_utils:
                success = safe_json_dump(combined_result, results_file)
                if success:
                    logger.info(f"Resultado salvo com sucesso em: {results_file}")
                else:
                    logger.error(f"Falha ao salvar resultado em: {results_file}")
            else:
                try:
                    def sanitize_basic(obj):
                        import math
                        if isinstance(obj, dict):
                            return {k: sanitize_basic(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [sanitize_basic(item) for item in obj]
                        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                            return 0.0
                        else:
                            return obj
                    
                    sanitized_result = sanitize_basic(combined_result)
                    
                    with open(results_file, "w") as f:
                        json.dump(sanitized_result, f, indent=2)
                    logger.info(f"Resultado salvo com sanitização básica em: {results_file}")
                except Exception as e:
                    logger.error(f"Erro ao salvar resultado: {str(e)}")
            
            update_progress_callback(job_id)
            
            rocessing_time = time.time() - start_time
            logger.info(f"⏱️ Tempo total de processamento: {processing_time:.2f}s")  
            logger.info(f"📊 Taxa de produtos por segundo: {total_products/processing_time:.2f}")
            
            needs_validation = False
            for product in combined_result.get("products", []):
                for color in product.get("colors", []):
                    color_code = color.get("color_code", "")
                    # Se tem código original (22222, X0707, etc.), precisa validação
                    if color_code and len(color_code) > 3:
                        needs_validation = True
                        break

            if needs_validation:
                logger.info("⚠️ Executando validação...")
                validation_agent = ValidationAgent()
                validated_result = validation_agent.validate_extraction_result(combined_result)
                if validated_result:
                    combined_result = validated_result
            else:
                logger.info("✅ Color mapping preservado - skip validação")
                
                
            return combined_result
                
        except Exception as e:
            error_message = f"Erro durante o processamento: {str(e)}"
            
            jobs_store[job_id]["model_results"]["gemini"] = {
                "model_name": GEMINI_MODEL,
                "status": "failed",
                "progress": 0.0,
                "error": error_message,
                "processing_time": time.time() - start_time
            }
            
            update_progress_callback(job_id)
            
            return {"error": error_message}
    
    def _post_process_products(self, products: List[Dict[str, Any]], context_info: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:

        processed_products = []
        seen_material_codes = set()
        ref_counters = {}
        
        # ETAPA 1: DETERMINAR FORNECEDOR DO DOCUMENTO (APENAS UMA VEZ)
        supplier_name, supplier_code, markup = determine_best_supplier(context_info)
        original_brand = context_info.get("brand", "")

        # Log do resumo da determinação
        logger.info(f"Fornecedor determinado: '{supplier_name}' (código: {supplier_code}, markup: {markup})")
        
        # DEBUG: Log inicial
        logger.info(f"🔍 Iniciando pós-processamento de {len(products)} produtos")
        
        # ETAPA 2: PROCESSAR PRODUTOS (SEM LÓGICA DE FORNECEDOR INDIVIDUAL)
        for idx, product in enumerate(products):
            # DEBUG: Ver estrutura do produto
            if idx == 0:  # Primeiro produto como exemplo
                logger.debug(f"Exemplo de produto recebido: {json.dumps(product, indent=2, default=str)}")
            
            material_code = product.get("material_code")
            if not material_code:
                logger.warning(f"Produto sem código de material ignorado: {product.get('name', 'sem nome')}")
                continue
            
            # Limpeza do nome do produto
            product_name = product.get("name", "")
            if product_name is None: 
                product_name = ""

            pattern = r'^([A-Za-z\s]+)(?:\s+\d+.*)?$'
            match = re.match(pattern, product_name)

            if match:
                clean_name = match.group(1).strip()
                product["name"] = clean_name
            else:
                clean_name = re.sub(r'\d+', '', product_name).strip()
                clean_name = re.sub(r'\s+', ' ', clean_name).strip()
                product["name"] = clean_name

            if "name" in product:
                product["name"] = product["name"].upper()

            if "brand" in product:
                product["brand"] = product["brand"].upper()

            product_brand = product.get("brand", "") or original_brand
            product["gender"] = determine_gender_by_brand(product_brand)

            if product["gender"] == "MULHER":
                logger.info(f"Produto '{product['name']}' da marca '{product_brand}' definido como MULHER")

            has_valid_data = False
            has_basic_info = bool(material_code and product_name)
            
            # Verificar se tem estrutura de cores válida
            if "colors" in product and isinstance(product.get("colors"), list) and len(product.get("colors")) > 0:
                for color in product["colors"]:
                    if isinstance(color, dict):
                        # Verificar se tem tamanhos OU se tem informações básicas da cor
                        if ("sizes" in color and isinstance(color.get("sizes"), list) and len(color.get("sizes")) > 0) or \
                        (color.get("color_code") or color.get("color_name")):
                            has_valid_data = True
                            break
            
            # ALTERNATIVA: Produto pode ter estrutura diferente (sem array de cores)
            # Verificar se tem informações de produto válidas mesmo sem estrutura de cores
            if not has_valid_data and has_basic_info:
                # Tentar criar estrutura de cores se não existir
                if "colors" not in product or not product["colors"]:
                    # Verificar se há informações de cor/tamanho no nível do produto
                    if any(key in product for key in ["color_code", "color_name", "sizes", "quantity"]):
                        logger.info(f"🔧 Produto {material_code}: Criando estrutura de cores")
                        # Criar estrutura de cor única
                        color_entry = {
                            "color_code": product.get("color_code", "000"),
                            "color_name": product.get("color_name", "Cor Única"),
                            "sizes": []
                        }
                        
                        # Se tem tamanhos diretos no produto
                        if "sizes" in product and isinstance(product["sizes"], list):
                            color_entry["sizes"] = product["sizes"]
                        # Se tem quantidade direta no produto
                        elif "quantity" in product:
                            color_entry["sizes"] = [{
                                "size": product.get("size", "UNICO"),
                                "quantity": product.get("quantity", 0)
                            }]
                        
                        product["colors"] = [color_entry]
                        has_valid_data = True
            
            # DEBUG: Log do status de validação
            logger.debug(f"Produto {material_code}: has_valid_data={has_valid_data}, has_basic_info={has_basic_info}")
            
            if has_valid_data or has_basic_info:
                # NORMALIZAÇÃO DE CATEGORIA
                original_category = product.get("category", "")
                category_upper = original_category.upper() if original_category else ""
                
                # Garantir categoria consistente
                if any(term in category_upper for term in ['POLO', 'POLOSHIRT']):
                    normalized_category = "POLOS"
                elif any(term in category_upper for term in ['SWEATER', 'SWEAT', 'MALHA', 'JERSEY']):
                    normalized_category = "MALHAS"
                else:
                    # Para outras categorias, procurar correspondência em CATEGORIES
                    normalized_category = None
                    for category in CATEGORIES:
                        if category in category_upper or category_upper in category:
                            normalized_category = category
                            break
                    
                    # Se não encontrar, usar "ACESSÓRIOS" como fallback
                    if not normalized_category:
                        normalized_category = "ACESSÓRIOS"
                
                # Atualizar a categoria do produto
                product["category"] = normalized_category
                
                # Logging para debug
                if original_category != normalized_category:
                    logger.debug(f"Categoria normalizada: '{original_category}' → '{normalized_category}' para produto '{product['name']}'")
                
                # Verificar se já processamos este produto (pelo código de material)
                if material_code in seen_material_codes:
                    # Mesclar com produto existente
                    for existing_product in processed_products:
                        if existing_product.get("material_code") == material_code:
                            # Mesclar cores não duplicadas
                            existing_color_codes = {c.get("color_code") for c in existing_product.get("colors", [])}
                            
                            for color in product.get("colors", []):
                                color_code = color.get("color_code")
                                if color_code and color_code not in existing_color_codes:
                                    # Adicionar cor ainda não existente
                                    existing_product["colors"].append(color)
                                    existing_color_codes.add(color_code)
                            
                            # Recalcular total_price
                            subtotals = [color.get("subtotal", 0) for color in existing_product["colors"] 
                                        if color.get("subtotal") is not None]
                            existing_product["total_price"] = sum(subtotals) if subtotals else None
                            
                            logger.debug(f"Produto {material_code} mesclado com existente")
                            break
                else:
                    # Novo produto, adicionar à lista de processados
                    seen_material_codes.add(material_code)
                    
                    # Inicializar contador para este código de material
                    if material_code not in ref_counters:
                        ref_counters[material_code] = 0
                    
                    # Adicionar campo de referências para cada cor e tamanho
                    product_references = []
                    
                    for color in product.get("colors", []):
                        color_code = color.get("color_code", "")
                        color_name = color.get("color_name", "")
                        
                        for size_info in color.get("sizes", []):
                            size = size_info.get("size", "")
                            quantity = size_info.get("quantity", 0)
                            
                            if quantity <= 0:
                                continue
                            
                            # Incrementar contador para este material
                            ref_counters[material_code] += 1
                            counter = ref_counters[material_code]
                            
                            # Criar referência completa
                            reference = f"{material_code}.{counter}"
                            
                            # Criar descrição formatada
                            description = f"{product['name']}[{color_code}/{size}]"
                            
                            # Adicionar referência à lista
                            product_references.append({
                                "reference": reference,
                                "counter": counter,
                                "color_code": color_code,
                                "color_name": color_name,
                                "size": size,
                                "quantity": quantity,
                                "description": description
                            })
                    
                    product["references"] = product_references
                    processed_products.append(product)
                    logger.debug(f"✅ Produto {material_code} adicionado aos processados")
            else:
                logger.warning(f"❌ Produto {material_code} ignorado - sem dados válidos")
        
        # LOG CRÍTICO: Quantos produtos foram processados
        logger.info(f"📊 Pós-processamento concluído: {len(processed_products)} de {len(products)} produtos válidos")
        
        # ETAPA 3: ATRIBUIR FORNECEDOR A TODOS OS PRODUTOS (APENAS UMA VEZ)
        logger.debug(f"📦 Antes de atribuir fornecedor: {len(processed_products)} produtos")
        
        if processed_products:  # Só se houver produtos
            # DEBUG: Verificar se assign_supplier_to_products está funcionando
            produtos_antes = len(processed_products)
            processed_products = assign_supplier_to_products(processed_products, supplier_name, markup)
            produtos_depois = len(processed_products) if processed_products else 0
            
            logger.debug(f"📦 Após assign_supplier_to_products: {produtos_antes} → {produtos_depois} produtos")
            
            # Verificar se processed_products ainda existe e não é None
            if not processed_products:
                logger.error("❌ assign_supplier_to_products retornou None ou lista vazia!")
                processed_products = []
            else:
                # ETAPA 3.5: GARANTIR QUE TODOS OS CAMPOS ESTÃO CORRETOS
                for product in processed_products:
                    # Preservar marca original se existir
                    if original_brand and original_brand not in ["", "Marca não identificada"]:
                        product["brand"] = original_brand
                    
                    # Forçar o fornecedor normalizado
                    product["supplier"] = supplier_name
                    
                    # Garantir que cores têm fornecedor correto
                    for color in product.get("colors", []):
                        color["supplier"] = supplier_name
                    
                    # CRÍTICO: Garantir que referências têm fornecedor correto
                    for reference in product.get("references", []):
                        reference["supplier"] = supplier_name

                # ETAPA 4: FINALIZAR
                processed_products.sort(key=lambda p: p.get("material_code", ""))
                
                try:
                    from app.utils.barcode_generator import add_barcodes_to_products
                    produtos_antes_barcode = len(processed_products)
                    processed_products = add_barcodes_to_products(processed_products)
                    produtos_depois_barcode = len(processed_products) if processed_products else 0
                    logger.debug(f"📦 Após add_barcodes_to_products: {produtos_antes_barcode} → {produtos_depois_barcode} produtos")
                except ImportError:
                    logger.warning("Módulo barcode_generator não encontrado, pulando geração de códigos de barras")
        
        # LOG FINAL ANTES DE RETORNAR
        logger.info(f"📊 Retornando {len(processed_products)} produtos processados")
        
        quality_report = self._check_size_quality(processed_products)
        
        if quality_report['issues_found'] > 0:
            logger.warning(f"⚠️ {quality_report['issues_found']} problemas de tamanhos detectados")
            for issue in quality_report['sample_issues'][:3]:
                logger.warning(f"   - {issue}")
        
        logger.info(f"📊 Qualidade dos tamanhos: {quality_report['quality_score']:.1%}")

        return processed_products, supplier_name

    def _check_size_quality(self, products: List[Dict]) -> Dict[str, Any]:
        """
        Verifica qualidade geral dos tamanhos extraídos
        """
        total_products = len(products)
        issues = []
        single_size_count = 0
        uniform_quantity_count = 0
        
        for product in products:
            material_code = product.get('material_code', '')
            category = product.get('category', '').upper()
            
            for color in product.get('colors', []):
                sizes = color.get('sizes', [])
                
                # Issue 1: Tamanho único suspeito
                if len(sizes) == 1:
                    size_name = sizes[0].get('size', '')
                    if (size_name in ['UN', 'UNICO', 'UNI'] and 
                        category in ['MALHAS', 'CAMISAS', 'VESTIDOS', 'CALÇAS']):
                        single_size_count += 1
                        issues.append(f"{material_code}: Tamanho único '{size_name}' para {category}")
                
                # Issue 2: Quantidades todas iguais (suspeito)
                if len(sizes) > 2:
                    quantities = [s.get('quantity', 0) for s in sizes]
                    if len(set(quantities)) == 1 and quantities[0] == 1:
                        uniform_quantity_count += 1
                        issues.append(f"{material_code}: Todas quantidades = 1")
        
        issues_found = len(issues)
        quality_score = max(0, 1 - (issues_found / max(1, total_products)))
        
        return {
            'total_products': total_products,
            'issues_found': issues_found,
            'single_size_issues': single_size_count,
            'uniform_quantity_issues': uniform_quantity_count,
            'quality_score': quality_score,
            'sample_issues': issues[:5]
        }
    