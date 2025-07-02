# app/extractors/data_verification_agent.py
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import google.generativeai as genai
from PIL import Image

from app.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """Resultado da verificação de um produto"""
    product_index: int
    material_code_match: float
    colors_match: float
    sizes_match: float
    prices_match: float
    overall_accuracy: float
    discrepancies: List[str]
    verification_details: Dict[str, Any]

class DataVerificationAgent:
    """
    Agente que confirma os dados extraídos comparando com o PDF original.
    Dá percentagens de precisão para cada produto e identifica discrepâncias.
    """
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        
        self.verification_stats = {
            "total_verifications": 0,
            "average_accuracy": 0.0,
            "products_verified": 0,
            "discrepancies_found": 0,
            "verification_details": []
        }
    
    async def verify_extracted_data(
        self, 
        extracted_products: List[Dict[str, Any]], 
        original_images: List[str],
        document_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Verifica dados extraídos comparando com imagens originais
        
        Args:
            extracted_products: Produtos extraídos pelo sistema
            original_images: Lista de caminhos das imagens do PDF original
            document_context: Contexto adicional do documento
            
        Returns:
            Dict com resultados de verificação detalhados
        """
        if not extracted_products or not original_images:
            return {"verification_results": [], "summary": {}}
        
        logger.info(f"🔍 Iniciando verificação de {len(extracted_products)} produtos contra {len(original_images)} páginas...")
        
        verification_results = []
        total_accuracy = 0.0
        total_discrepancies = 0
        
        # Para cada produto extraído, verificar contra as imagens
        for i, product in enumerate(extracted_products):
            logger.debug(f"Verificando produto {i+1}: {product.get('name', 'N/A')}")
            
            # Encontrar a melhor imagem correspondente para este produto
            best_image_index = await self._find_best_matching_image(
                product, original_images, i
            )
            
            if best_image_index is not None:
                # Verificar produto contra a imagem específica
                verification = await self._verify_product_against_image(
                    product, original_images[best_image_index], i
                )
                
                verification_results.append(verification)
                total_accuracy += verification.overall_accuracy
                total_discrepancies += len(verification.discrepancies)
                
                # Log de resultado
                logger.debug(f"  Produto {i+1}: {verification.overall_accuracy:.1f}% de precisão, "
                           f"{len(verification.discrepancies)} discrepâncias")
            else:
                # Não foi possível encontrar imagem correspondente
                logger.warning(f"Produto {i+1}: Não foi possível encontrar imagem correspondente")
                verification_results.append(VerificationResult(
                    product_index=i,
                    material_code_match=0.0,
                    colors_match=0.0,
                    sizes_match=0.0,
                    prices_match=0.0,
                    overall_accuracy=0.0,
                    discrepancies=["Imagem correspondente não encontrada"],
                    verification_details={"error": "No matching image found"}
                ))
        
        # Gerar resumo da verificação
        summary = self._generate_verification_summary(
            verification_results, extracted_products, total_accuracy, total_discrepancies
        )
        
        # Atualizar estatísticas
        self.verification_stats["total_verifications"] += 1
        self.verification_stats["products_verified"] += len(verification_results)
        self.verification_stats["average_accuracy"] = total_accuracy / max(len(verification_results), 1)
        self.verification_stats["discrepancies_found"] += total_discrepancies
        
        # Log do resumo
        self._log_verification_summary(summary)
        
        return {
            "verification_results": [self._verification_result_to_dict(vr) for vr in verification_results],
            "summary": summary,
            "global_stats": self.verification_stats
        }
    
    async def _find_best_matching_image(
        self, 
        product: Dict[str, Any], 
        images: List[str], 
        product_index: int
    ) -> Optional[int]:
        """
        Encontra a imagem que melhor corresponde ao produto
        """
        material_code = product.get("material_code", "")
        
        if not material_code:
            # Se não há material code, usar índice estimado
            estimated_page = min(product_index // 5, len(images) - 1)  # ~5 produtos por página
            return estimated_page
        
        # Verificar cada imagem em busca do material code
        for i, image_path in enumerate(images):
            try:
                contains_code = await self._check_image_contains_material_code(
                    image_path, material_code
                )
                if contains_code:
                    return i
            except Exception as e:
                logger.debug(f"Erro ao verificar imagem {i}: {str(e)}")
                continue
        
        # Fallback: usar posição estimada
        estimated_page = min(product_index // 5, len(images) - 1)
        return estimated_page
    
    async def _check_image_contains_material_code(
        self, 
        image_path: str, 
        material_code: str
    ) -> bool:
        """
        Verifica se uma imagem contém um material code específico
        """
        try:
            image = Image.open(image_path)
            
            prompt = f"""
            Verifica se esta imagem contém o código de material "{material_code}".
            
            Responde apenas:
            - "SIM" se o código estiver visível na imagem
            - "NÃO" se o código não estiver visível
            
            Procura por: {material_code}
            """
            
            response = self.model.generate_content([prompt, image])
            return "SIM" in response.text.upper()
            
        except Exception as e:
            logger.debug(f"Erro ao verificar material code na imagem: {str(e)}")
            return False
    
    async def _verify_product_against_image(
        self, 
        product: Dict[str, Any], 
        image_path: str, 
        product_index: int
    ) -> VerificationResult:
        """
        Verifica um produto específico contra uma imagem
        """
        try:
            image = Image.open(image_path)
            
            # Preparar dados do produto para verificação
            product_summary = self._prepare_product_for_verification(product)
            
            verification_prompt = f"""
            # VERIFICAÇÃO DE DADOS EXTRAÍDOS
            
            Verifica se os dados extraídos correspondem ao que está visível na imagem.
            
            ## DADOS EXTRAÍDOS PARA VERIFICAR:
            {product_summary}
            
            ## TAREFA:
            Para cada campo, analisa a imagem e indica:
            1. **Material Code**: O código está correto? (0-100%)
            2. **Cores**: As cores e códigos estão corretos? (0-100%)
            3. **Tamanhos**: Os tamanhos e quantidades estão corretos? (0-100%) 
            4. **Preços**: Os preços estão corretos? (0-100%)
            
            ## RESPOSTA OBRIGATÓRIA:
            ```json
            {{
              "material_code_accuracy": 85,
              "colors_accuracy": 90,
              "sizes_accuracy": 75,
              "prices_accuracy": 95,
              "discrepancies": [
                "Tamanho 46 tem quantidade 3 na imagem mas 2 nos dados extraídos",
                "Preço deveria ser 75.50 em vez de 75.00"
              ],
              "verification_notes": "Observações gerais sobre a verificação"
            }}
            ```
            
            Sê preciso e específico nas discrepâncias encontradas.
            """
            
            response = self.model.generate_content([verification_prompt, image])
            
            # Extrair dados da resposta
            verification_data = self._extract_verification_json(response.text)
            
            if verification_data:
                # Calcular precisão geral
                accuracies = [
                    verification_data.get("material_code_accuracy", 0),
                    verification_data.get("colors_accuracy", 0),
                    verification_data.get("sizes_accuracy", 0),
                    verification_data.get("prices_accuracy", 0)
                ]
                overall_accuracy = sum(accuracies) / len(accuracies)
                
                return VerificationResult(
                    product_index=product_index,
                    material_code_match=verification_data.get("material_code_accuracy", 0),
                    colors_match=verification_data.get("colors_accuracy", 0),
                    sizes_match=verification_data.get("sizes_accuracy", 0),
                    prices_match=verification_data.get("prices_accuracy", 0),
                    overall_accuracy=overall_accuracy,
                    discrepancies=verification_data.get("discrepancies", []),
                    verification_details=verification_data
                )
            else:
                raise Exception("Não foi possível extrair dados de verificação")
                
        except Exception as e:
            logger.error(f"Erro na verificação do produto {product_index}: {str(e)}")
            return VerificationResult(
                product_index=product_index,
                material_code_match=0.0,
                colors_match=0.0,
                sizes_match=0.0,
                prices_match=0.0,
                overall_accuracy=0.0,
                discrepancies=[f"Erro na verificação: {str(e)}"],
                verification_details={"error": str(e)}
            )
    
    def _prepare_product_for_verification(self, product: Dict[str, Any]) -> str:
        """
        Prepara resumo do produto para verificação
        """
        summary_parts = []
        
        # Informações básicas
        summary_parts.append(f"**Nome**: {product.get('name', 'N/A')}")
        summary_parts.append(f"**Material Code**: {product.get('material_code', 'N/A')}")
        summary_parts.append(f"**Categoria**: {product.get('category', 'N/A')}")
        
        # Cores e tamanhos
        colors = product.get("colors", [])
        summary_parts.append(f"**Total de Cores**: {len(colors)}")
        
        for i, color in enumerate(colors[:3]):  # Mostrar apenas primeiras 3 cores
            color_info = f"  Cor {i+1}: {color.get('color_name', 'N/A')} (código: {color.get('color_code', 'N/A')})"
            
            sizes = color.get("sizes", [])
            if sizes:
                size_list = [f"{s.get('size', 'N/A')}:{s.get('quantity', 0)}" for s in sizes]
                color_info += f" - Tamanhos: {', '.join(size_list)}"
            
            price = color.get("unit_price", 0)
            if price:
                color_info += f" - Preço: {price}"
            
            summary_parts.append(color_info)
        
        if len(colors) > 3:
            summary_parts.append(f"  ... e mais {len(colors) - 3} cores")
        
        return "\n".join(summary_parts)
    
    def _extract_verification_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Extrai JSON de verificação da resposta
        """
        try:
            import json
            
            # Procurar bloco JSON
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, response_text)
            
            if matches:
                return json.loads(matches[0])
            
            # Tentar extrair objeto JSON
            json_pattern = r'\{[\s\S]*\}'
            matches = re.findall(json_pattern, response_text)
            
            for potential_json in matches:
                try:
                    result = json.loads(potential_json)
                    if isinstance(result, dict) and "material_code_accuracy" in result:
                        return result
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Erro ao extrair JSON de verificação: {str(e)}")
            return None
    
    def _generate_verification_summary(
        self, 
        verification_results: List[VerificationResult],
        original_products: List[Dict[str, Any]],
        total_accuracy: float,
        total_discrepancies: int
    ) -> Dict[str, Any]:
        """
        Gera resumo da verificação
        """
        if not verification_results:
            return {}
        
        # Calcular estatísticas
        avg_accuracy = total_accuracy / len(verification_results)
        
        # Contar produtos por nível de precisão
        high_accuracy = sum(1 for vr in verification_results if vr.overall_accuracy >= 80)
        medium_accuracy = sum(1 for vr in verification_results if 50 <= vr.overall_accuracy < 80)
        low_accuracy = sum(1 for vr in verification_results if vr.overall_accuracy < 50)
        
        # Estatísticas por campo
        field_stats = {
            "material_code": sum(vr.material_code_match for vr in verification_results) / len(verification_results),
            "colors": sum(vr.colors_match for vr in verification_results) / len(verification_results),
            "sizes": sum(vr.sizes_match for vr in verification_results) / len(verification_results),
            "prices": sum(vr.prices_match for vr in verification_results) / len(verification_results)
        }
        
        # Discrepâncias mais comuns
        all_discrepancies = []
        for vr in verification_results:
            all_discrepancies.extend(vr.discrepancies)
        
        # Produtos com maior precisão
        top_products = sorted(verification_results, key=lambda x: x.overall_accuracy, reverse=True)[:3]
        
        # Produtos que precisam revisão
        problematic_products = [vr for vr in verification_results if vr.overall_accuracy < 70]
        
        return {
            "verification_overview": {
                "total_products_verified": len(verification_results),
                "average_accuracy": f"{avg_accuracy:.1f}%",
                "total_discrepancies": total_discrepancies,
                "accuracy_distribution": {
                    "high_accuracy": f"{high_accuracy} produtos (≥80%)",
                    "medium_accuracy": f"{medium_accuracy} produtos (50-79%)",
                    "low_accuracy": f"{low_accuracy} produtos (<50%)"
                }
            },
            "field_accuracy": {
                "material_codes": f"{field_stats['material_code']:.1f}%",
                "colors": f"{field_stats['colors']:.1f}%", 
                "sizes": f"{field_stats['sizes']:.1f}%",
                "prices": f"{field_stats['prices']:.1f}%"
            },
            "top_performing_products": [
                {
                    "product_index": vr.product_index,
                    "material_code": original_products[vr.product_index].get('material_code', 'N/A'),
                    "accuracy": f"{vr.overall_accuracy:.1f}%"
                }
                for vr in top_products
            ],
            "products_needing_review": [
                {
                    "product_index": vr.product_index,
                    "material_code": original_products[vr.product_index].get('material_code', 'N/A'),
                    "accuracy": f"{vr.overall_accuracy:.1f}%",
                    "main_issues": vr.discrepancies[:2]
                }
                for vr in problematic_products
            ],
            "recommendations": self._generate_verification_recommendations(verification_results, field_stats)
        }
    
    def _generate_verification_recommendations(
        self, 
        verification_results: List[VerificationResult],
        field_stats: Dict[str, float]
    ) -> List[str]:
        """
        Gera recomendações baseadas na verificação
        """
        recommendations = []
        
        avg_accuracy = sum(vr.overall_accuracy for vr in verification_results) / len(verification_results)
        
        if avg_accuracy < 70:
            recommendations.append("URGENTE: Precisão geral baixa - revisar processo de extração")
        elif avg_accuracy < 85:
            recommendations.append("Precisão moderada - considerar melhorias no sistema")
        
        # Recomendações por campo
        if field_stats['material_code'] < 80:
            recommendations.append("Melhorar detecção de material codes")
        
        if field_stats['sizes'] < 75:
            recommendations.append("Verificar extração de tamanhos e quantidades")
        
        if field_stats['colors'] < 80:
            recommendations.append("Melhorar mapeamento de cores")
        
        if field_stats['prices'] < 85:
            recommendations.append("Verificar extração de preços")
        
        # Verificar produtos problemáticos
        problematic_count = sum(1 for vr in verification_results if vr.overall_accuracy < 50)
        if problematic_count > len(verification_results) * 0.3:
            recommendations.append("Muitos produtos com baixa precisão - verificar qualidade do PDF original")
        
        if not recommendations:
            recommendations.append("Excelente precisão - sistema funcionando bem")
        
        return recommendations
    
    def _verification_result_to_dict(self, vr: VerificationResult) -> Dict[str, Any]:
        """
        Converte VerificationResult para dicionário
        """
        return {
            "product_index": vr.product_index,
            "accuracy_scores": {
                "material_code": f"{vr.material_code_match:.1f}%",
                "colors": f"{vr.colors_match:.1f}%",
                "sizes": f"{vr.sizes_match:.1f}%",
                "prices": f"{vr.prices_match:.1f}%",
                "overall": f"{vr.overall_accuracy:.1f}%"
            },
            "discrepancies_found": vr.discrepancies,
            "verification_details": vr.verification_details,
            "accuracy_level": "ALTO" if vr.overall_accuracy >= 80 else "MÉDIO" if vr.overall_accuracy >= 50 else "BAIXO"
        }
    
    def _log_verification_summary(self, summary: Dict[str, Any]):
        """
        Log do resumo de verificação
        """
        if not summary:
            return
        
        overview = summary.get("verification_overview", {})
        field_accuracy = summary.get("field_accuracy", {})
        
        logger.info("=" * 70)
        logger.info("🔍 RELATÓRIO DE VERIFICAÇÃO DE DADOS")
        logger.info("=" * 70)
        logger.info(f"   Produtos verificados: {overview.get('total_products_verified', 0)}")
        logger.info(f"   Precisão média: {overview.get('average_accuracy', '0%')}")
        logger.info(f"   Total de discrepâncias: {overview.get('total_discrepancies', 0)}")
        
        logger.info("\n📊 PRECISÃO POR CAMPO:")
        logger.info(f"   Material Codes: {field_accuracy.get('material_codes', '0%')}")
        logger.info(f"   Cores: {field_accuracy.get('colors', '0%')}")
        logger.info(f"   Tamanhos: {field_accuracy.get('sizes', '0%')}")
        logger.info(f"   Preços: {field_accuracy.get('prices', '0%')}")
        
        distribution = overview.get("accuracy_distribution", {})
        logger.info("\n📈 DISTRIBUIÇÃO DE PRECISÃO:")
        logger.info(f"   🟢 Alta: {distribution.get('high_accuracy', '0')}")
        logger.info(f"   🟡 Média: {distribution.get('medium_accuracy', '0')}")
        logger.info(f"   🔴 Baixa: {distribution.get('low_accuracy', '0')}")
        
        problematic = summary.get("products_needing_review", [])
        if problematic:
            logger.info(f"\n⚠️  PRODUTOS PARA REVISÃO ({len(problematic)}):")
            for product in problematic[:3]:
                logger.info(f"   • {product['material_code']}: {product['accuracy']} - {', '.join(product['main_issues'][:1])}")
        
        recommendations = summary.get("recommendations", [])
        if recommendations:
            logger.info(f"\n💡 RECOMENDAÇÕES:")
            for rec in recommendations[:3]:
                logger.info(f"   • {rec}")
        
        logger.info("=" * 70)