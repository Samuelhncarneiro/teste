# app/extractors/generic_strategy_agent.py
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExtractionStrategy:
    """Estratégia de extração genérica"""
    name: str
    confidence: float
    approach: str
    specific_instructions: Dict[str, str]
    fallback_strategy: Optional[str] = None

class GenericStrategyAgent:
    """
    Agente de estratégia completamente genérico que seleciona abordagens
    baseado apenas na análise estrutural do documento, sem conhecimento
    específico de fornecedores.
    """
    
    def __init__(self):
        self.strategies = self._initialize_generic_strategies()
        self.performance_history = {} 
    
    def _initialize_generic_strategies(self) -> Dict[str, ExtractionStrategy]:
        """
        Estratégias puramente baseadas em estrutura de layout
        """
        return {
            "structured_table": ExtractionStrategy(
                name="Structured Table Extraction",
                confidence=0.9,
                approach="table_extraction",
                specific_instructions={
                    "scanning_method": "Processar sistematicamente linha por linha",
                    "column_mapping": "Identificar colunas por posição e cabeçalhos",
                    "cell_interpretation": "Células vazias = dados não disponíveis",
                    "data_validation": "Verificar consistência entre linhas",
                    "header_detection": "Usar primeira linha para mapear estrutura",
                    "row_processing": "Manter correspondência rigorosa linha-dados"
                },
                fallback_strategy="adaptive_hybrid"
            ),
            
            "sequential_list": ExtractionStrategy(
                name="Sequential List Extraction", 
                confidence=0.8,
                approach="list_extraction",
                specific_instructions={
                    "item_detection": "Cada linha é um item independente",
                    "data_parsing": "Extrair dados na ordem de aparição",
                    "pattern_recognition": "Identificar padrão repetitivo por linha",
                    "grouping_logic": "Agrupar dados relacionados na mesma linha",
                    "sequence_validation": "Verificar continuidade lógica",
                    "line_processing": "Processar linha completa como unidade"
                },
                fallback_strategy="adaptive_hybrid"
            ),
            
            "adaptive_hybrid": ExtractionStrategy(
                name="Adaptive Hybrid Extraction",
                confidence=0.7,
                approach="hybrid_extraction", 
                specific_instructions={
                    "section_analysis": "Analisar cada seção independentemente",
                    "strategy_switching": "Adaptar método por região do documento",
                    "pattern_detection": "Detectar mudanças na estrutura",
                    "multi_approach": "Combinar múltiplas técnicas conforme necessário",
                    "flexibility": "Ajustar estratégia baseada em resultados",
                    "validation": "Verificar consistência entre seções"
                },
                fallback_strategy="conservative_scan"
            ),
            
            "form_field_extraction": ExtractionStrategy(
                name="Form Field Extraction",
                confidence=0.6,
                approach="form_extraction",
                specific_instructions={
                    "field_mapping": "Mapear campos por posição e labels",
                    "value_extraction": "Extrair valores associados a campos",
                    "structure_detection": "Identificar grupos de campos relacionados",
                    "label_recognition": "Usar labels para identificar tipos de dados",
                    "field_validation": "Verificar completude dos campos",
                    "form_logic": "Seguir lógica de formulário estruturado"
                },
                fallback_strategy="adaptive_hybrid"
            ),
            
            "conservative_scan": ExtractionStrategy(
                name="Conservative Scan Extraction",
                confidence=0.4,
                approach="adaptive_extraction",
                specific_instructions={
                    "broad_scanning": "Usar múltiplas técnicas simultaneamente",
                    "pattern_searching": "Procurar padrões conhecidos universais",
                    "safe_extraction": "Extrair apenas dados com alta confiança",
                    "multiple_passes": "Fazer múltiplas passadas com técnicas diferentes",
                    "conservative_approach": "Evitar assumir estruturas não confirmadas",
                    "verification": "Verificar cada extração independentemente"
                },
                fallback_strategy=None
            )
        }
    
    def select_strategy(
        self, 
        layout_analysis: Dict[str, Any],
        page_number: int = 1,
        previous_results: Optional[List[Dict]] = None
    ) -> ExtractionStrategy:
        """
        Seleciona estratégia baseada puramente na análise de layout
        """
        strategy_scores = {}
        
        for strategy_name, strategy in self.strategies.items():
            score = self._calculate_layout_based_score(
                strategy, layout_analysis, page_number, previous_results
            )
            strategy_scores[strategy_name] = score
            
            logger.debug(f"Estratégia '{strategy_name}': score {score:.3f}")
        
        # Aplicar histórico de performance se disponível
        adjusted_scores = self._apply_performance_history(strategy_scores)
        
        # Selecionar melhor estratégia
        best_strategy_name = max(adjusted_scores, key=adjusted_scores.get)
        best_strategy = self.strategies[best_strategy_name]
        
        logger.info(f"Estratégia selecionada: '{best_strategy_name}' "
                   f"(score: {adjusted_scores[best_strategy_name]:.3f})")
        
        return best_strategy
    
    def _calculate_layout_based_score(
        self,
        strategy: ExtractionStrategy,
        layout_analysis: Dict[str, Any],
        page_number: int,
        previous_results: Optional[List[Dict]]
    ) -> float:
        """
        Calcula score baseado exclusivamente na análise de layout
        """
        score = strategy.confidence  # Score base
        
        layout_type = layout_analysis.get("layout_type", "UNKNOWN")
        extraction_strategy = layout_analysis.get("extraction_strategy", "")
        confidence = layout_analysis.get("confidence", 0.0)
        
        # FATOR 1: Compatibilidade direta com layout detectado
        compatibility_bonus = self._get_layout_compatibility(strategy.approach, layout_type)
        score += compatibility_bonus * confidence
        
        # FATOR 2: Análise técnica - estrutura de colunas
        technical_analysis = layout_analysis.get("technical_analysis", {})
        if not technical_analysis.get("error"):
            column_info = technical_analysis.get("column_detection", {})
            column_count = column_info.get("column_count", 0)
            column_confidence = column_info.get("confidence", 0)
            
            if strategy.approach == "table_extraction":
                if column_count >= 5:  # Muitas colunas = provável tabela
                    score += 0.3 * column_confidence
                elif column_count >= 3:
                    score += 0.2 * column_confidence
            elif strategy.approach == "list_extraction":
                if 2 <= column_count <= 4:  # Poucas colunas = provável lista
                    score += 0.2 * column_confidence
        
        # FATOR 3: Análise visual - elementos estruturais
        visual_analysis = layout_analysis.get("visual_analysis", {})
        if not visual_analysis.get("error"):
            structural = visual_analysis.get("structural_elements", {})
            
            if strategy.approach == "table_extraction":
                if structural.get("has_clear_grid"):
                    score += 0.2
                if structural.get("has_headers"):
                    score += 0.1
                if structural.get("alignment_quality") == "high":
                    score += 0.1
            
            elif strategy.approach == "list_extraction":
                if structural.get("repetitive_patterns"):
                    score += 0.2
                if structural.get("data_density") in ["medium", "high"]:
                    score += 0.1
            
            elif strategy.approach == "form_extraction":
                if visual_analysis.get("primary_layout") == "FORM_FIELDS":
                    score += 0.3
        
        # FATOR 4: Análise de conteúdo - padrões de dados
        content_analysis = layout_analysis.get("content_analysis", {})
        if not content_analysis.get("error"):
            # Indicadores de tabela
            table_indicators = content_analysis.get("table_indicators", {})
            if table_indicators.get("has_table_structure"):
                table_confidence = table_indicators.get("table_confidence", 0)
                if strategy.approach == "table_extraction":
                    score += 0.2 * table_confidence
                elif strategy.approach == "hybrid_extraction":
                    score += 0.1 * table_confidence
            
            # Estrutura de produtos
            product_indicators = content_analysis.get("product_indicators", {})
            if product_indicators.get("has_product_structure"):
                if strategy.approach in ["table_extraction", "list_extraction"]:
                    score += 0.1
        
        # FATOR 5: Complexidade do documento
        complexity_bonus = self._calculate_complexity_bonus(strategy, layout_analysis)
        score += complexity_bonus
        
        # FATOR 6: Consistência com páginas anteriores
        if previous_results and page_number > 1:
            consistency_bonus = self._calculate_consistency_bonus(strategy, previous_results)
            score += consistency_bonus
        
        return max(0.0, min(1.0, score))
    
    def _get_layout_compatibility(self, approach: str, layout_type: str) -> float:
        """
        Matriz de compatibilidade entre abordagem e tipo de layout
        """
        compatibility_matrix = {
            "table_extraction": {
                "GRID_TABULAR": 0.4,
                "LIST_VERTICAL": 0.1,
                "LIST_HORIZONTAL": 0.2,
                "HYBRID_MIXED": 0.2,
                "FORM_FIELDS": 0.0,
                "FREE_TEXT": 0.0
            },
            "list_extraction": {
                "GRID_TABULAR": 0.1,
                "LIST_VERTICAL": 0.4,
                "LIST_HORIZONTAL": 0.3,
                "HYBRID_MIXED": 0.2,
                "FORM_FIELDS": 0.1,
                "FREE_TEXT": 0.0
            },
            "hybrid_extraction": {
                "GRID_TABULAR": 0.2,
                "LIST_VERTICAL": 0.2,
                "LIST_HORIZONTAL": 0.2,
                "HYBRID_MIXED": 0.4,
                "FORM_FIELDS": 0.2,
                "FREE_TEXT": 0.1
            },
            "form_extraction": {
                "GRID_TABULAR": 0.0,
                "LIST_VERTICAL": 0.1,
                "LIST_HORIZONTAL": 0.1,
                "HYBRID_MIXED": 0.1,
                "FORM_FIELDS": 0.4,
                "FREE_TEXT": 0.0
            },
            "adaptive_extraction": {
                "GRID_TABULAR": 0.1,
                "LIST_VERTICAL": 0.1,
                "LIST_HORIZONTAL": 0.1,
                "HYBRID_MIXED": 0.1,
                "FORM_FIELDS": 0.1,
                "FREE_TEXT": 0.3
            }
        }
        
        return compatibility_matrix.get(approach, {}).get(layout_type, 0.0)
    
    def _calculate_complexity_bonus(
        self, 
        strategy: ExtractionStrategy, 
        layout_analysis: Dict[str, Any]
    ) -> float:
        """
        Calcula bonus baseado na complexidade detectada
        """
        bonus = 0.0
        
        # Análise visual - indicadores de complexidade
        visual = layout_analysis.get("visual_analysis", {})
        if not visual.get("error"):
            extraction_hints = visual.get("extraction_hints", {})
            complexity = extraction_hints.get("complexity_level", "simple")
            
            if complexity == "complex":
                if strategy.approach in ["hybrid_extraction", "adaptive_extraction"]:
                    bonus += 0.15
                else:
                    bonus -= 0.05  # Penalizar estratégias simples para docs complexos
            
            elif complexity == "simple":
                if strategy.approach in ["table_extraction", "list_extraction"]:
                    bonus += 0.1
        
        # Múltiplas páginas
        visual_elements = visual.get("structural_elements", {})
        if visual_elements.get("data_density") == "high":
            if strategy.approach == "adaptive_extraction":
                bonus += 0.1
        
        return bonus
    
    def _calculate_consistency_bonus(
        self, 
        strategy: ExtractionStrategy, 
        previous_results: List[Dict]
    ) -> float:
        """
        Calcula bonus por consistência com resultados anteriores
        """
        if not previous_results:
            return 0.0
        
        # Analisar qualidade dos resultados anteriores
        recent_results = previous_results[-2:]  # Últimas 2 páginas
        
        success_rate = 0.0
        for result in recent_results:
            products = result.get("products", [])
            if len(products) > 0:
                # Verificar qualidade dos produtos
                complete_products = sum(
                    1 for p in products 
                    if p.get("name") and p.get("colors") and 
                    any(len(c.get("sizes", [])) > 0 for c in p.get("colors", []))
                )
                success_rate += complete_products / max(1, len(products))
        
        avg_success = success_rate / len(recent_results) if recent_results else 0
        
        # Se estratégia anterior funcionou bem, dar bonus
        if avg_success > 0.7:
            return 0.1
        elif avg_success < 0.3:
            return -0.1  # Penalizar se estratégia anterior falhou
        
        return 0.0
    
    def _apply_performance_history(self, strategy_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Aplica histórico de performance para ajustar scores
        """
        adjusted_scores = strategy_scores.copy()
        
        for strategy_name, score in strategy_scores.items():
            if strategy_name in self.performance_history:
                history = self.performance_history[strategy_name]
                
                # Média de sucesso histórico
                avg_success = sum(history) / len(history)
                
                # Ajustar score baseado no histórico
                if avg_success > 0.8:
                    adjusted_scores[strategy_name] += 0.05  # Bonus por bom histórico
                elif avg_success < 0.3:
                    adjusted_scores[strategy_name] -= 0.05  # Penalidade por mau histórico
        
        return adjusted_scores
    
    def record_strategy_performance(self, strategy_name: str, success_rate: float):
        """
        Registra performance de uma estratégia para aprendizagem
        """
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []
        
        # Manter apenas últimos 10 registros
        history = self.performance_history[strategy_name]
        history.append(success_rate)
        
        if len(history) > 10:
            history.pop(0)
        
        logger.debug(f"Performance registrada para '{strategy_name}': {success_rate:.2f}")
    
    def adapt_strategy_for_page(
        self,
        current_strategy: ExtractionStrategy,
        page_result: Dict[str, Any],
        page_number: int,
        layout_analysis: Dict[str, Any]
    ) -> Optional[ExtractionStrategy]:
        """
        Adapta estratégia baseada nos resultados da página
        """
        # Calcular qualidade do resultado
        products = page_result.get("products", [])
        has_error = "error" in page_result
        
        if has_error:
            quality = 0.0
        elif not products:
            quality = 0.0
        else:
            # Calcular qualidade baseada na completude dos produtos
            complete_products = 0
            for product in products:
                if (product.get("name") and 
                    product.get("colors") and 
                    any(len(c.get("sizes", [])) > 0 for c in product.get("colors", []))):
                    complete_products += 1
            
            quality = complete_products / len(products) if products else 0.0
        
        # Registrar performance
        self.record_strategy_performance(current_strategy.name, quality)
        
        # Decidir se mudar estratégia
        if quality < 0.3:  # Resultado pobre
            logger.warning(f"Página {page_number}: Qualidade baixa ({quality:.2f}) "
                          f"com estratégia '{current_strategy.name}'")
            
            # Tentar estratégia de fallback
            if current_strategy.fallback_strategy:
                fallback = self.strategies.get(current_strategy.fallback_strategy)
                if fallback:
                    logger.info(f"Mudando para estratégia de fallback: '{fallback.name}'")
                    return fallback
            
            # Se não tem fallback, selecionar nova estratégia
            return self.select_strategy(layout_analysis, page_number)
        
        # Se qualidade boa, manter estratégia
        return None
    
    def get_strategy_instructions(
        self, 
        strategy: ExtractionStrategy, 
        layout_analysis: Dict[str, Any],
        page_context: Dict[str, Any]
    ) -> str:
        """
        Gera instruções específicas para a estratégia selecionada
        """
        instructions = [f"## ESTRATÉGIA: {strategy.name.upper()}"]
        instructions.append(f"**Abordagem**: {strategy.approach}")
        instructions.append(f"**Confiança Base**: {strategy.confidence:.2f}")
        instructions.append("")
        
        # Instruções específicas da estratégia
        instructions.append("### INSTRUÇÕES ESPECÍFICAS:")
        for key, instruction in strategy.specific_instructions.items():
            readable_key = key.replace('_', ' ').title()
            instructions.append(f"- **{readable_key}**: {instruction}")
        
        # Adicionar insights da análise de layout
        layout_instructions = self._get_layout_specific_instructions(
            strategy, layout_analysis
        )
        if layout_instructions:
            instructions.append("")
            instructions.append("### ADAPTAÇÕES BASEADAS NO LAYOUT:")
            instructions.extend(layout_instructions)
        
        page_number = page_context.get("page_number", 1)
        if page_number > 1:
            instructions.append("")
            instructions.append("### CONSIDERAÇÕES PARA PÁGINAS ADICIONAIS:")
            instructions.append("- Manter consistência com padrão estabelecido")
            instructions.append("- Verificar se estrutura se mantém igual")
            instructions.append("- Adaptar se houver mudanças detectadas")
        
        return "\n".join(instructions)
    
    def _get_layout_specific_instructions(
        self, 
        strategy: ExtractionStrategy, 
        layout_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Gera instruções específicas baseadas na análise de layout
        """
        instructions = []
        
        # Da análise técnica
        technical = layout_analysis.get("technical_analysis", {})
        if not technical.get("error"):
            column_info = technical.get("column_detection", {})
            column_count = column_info.get("column_count", 0)
            
            if column_count > 0:
                instructions.append(f"- {column_count} colunas detectadas - usar para mapeamento")
                
                if strategy.approach == "table_extraction" and column_count > 6:
                    instructions.append("- Muitas colunas: provavelmente tamanhos organizados horizontalmente")
        
        # Da análise visual
        visual = layout_analysis.get("visual_analysis", {})
        if not visual.get("error"):
            structural = visual.get("structural_elements", {})
            
            if structural.get("has_dividing_lines"):
                instructions.append("- Linhas divisórias presentes: respeitar separações")
            
            if structural.get("data_density") == "high":
                instructions.append("- Alta densidade de dados: processar cuidadosamente")
            
            data_org = visual.get("data_organization", {})
            if data_org.get("item_presentation") == "grouped_sections":
                instructions.append("- Produtos em seções agrupadas: processar por grupo")
        
        # Das instruções de extração detectadas
        extraction_instructions = layout_analysis.get("extraction_instructions", {})
        if extraction_instructions.get("special_considerations"):
            considerations = extraction_instructions["special_considerations"]
            instructions.append(f"- Consideração especial: {considerations}")
        
        return instructions