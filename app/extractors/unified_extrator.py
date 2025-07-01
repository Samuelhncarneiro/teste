# app/extractors/unified_extractor.py
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.extractors.base import BaseExtractor
from PIL import Image

logger = logging.getLogger(__name__)

class ExtractionStrategy(Enum):
    AUTO = "auto"
    TABULAR = "tabular"
    SEQUENTIAL = "sequential"
    HYBRID = "hybrid"

@dataclass
class ExtractionConfig:
    """Configuração unificada para extração"""
    strategy: ExtractionStrategy = ExtractionStrategy.AUTO
    enable_color_mapping: bool = True
    enable_barcode_generation: bool = True
    enable_supplier_detection: bool = True
    max_retries: int = 3
    timeout_seconds: int = 30
    confidence_threshold: float = 0.7

class UnifiedExtractor(BaseExtractor):
    """
    Extrator unificado que combina todas as funcionalidades
    em um pipeline mais simples e eficiente
    """
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        self.api_key = GEMINI_API_KEY
        self._initialize_components()
        
        # Métricas de performance
        self.stats = {
            "pages_processed": 0,
            "products_extracted": 0,
            "errors_recovered": 0,
            "processing_time": 0.0
        }
    
    def _initialize_components(self):
        """Inicializa apenas os componentes necessários"""
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Lazy loading dos agentes especializados
        self._context_agent = None
        self._color_mapper = None
        
    @property
    def context_agent(self):
        """Lazy loading do context agent"""
        if self._context_agent is None:
            from app.extractors.context_agent import ContextAgent
            self._context_agent = ContextAgent(self.api_key)
        return self._context_agent
    
    @property 
    def color_mapper(self):
        """Lazy loading do color mapper"""
        if self._color_mapper is None:
            from app.extractors.color_mapping_agent import ColorMappingAgent
            self._color_mapper = ColorMappingAgent()
        return self._color_mapper
    
    async def extract_document(
        self, 
        document_path: str,
        job_id: str,
        jobs_store: Dict[str, Any],
        update_progress_callback: Callable
    ) -> Dict[str, Any]:
        """Pipeline de extração simplificado"""
        
        try:
            # 1. Análise rápida do documento
            context_info = await self._quick_analysis(document_path)
            
            # 2. Determinar estratégia automaticamente
            strategy = self._determine_strategy(context_info)
            
            # 3. Extração principal
            raw_result = await self._extract_with_strategy(
                document_path, strategy, job_id, jobs_store, update_progress_callback
            )
            
            # 4. Pós-processamento unificado
            final_result = await self._post_process(raw_result, context_info)
            
            return final_result
            
        except Exception as e:
            logger.exception(f"Erro na extração: {str(e)}")
            return self._create_error_response(str(e))
    
    async def _quick_analysis(self, document_path: str) -> Dict[str, Any]:
        """Análise rápida e eficiente do documento"""
        
        # Usar apenas text extraction para análise inicial
        from app.utils.file_utils import extract_text_from_pdf
        
        text = extract_text_from_pdf(document_path)
        
        # Análise de padrões simples e eficiente
        analysis = {
            "supplier": self._detect_supplier_quick(text),
            "layout_type": self._detect_layout_quick(text),
            "product_patterns": self._detect_product_patterns_quick(text),
            "document_type": "order"
        }
        
        return analysis
    
    def _detect_supplier_quick(self, text: str) -> str:
        """Detecção rápida de fornecedor"""
        from app.data.reference_data import SUPPLIER_MAP
        
        text_upper = text.upper()
        
        # Busca direta nos fornecedores conhecidos
        for supplier in SUPPLIER_MAP.values():
            if supplier.upper() in text_upper:
                return supplier
        
        return "UNKNOWN"
    
    def _detect_layout_quick(self, text: str) -> str:
        """Detecção rápida de layout"""
        lines = text.split('\n')
        
        # Contar linhas com múltiplas colunas
        tabular_lines = sum(1 for line in lines if len(line.split()) > 5)
        
        if tabular_lines > len(lines) * 0.3:
            return "TABULAR"
        else:
            return "SEQUENTIAL"
    
    def _detect_product_patterns_quick(self, text: str) -> Dict[str, Any]:
        """Detecção rápida de padrões de produto"""
        import re
        
        # Padrões comuns
        patterns = {
            "codes": len(re.findall(r'\b[A-Z]{2,}\d{4,}\b', text)),
            "sizes": len(re.findall(r'\b(XS|S|M|L|XL|XXL|3[4-9]|4[0-9])\b', text)),
            "prices": len(re.findall(r'\b\d+[,\.]\d{2}\b', text))
        }
        
        return patterns
    
    def _determine_strategy(self, context_info: Dict[str, Any]) -> ExtractionStrategy:
        """Determina estratégia baseada na análise"""
        
        if self.config.strategy != ExtractionStrategy.AUTO:
            return self.config.strategy
        
        layout = context_info.get("layout_type", "SEQUENTIAL")
        
        if layout == "TABULAR":
            return ExtractionStrategy.TABULAR
        else:
            return ExtractionStrategy.SEQUENTIAL
    
    async def _extract_with_strategy(
        self,
        document_path: str,
        strategy: ExtractionStrategy,
        job_id: str,
        jobs_store: Dict[str, Any],
        update_progress_callback: Callable
    ) -> Dict[str, Any]:
        """Extração usando estratégia determinada"""
        
        from app.utils.file_utils import convert_pdf_to_images
        from app.config import CONVERTED_DIR
        
        # Converter para imagens
        if document_path.lower().endswith('.pdf'):
            image_paths = convert_pdf_to_images(document_path, CONVERTED_DIR)
        else:
            image_paths = [document_path]
        
        # Processar páginas
        combined_result = {"products": [], "order_info": {}}
        
        for i, img_path in enumerate(image_paths):
            # Atualizar progresso
            progress = 20 + (i / len(image_paths)) * 60
            jobs_store[job_id]["model_results"]["gemini"]["progress"] = progress
            
            # Processar página
            page_result = await self._process_page_unified(img_path, strategy)
            
            # Mesclar resultados
            if "products" in page_result:
                combined_result["products"].extend(page_result["products"])
            
            if "order_info" in page_result:
                combined_result["order_info"].update(page_result["order_info"])
        
        return combined_result
    
    async def _process_page_unified(
        self, 
        image_path: str, 
        strategy: ExtractionStrategy
    ) -> Dict[str, Any]:
        """Processamento unificado de página"""
        
        # Preparar prompt baseado na estratégia
        prompt = self._create_strategy_prompt(strategy)
        
        # Processar com Gemini
        image = Image.open(image_path)
        response = self.model.generate_content([prompt, image])
        
        # Extrair e validar JSON
        try:
            import json
            import re
            
            # Extrair JSON da resposta
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = json.loads(response.text)
            
            # Validação básica
            if not isinstance(result, dict):
                raise ValueError("Resposta não é um objeto JSON válido")
            
            if "products" not in result:
                result["products"] = []
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao processar página: {str(e)}")
            return {"products": [], "error": str(e)}
    
    def _create_strategy_prompt(self, strategy: ExtractionStrategy) -> str:
        """Cria prompt otimizado para a estratégia"""
        
        base_prompt = """
        Extraia informações de produtos deste documento comercial.
        
        Para cada produto, extraia:
        - Nome e código do material
        - Categoria (em português)
        - Cores com códigos
        - Tamanhos e quantidades
        - Preços (unitário e de venda)
        
        Retorne em formato JSON:
        {
          "products": [...],
          "order_info": {...}
        }
        """
        
        if strategy == ExtractionStrategy.TABULAR:
            base_prompt += """
            
            ESTRATÉGIA TABULAR:
            - Processe linha por linha
            - Mapeie colunas por posição
            - Células vazias = tamanho indisponível
            """
        
        elif strategy == ExtractionStrategy.SEQUENTIAL:
            base_prompt += """
            
            ESTRATÉGIA SEQUENCIAL:
            - Cada linha é um item completo
            - Dados organizados horizontalmente
            - Processe sequencialmente
            """
        
        return base_prompt
    
    async def _post_process(
        self, 
        raw_result: Dict[str, Any], 
        context_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Pós-processamento unificado"""
        
        # 1. Mapeamento de cores (se habilitado)
        if self.config.enable_color_mapping and raw_result.get("products"):
            try:
                raw_result["products"] = self.color_mapper.map_product_colors(
                    raw_result["products"]
                )
            except Exception as e:
                logger.warning(f"Erro no mapeamento de cores: {str(e)}")
        
        # 2. Processamento de fornecedor
        if self.config.enable_supplier_detection:
            raw_result = self._process_supplier_info(raw_result, context_info)
        
        # 3. Geração de códigos de barras (se habilitado)
        if self.config.enable_barcode_generation:
            try:
                from app.utils.barcode_generator import add_barcodes_to_extraction_result
                raw_result = add_barcodes_to_extraction_result(raw_result)
            except Exception as e:
                logger.warning(f"Erro na geração de códigos de barras: {str(e)}")
        
        # 4. Validação final e limpeza
        raw_result = self._final_validation(raw_result)
        
        return raw_result
    
    def _process_supplier_info(
        self, 
        result: Dict[str, Any], 
        context_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Processa informações de fornecedor"""
        
        from app.utils.supplier_assignment import determine_best_supplier
        
        supplier_name, supplier_code, markup = determine_best_supplier(context_info)
        
        # Atualizar order_info
        if "order_info" not in result:
            result["order_info"] = {}
        
        result["order_info"]["supplier"] = supplier_name
        result["order_info"]["supplier_code"] = supplier_code
        result["order_info"]["markup"] = markup
        
        return result
    
    def _final_validation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validação final dos dados"""
        
        if "products" not in result:
            result["products"] = []
        
        if "order_info" not in result:
            result["order_info"] = {}
        
        # Remover produtos sem dados essenciais
        valid_products = []
        for product in result["products"]:
            if (product.get("name") and 
                product.get("material_code") and 
                product.get("colors")):
                valid_products.append(product)
        
        result["products"] = valid_products
        
        # Estatísticas
        self.stats["products_extracted"] = len(valid_products)
        
        return result
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Cria resposta de erro padronizada"""
        return {
            "error": error_message,
            "products": [],
            "order_info": {},
            "stats": self.stats
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de processamento"""
        return self.stats.copy()