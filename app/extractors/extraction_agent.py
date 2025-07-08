# app/extractors/extraction_agent.py
import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
import google.generativeai as genai
from PIL import Image

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.utils.file_utils import optimize_image
from app.data.reference_data import CATEGORIES
from app.extractors.size_detection_agent import SizeDetectionAgent

logger = logging.getLogger(__name__)

class ExtractionAgent:
    def __init__(self, api_key: str = GEMINI_API_KEY):

        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.size_detector = SizeDetectionAgent()

    async def process_page(
        self, 
        image_path: str, 
        context: str,
        page_number: int,
        total_pages: int,
        previous_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        is_first_page = (page_number == 1)
        
        try:
            # Otimizar imagem para melhor processamento
            optimized_path = optimize_image(image_path, os.path.dirname(image_path))
            
            # Carregar a imagem
            image = Image.open(optimized_path)
            
            # Preparar o prompt JSON para o exemplo de resposta esperada
            json_template = self._get_json_template()
            
            # Preparar prompt adequado para a página
            if is_first_page:
                prompt = self._create_first_page_prompt(
                    context, page_number, total_pages, json_template
                )
            else:
                # Para páginas adicionais, informar sobre produtos já encontrados
                previous_products_count = len(previous_result.get("products", [])) if previous_result else 0
                prompt = self._create_additional_page_prompt(
                    context, page_number, total_pages, previous_products_count, json_template
                )
            
            response = self.model.generate_content([prompt, image])
            response_text = response.text
            
            # Extrair e processar o JSON da resposta
            try:
                result = self._extract_and_clean_json(response_text, page_number)
                
                # Registrar resultados básicos
                products_count = len(result.get("products", []))
                logger.info(f"Página {page_number}: Extraídos {products_count} produtos")
                
                for i, product in enumerate(result.get("products", [])[:3]):  # Log apenas primeiros 3
                    colors_count = len(product.get("colors", []))
                    logger.debug(f"Produto {i+1}: {product.get('name', 'Sem nome')} - {colors_count} cores")
                
                return result
                
            except Exception as e:
                logger.error(f"Erro ao processar JSON da página {page_number}: {str(e)}")
                
                # Tentar recuperar o máximo de informação possível
                fallback_result = self._attempt_json_recovery(response_text, page_number)
                
                if fallback_result and "products" in fallback_result:
                    logger.info(f"Recuperação parcial: {len(fallback_result['products'])} produtos")
                    return fallback_result
                
                # Se não foi possível recuperar, retornar erro
                return {"error": str(e), "products": [], "raw_text": response_text[:1000]}
                
        except Exception as e:
            logger.error(f"Erro ao processar página {page_number}: {str(e)}")
            return {"error": str(e), "products": []}
    
    def _extract_headers_from_context(self, context: str) -> List[str]:
        headers_match = re.search(r'Cabeçalhos Detectados: (.+)', context)
        if headers_match:
            headers_text = headers_match.group(1)
            headers = re.findall(r'"([^"]+)"', headers_text)
            return headers
        return []

    def _get_material_code_instructions(self, headers: List[str]) -> str:
        headers_upper = [h.upper() for h in headers]
        
        if "MODEL" in headers_upper:
            return "- Código do material: EXTRAIR da coluna 'Model'"
        elif "REFERENCE" in headers_upper:
            return "- Código do material: EXTRAIR da coluna 'Reference'"
        elif "ARTICLE" in headers_upper:
            return "- Código do material: EXTRAIR da coluna 'Article'"
        elif "SKU" in headers_upper:
            return "- Código do material: EXTRAIR da coluna 'SKU'"
        elif "ITEM" in headers_upper:
            return "- Código do material: EXTRAIR da coluna 'Item'"
        else:
            return """- Código do material: IDENTIFICAR por padrões alfanuméricos únicos no texto
            - Procurar por códigos como: CF5015E0624, AB123456, 50469055, T3216
            - Geralmente próximo ao nome do produto ou no início da linha"""

    def _create_individual_size_prompt(self, context: str, page_number: int, total_pages: int) -> str:

        headers_detected = self._extract_headers_from_context(context)
        material_code_instructions = self._get_material_code_instructions(headers_detected)
        
        return f"""
        # INSTRUÇÕES PARA EXTRAÇÃO GRANULAR POR TAMANHO
        
        Você é um especialista em extrair dados de produtos de documentos comerciais.
        Esta é a página {page_number} de {total_pages}.
        
        {context}
        
        ## REGRA FUNDAMENTAL: GRANULARIDADE MÁXIMA
        
        **CADA COMBINAÇÃO CÓDIGO+COR+TAMANHO = UM PRODUTO SEPARADO**
        
        **EXEMPLO DO QUE QUERO:**
        Se vir na tabela:
        ```
        CF5271MA96E  Preto(M9799)   S  M  L  →  1  1  1
        CF5271MA96E  Bege(M9990)   XS  S  M  →  1  1  1
        ```
        
        **DEVE EXTRAIR 6 PRODUTOS SEPARADOS:**
        ```json
        [
        {{"name": "Malha Fechada M/L", "material_code": "CF5271MA96E", "color_code": "010", "color_name": "Preto", "size": "S", "quantity": 1}},
        {{"name": "Malha Fechada M/L", "material_code": "CF5271MA96E", "color_code": "010", "color_name": "Preto", "size": "M", "quantity": 1}},
        {{"name": "Malha Fechada M/L", "material_code": "CF5271MA96E", "color_code": "010", "color_name": "Preto", "size": "L", "quantity": 1}},
        {{"name": "Malha Fechada M/L", "material_code": "CF5271MA96E", "color_code": "012", "color_name": "Bege", "size": "XS", "quantity": 1}},
        {{"name": "Malha Fechada M/L", "material_code": "CF5271MA96E", "color_code": "012", "color_name": "Bege", "size": "S", "quantity": 1}},
        {{"name": "Malha Fechada M/L", "material_code": "CF5271MA96E", "color_code": "012", "color_name": "Bege", "size": "M", "quantity": 1}}
        ]
        ```
        
        ## ALGORITMO DE EXTRAÇÃO:
        
        1. **Identificar produto base** (código + nome)
        2. **Para cada cor desse produto:**
        3. **Para cada tamanho com quantidade > 0:**
            4. **Criar produto individual separado**
        
        ## REGRAS CRÍTICAS:
        
        - **NUNCA** agrupar tamanhos no mesmo produto
        - **NUNCA** agrupar cores no mesmo produto  
        - **SEMPRE** criar produto separado para cada combinação única
        - **SÓ** incluir tamanhos que têm quantidade explícita > 0
        - **USAR** mapeamento posicional rigoroso (tamanho = posição da quantidade)
        
        {material_code_instructions}
        
        ## FORMATO DE RESPOSTA OBRIGATÓRIO:
        
        ```json
        {{
        "products": [
            {{
            "name": "Nome do produto",
            "material_code": "Código do material",
            "category": "Categoria em português - usar: {CATEGORIES}",
            "model": "Modelo/descrição",
            "composition": null,
            "color_code": "Código da cor",
            "color_name": "Nome da cor",
            "size": "Tamanho específico",
            "quantity": 1,
            "unit_price": 0.0,
            "sales_price": 0.0
            }}
        ]
        }}
        ```
        
        ## VERIFICAÇÕES OBRIGATÓRIAS:
        
        ✅ Cada objeto no array "products" tem APENAS 1 tamanho
        ✅ Cada objeto no array "products" tem APENAS 1 cor
        ✅ Tamanhos sem quantidade NÃO são incluídos
        ✅ Mapeamento posicional correto entre tamanho e quantidade
        
        EXTRAIA AGORA seguindo RIGOROSAMENTE estas regras.
        """
        
    def _create_first_page_prompt(
        self, context: str, page_number: int, total_pages: int, json_template: str
    ) -> str:

        headers_detected = self._extract_headers_from_context(context)
        
        material_code_instructions = self._get_material_code_instructions(headers_detected)
    
        return f"""
        # INSTRUÇÕES PARA EXTRAÇÃO DE PRODUTOS
        
        Você é um especialista em extrair dados de produtos de documentos comerciais.
        Esta é a página {page_number} de {total_pages}.
        
        {context}
        
        ## REGRAS CRÍTICAS PARA TAMANHOS:
    
        **MAPEAMENTO POSICIONAL**: Quando vir tamanhos em uma linha e quantidades em outra:
        
        EXEMPLO:
        ```
        XS   S    M    L    XL   XXL
            1    1    1
        ```
        ➜ INTERPRETAR: S=1, M=1, L=1 (XS, XL, XXL = SEM quantidade = NÃO incluir)
        
        **ALGORITMO**:
        1. Identificar linha com tamanhos
        2. Localizar linha com quantidades (normalmente seguinte)
        3. Mapear por posição: quantidade corresponde ao tamanho na mesma coluna
        4. Incluir APENAS tamanhos que têm quantidade > 0
        5. Ignorar células vazias ou quantidades zero

        ## PADRÕES COMUNS DE CÓDIGOS DE PRODUTOS:
            - Alfanuméricos: CF5015E0624, AB123456, T3216
            - Numéricos longos: 50469055, 23411201
            - Híbridos: MA82O, MS55N, T054A

        ## Tarefa de Extração
        Analise esta página e extraia todas as informações de produtos presentes, seguindo todas as orientações de layout e estrutura descritas acima.
        
        Para cada produto, extraia:
        - Nome do produto: Em português, se possível faz tradução
        - **Código do material**: Identificar por padrões acima OU campo específico detectado        
        - Categoria do produto - DEVE ser traduzido para PORTUGUÊS, usando APENAS uma das seguintes categorias: {CATEGORIES}
        - Modelo
        - Composição (se disponível) - Deve ser traduzido para Português - Portugal
        - Para CADA COR do produto:
          * Código da cor
          * Nome da cor (se disponível)
          * Tamanhos disponíveis e suas quantidades
          * Preço unitário
          * Preço de venda (se disponível)
          * Subtotal para esta cor

        ## Regras Críticas:
        1. Extraia APENAS o que está visível nesta página específica
        2. Inclua APENAS tamanhos com quantidades explicitamente indicadas
        3. NÃO inclua tamanhos com células vazias ou quantidade zero
        4. Utilize NULL para campos não encontrados, mas mantenha a estrutura JSON
        5. Preste atenção especial a como as cores são organizadas conforme as instruções
        6. NÃO invente dados ou adicione produtos que não estão claramente na imagem

        ## Formato de Resposta
        Retorne os dados extraídos em formato JSON estrito:
        
        {json_template}
        """
    
    def _create_additional_page_prompt(
        self, context: str, page_number: int, total_pages: int, previous_products_count: int, json_template: str
    ) -> str:

        headers_detected = self._extract_headers_from_context(context)
        
        material_code_instructions = self._get_material_code_instructions(headers_detected)
    
        return f"""
        # INSTRUÇÕES PARA EXTRAÇÃO DE PRODUTOS
        
        Você é um especialista em extrair dados de produtos de documentos comerciais.
        Esta é a página {page_number} de {total_pages}.
        
        {context}
        
        ## REGRAS CRÍTICAS PARA TAMANHOS:
    
        **MAPEAMENTO POSICIONAL**: Quando vir tamanhos em uma linha e quantidades em outra:
        
        EXEMPLO:
        ```
        XS   S    M    L    XL   XXL
            1    1    1
        ```
        ➜ INTERPRETAR: S=1, M=1, L=1 (XS, XL, XXL = SEM quantidade = NÃO incluir)
        
        **ALGORITMO**:
        1. Identificar linha com tamanhos
        2. Localizar linha com quantidades (normalmente seguinte)
        3. Mapear por posição: quantidade corresponde ao tamanho na mesma coluna
        4. Incluir APENAS tamanhos que têm quantidade > 0
        5. Ignorar células vazias ou quantidades zero

        ## PADRÕES COMUNS DE CÓDIGOS DE PRODUTOS:
            - Alfanuméricos: CF5015E0624, AB123456, T3216
            - Numéricos longos: 50469055, 23411201
            - Híbridos: MA82O, MS55N, T054A

        ## Tarefa de Extração
        Analise esta página e extraia todas as informações de produtos presentes, seguindo todas as orientações de layout e estrutura descritas acima.
        
        Para cada produto, extraia:
        - Nome do produto: Em português, se possível faz tradução
        - **Código do material**: Identificar por padrões acima OU campo específico detectado        
        - Categoria do produto - DEVE ser traduzido para PORTUGUÊS, usando APENAS uma das seguintes categorias: {CATEGORIES}
        - Modelo
        - Composição (se disponível) - Deve ser traduzido para Português - Portugal
        - Para CADA COR do produto:
          * Código da cor
          * Nome da cor (se disponível)
          * Tamanhos disponíveis e suas quantidades
          * Preço unitário
          * Preço de venda (se disponível)
          * Subtotal para esta cor

        ## Regras Críticas:
        1. Extraia APENAS o que está visível nesta página específica
        2. Inclua APENAS tamanhos com quantidades explicitamente indicadas
        3. NÃO inclua tamanhos com células vazias ou quantidade zero
        4. Utilize NULL para campos não encontrados, mas mantenha a estrutura JSON
        5. Preste atenção especial a como as cores são organizadas conforme as instruções
        6. NÃO invente dados ou adicione produtos que não estão claramente na imagem

        ## Formato de Resposta
        Retorne os dados extraídos em formato JSON estrito:
        
        {json_template}
        """
    
    def _get_json_template(self) -> str:
        return '''
        {
          "products": [
            {
              "name": "Nome do produto",
              "material_code": "Código identificador único (OBRIGATÓRIO)",
              "category": "Categoria",
              "model": "Modelo",
              "composition": "100% algodão",
              "colors": [
                {
                  "color_code": "807",
                  "color_name": "Azul",
                  "sizes": [
                    {"size": "S", "quantity": 1},
                    {"size": "M", "quantity": 2}
                  ],
                  "unit_price": 79.00,
                  "sales_price": 119.00,
                  "subtotal": 474.00
                }
              ],
              "total_price": 474.00
            }
          ],
          "order_info": {
            "total_pieces": 122,
            "total_value": 9983.00
          }
        }
        '''
    
    def _extract_and_clean_json(self, response_text: str, page_number: int) -> Dict[str, Any]:

        # Verificar se tem bloco de código JSON
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, response_text)
        
        if matches:
            # Usar o primeiro bloco JSON encontrado
            json_str = matches[0]
            logger.info(f"JSON encontrado em bloco de código para página {page_number}")
        else:
            # Tentar encontrar objeto JSON na string
            json_pattern = r'(\{[\s\S]*\})'
            matches = re.findall(json_pattern, response_text)
            
            if matches:
                # Buscar o JSON mais completo (maior)
                json_candidates = []
                for potential_json in matches:
                    try:
                        parsed = json.loads(potential_json)
                        if isinstance(parsed, dict) and "products" in parsed:
                            json_candidates.append((len(potential_json), potential_json))
                    except:
                        continue
                
                if json_candidates:
                    # Ordenar por tamanho e pegar o maior
                    json_candidates.sort(reverse=True)
                    json_str = json_candidates[0][1]
                    logger.info(f"JSON encontrado no texto para página {page_number}")
                else:
                    raise ValueError("Nenhum JSON válido com estrutura de produtos encontrado")
            else:
                # Tentar interpretar a string inteira
                try:
                    json.loads(response_text)
                    json_str = response_text
                    logger.info(f"Resposta completa interpretada como JSON para página {page_number}")
                except:
                    raise ValueError("Nenhum JSON válido encontrado na resposta")
        
        # Processar o JSON encontrado
        try:
            result = json.loads(json_str)
            
            # Validar e limpar a estrutura
            if not isinstance(result, dict):
                raise ValueError("O JSON não é um objeto como esperado")
            
            # Garantir que temos produtos
            if "products" not in result or not isinstance(result["products"], list):
                result["products"] = []
            
            # Garantir que temos order_info
            if "order_info" not in result or not isinstance(result["order_info"], dict):
                result["order_info"] = {}
            
            # Limpar os produtos
            clean_products = []
            for product in result["products"]:
                # Verificar se é um produto válido
                if not isinstance(product, dict):
                    continue
                
                # Garantir que campos críticos existem
                for field in ["name", "material_code", "colors"]:
                    if field not in product:
                        product[field] = None if field != "colors" else []
                
                if product.get("name") is None:
                    product["name"] = ""
                if product.get("material_code") is None:
                    product["material_code"] = ""

                # Limpar as cores
                if isinstance(product["colors"], list):
                    clean_colors = []
                    for color in product["colors"]:
                        # Verificar se é uma cor válida
                        if not isinstance(color, dict):
                            continue
                        
                        # Garantir que campos críticos existem
                        for field in ["color_code", "sizes"]:
                            if field not in color:
                                color[field] = None if field != "sizes" else []
                        
                        # SEÇÃO CORRIGIDA: Limpar os tamanhos
                        if isinstance(color["sizes"], list):
                            # Primeiro, extrair tamanhos como antes
                            preliminary_sizes = []
                            for size in color["sizes"]:
                                if isinstance(size, dict) and "size" in size and "quantity" in size:
                                    # Garantir que quantity é um número positivo
                                    try:
                                        quantity = float(size["quantity"])
                                        if quantity <= 0:
                                            continue
                                        size["quantity"] = int(quantity) if quantity.is_integer() else quantity
                                        preliminary_sizes.append(size)
                                    except (ValueError, TypeError):
                                        continue
                            
                            clean_sizes = self.size_detector.normalize_size_extraction(
                                preliminary_sizes, 
                                product.get("category", "")
                            )
                            
                            # Atualizar tamanhos limpos
                            color["sizes"] = clean_sizes
                            
                            # Ignorar cores sem tamanhos
                            if clean_sizes:
                                clean_colors.append(color)
                        else:
                            # Ignorar cores sem tamanhos válidos
                            continue
                    
                    # Atualizar cores limpas
                    product["colors"] = clean_colors
                    
                    # Ignorar produtos sem cores
                    if clean_colors:
                        # Garantir que unit_price e subtotal são números
                        for color in product["colors"]:
                            for field in ["unit_price", "sales_price", "subtotal"]:
                                if field in color and color[field] is not None:
                                    try:
                                        color[field] = float(color[field])
                                    except (ValueError, TypeError):
                                        color[field] = None
                        
                        # Calcular total_price se não existir
                        if "total_price" not in product or product["total_price"] is None:
                            subtotals = [color.get("subtotal", 0) for color in product["colors"] 
                                        if color.get("subtotal") is not None]
                            product["total_price"] = sum(subtotals) if subtotals else None
                        else:
                            # Garantir que é um número
                            try:
                                product["total_price"] = float(product["total_price"])
                            except (ValueError, TypeError):
                                product["total_price"] = None
                        
                        clean_products.append(product)
                    else:
                        # Ignorar produtos sem cores válidas
                        continue
            
            # Atualizar produtos limpos
            result["products"] = clean_products
            
            # Limpar order_info
            for field in ["total_pieces", "total_value"]:
                if field in result["order_info"] and result["order_info"][field] is not None:
                    try:
                        value = result["order_info"][field]
                        result["order_info"][field] = int(value) if field == "total_pieces" else float(value)
                    except (ValueError, TypeError):
                        result["order_info"][field] = None
            
            logger.info(f"JSON processado com sucesso: {len(clean_products)} produtos válidos")
            return result
            
        except Exception as e:
            logger.error(f"Erro ao processar JSON para página {page_number}: {str(e)}")
            raise ValueError(f"Erro ao processar JSON: {str(e)}")
    
    def _attempt_json_recovery(self, response_text: str, page_number: int) -> Optional[Dict[str, Any]]:
        """
        Tenta recuperar dados parciais de uma resposta inválida
        
        Args:
            response_text: Texto de resposta da API
            page_number: Número da página
            
        Returns:
            Optional[Dict]: Dados parcialmente recuperados ou None
        """
        try:
            # Buscar qualquer estrutura que se pareça com um produto
            product_pattern = r'{"name":[^{]*?,"colors":[^]]*?]}'
            product_matches = re.findall(product_pattern, response_text)
            
            products = []
            for product_text in product_matches:
                try:
                    # Tentar consertar o JSON do produto
                    fixed_text = product_text.replace("'", '"')
                    product = json.loads(f"{{{fixed_text}}}")
                    products.append(product)
                except:
                    continue
            
            if products:
                logger.info(f"Recuperados {len(products)} produtos parciais da página {page_number}")
                return {"products": products, "order_info": {}, "partially_recovered": True}
            
            return None
            
        except Exception as e:
            logger.warning(f"Falha na tentativa de recuperação: {str(e)}")
            return None