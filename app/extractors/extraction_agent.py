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
from app.utils.size_detection import SizeDetectionAgent

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
    
    async def extract_from_page(self, image_path: str, context: str, page_number: int, 
                               total_pages: int, previous_results: List[Dict]) -> Dict[str, Any]:
        """
        Versão melhorada com prompt focado em tamanhos
        """
        
        # PROMPT MELHORADO - A chave para resolver tudo
        enhanced_context = self._add_size_focused_instructions(context)
        
        try:
            image = Image.open(image_path)
            response = self.model.generate_content([enhanced_context, image])
            
            # Parse da resposta
            result = self._parse_extraction_response(response.text)
            
            # USAR SEU SIZE_DETECTION_AGENT para validar/melhorar
            if result.get('products'):
                result['products'] = self._improve_sizes_with_your_agent(result['products'])
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na extração: {e}")
            return {"error": str(e), "products": []}
    
    def _improve_sizes_with_your_agent(self, products: List[Dict]) -> List[Dict]:
        """
        Usa seu SizeDetectionAgent para melhorar os tamanhos extraídos
        """
        improved_products = []
        
        for product in products:
            improved_product = product.copy()
            product_code = product.get('material_code', '')
            category = product.get('category', '')
            
            for color in improved_product.get('colors', []):
                original_sizes = color.get('sizes', [])
                
                if original_sizes:
                    # USAR SEU AGENT para validar
                    validated_sizes = self.size_detector.normalize_size_extraction(
                        original_sizes, 
                        category=category
                    )
                    
                    if validated_sizes and len(validated_sizes) > 0:
                        color['sizes'] = validated_sizes
                        logger.info(f"✅ Tamanhos validados para {product_code}: {len(validated_sizes)} tamanhos")
                        
                        # Recalcular subtotal se necessário
                        if 'unit_price' in color:
                            total_qty = sum(s['quantity'] for s in validated_sizes)
                            color['subtotal'] = color['unit_price'] * total_qty
                    
                    else:
                        # Se validação rejeitar tudo, manter originais mas avisar
                        logger.warning(f"⚠️ SizeDetectionAgent rejeitou tamanhos para {product_code}")
                        color['sizes'] = original_sizes
                
                else:
                    # Sem tamanhos originais - problema de extração
                    logger.warning(f"❌ Nenhum tamanho extraído para {product_code}")
                    
                    # Fallback: criar tamanho básico baseado na categoria
                    fallback_size = self._create_fallback_size(category)
                    color['sizes'] = [fallback_size]
            
            improved_products.append(improved_product)
        
        return improved_products
    
    def _create_fallback_size(self, category: str) -> Dict[str, Any]:

        category_upper = category.upper() if category else ''
        
        # Tamanhos padrão por categoria
        if category_upper in ['MALHAS', 'T-SHIRTS', 'POLOS']:
            return {"size": "M", "quantity": 1}
        elif category_upper in ['VESTIDOS', 'SAIAS', 'CASACOS', 'BLUSAS']:
            return {"size": "40", "quantity": 1}
        elif category_upper in ['CALÇAS', 'JEANS']:
            return {"size": "30", "quantity": 1}
        else:
            return {"size": "M", "quantity": 1}
        
    def _add_size_focused_instructions(self, base_context: str) -> str:

        size_instructions = """

        🎯 INSTRUÇÕES ULTRA-ESPECÍFICAS PARA TAMANHOS:

        ## PASSO 1: ANÁLISE ESTRUTURAL DA TABELA
        1. Identifique a PRIMEIRA LINHA (cabeçalhos da tabela)
        2. Localize colunas que são TAMANHOS:
        - Tamanhos por letra: XS, S, M, L, XL, XXL
        - Tamanhos numéricos: 38, 40, 42, 44, 46, 48
        - Qualquer número de 2 dígitos nas colunas
        3. Memorize a POSIÇÃO de cada coluna de tamanho

        ## PASSO 2: MAPEAMENTO PRODUTO-TAMANHO
        Para cada linha de produto:
        1. Identifique o CÓDIGO do produto (primeira coluna)
        2. Identifique a COR do produto
        3. Para CADA coluna de tamanho:
        - Leia o VALOR na intersecção linha-produto × coluna-tamanho
        - CÉLULA VAZIA ou "0" = NÃO incluir esse tamanho
        - NÚMERO > 0 = incluir com quantidade EXATA

        ## PASSO 3: REGRAS ABSOLUTAS

        ❌ **JAMAIS FAÇA:**
        - Assumir "UN" ou "UNICO" quando vê colunas de tamanhos
        - Colocar quantity: 1 para todos sem verificar valores reais
        - Misturar informações de produtos diferentes

        ✅ **SEMPRE FAÇA:**
        - Mapear cada intersecção individualmente
        - Incluir apenas tamanhos com quantidade > 0
        - Manter correspondência exata posição→tamanho→quantidade

        ## EXEMPLO PRÁTICO:

        Se vê esta estrutura:
        ```
        Model        | Color  | XS | S | M | L | XL | Qty | Price
        CF5271MA96E  | M9799  | 1  | 1 | 1 |   |    | 3   | 71.00
        CF5245MS019  | 94028  |    | 1 | 1 | 1 |    | 3   | 67.00
        ```

        EXTRAIR como:
        ```json
        {
        "products": [
            {
            "material_code": "CF5271MA96E",
            "colors": [{
                "color_code": "M9799",
                "sizes": [
                {"size": "XS", "quantity": 1},
                {"size": "S", "quantity": 1},
                {"size": "M", "quantity": 1}
                ]
            }]
            },
            {
            "material_code": "CF5245MS019", 
            "colors": [{
                "color_code": "94028",
                "sizes": [
                {"size": "S", "quantity": 1},
                {"size": "M", "quantity": 1},
                {"size": "L", "quantity": 1}
                ]
            }]
            }
        ]
        }
        ```

        **CRÍTICO**: Se não conseguir identificar colunas de tamanhos claramente, 
        prefira retornar tamanhos vazios a assumir "UN".

        Continue com a extração seguindo rigorosamente estas regras.
        """
        
        return base_context + size_instructions
    
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
    
        ### PROBLEMA 1: TAMANHOS INCORRETOS ("UN" em vez dos tamanhos reais)
        **SOLUÇÃO - ALGORITMO DE MAPEAMENTO POSICIONAL**:
        
        1. **IDENTIFICAR ESTRUTURA DE TAMANHOS** na tabela:
           - Procurar colunas com números: 34,36,38,40,42,44,46,48 (roupas)
           - Procurar colunas com letras: XS,S,M,L,XL,XXL (t-shirts/malhas)
           - Procurar colunas com tamanhos de calças: 24,25,26,27,28,29,30,31,32
        
        2. **MAPEAR QUANTIDADES POR POSIÇÃO**:
           ```
           Exemplo visual:
           CABEÇALHOS:    | 38 | 40 | 42 | 44 | 46 | 48 |
           QUANTIDADES:   | 1  |    | 2  |    | 1  |    |
           ```
           ➜ INTERPRETAR: 38→1, 42→2, 46→1 (ignorar colunas vazias)
           
           **NUNCA usar tamanho "UN" para roupas!**
        
        3. **VALIDAR TAMANHOS POR CATEGORIA**:
           - VESTIDOS/BLUSAS/CASACOS: 34-48 ou XS-XL
           - CALÇAS/JEANS: 24-32 ou W28-W36  
           - MALHAS/T-SHIRTS: XS-XXL
           - "UN" só para acessórios especiais (cintos, carteiras)
        
        ### PROBLEMA 2: CORES EM FALTA (NULL em vez das cores visíveis)
        **SOLUÇÃO - EXTRAIR SEMPRE CÓDIGO E NOME**:
        
        1. **LOCALIZAR INFORMAÇÃO DE COR**:
           - Coluna "Color" ou "Cor"
           - Junto ao código do produto
           - Em linha separada
        
        2. **EXTRAIR AMBOS CÓDIGO E NOME**:
           - "X0707 Asparago" → color_code="X0707", color_name="Asparago"
           - "22222 Nero" → color_code="22222", color_name="Nero"
           - "94028 Blu marino" → color_code="94028", color_name="Blu marino"
           - Se só há código, investigar nome na linha/contexto
           - Se só há nome, investigar código na linha/contexto
        
        3. **PADRÕES DE CÓDIGOS COMUNS**:
           - Alfanuméricos: X0707, M9990, C3831, V9414
           - Numéricos: 22222, 94028, 02056, 03243
           - **NUNCA deixar color_name como NULL se há informação visível**
        
        ### PROBLEMA 3: AGRUPAMENTO INCORRETO
        **SOLUÇÃO - AGRUPAR VARIANTES POR COR**:
        
        Se vir códigos como CF5271MA96E.1, CF5271MA96E.2:
        ➜ AGRUPAR em UM produto CF5271MA96E com múltiplas cores
        
        ##  ALGORITMO PASSO-A-PASSO:
        
        ### PASSO 1: IDENTIFICAR ESTRUTURA
        1. Localizar tabela principal de produtos
        2. Identificar colunas de tamanhos (números ou letras)
        3. Identificar onde estão as cores (código + nome)
        4. Identificar onde estão as quantidades
        
        ### PASSO 2: EXTRAIR CADA PRODUTO
        Para cada linha/bloco de produto:
        1. **Código**: CF5015E0624, CF5271MA96E, etc.
        2. **Nome**: Traduzir para português se necessário
        3. **Cores**: Extrair código E nome da cor
        4. **Tamanhos**: Mapear posicionalmente com quantidades
        5. **Preços**: Extrair preços visíveis
        
        ### PASSO 3: VALIDAR E AGRUPAR
        1. Verificar tamanhos realistas
        2. Verificar cores completas
        3. Agrupar produtos com mesmo código base
        4. Incluir apenas tamanhos com quantidade > 0
        
        ## EXEMPLOS PRÁTICOS:
        
        ### Exemplo 1: Linha Simples
        ```
        CF5015E0624 | X0707 Asparago | 40 | 42 |    | 44 |
                    |                | 1  | 1  |    |    |
        ```
        ➜ RESULTADO: sizes: [{{"size":"40","quantity":1}}, {{"size":"42","quantity":1}}]
        
        ### Exemplo 2: Múltiplas Cores
        ```
        CF5271MA96E | M9990 Bege  | XS:1 | S:1 | M:1 | L:3 |
        CF5271MA96E | 22222 Preto | XS:1 | S:1 | M:1 |     |
        ```
        ➜ AGRUPAR em UM produto com 2 cores

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
        7. **NUNCA usar tamanho "UN"** para roupas (vestidos, blusas, calças, etc.)
        8. **SEMPRE extrair código E nome** da cor se visível
        9. **MAPEAR tamanhos por posição** nas colunas
        10. **SÓ incluir tamanhos com quantidade > 0**
        11. **AGRUPAR produtos** com mesmo código base
        12. **TRADUZIR nomes** para português

        ## Tarefa
        Analise a imagem seguindo RIGOROSAMENTE estas instruções e extraia todos os produtos.

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
    
        ### PROBLEMA 1: TAMANHOS INCORRETOS ("UN" em vez dos tamanhos reais)
        **SOLUÇÃO - ALGORITMO DE MAPEAMENTO POSICIONAL**:
        
        1. **IDENTIFICAR ESTRUTURA DE TAMANHOS** na tabela:
           - Procurar colunas com números: 34,36,38,40,42,44,46,48 (roupas)
           - Procurar colunas com letras: XS,S,M,L,XL,XXL (t-shirts/malhas)
           - Procurar colunas com tamanhos de calças: 24,25,26,27,28,29,30,31,32
        
        2. **MAPEAR QUANTIDADES POR POSIÇÃO**:
           ```
           Exemplo visual:
           CABEÇALHOS:    | 38 | 40 | 42 | 44 | 46 | 48 |
           QUANTIDADES:   | 1  |    | 2  |    | 1  |    |
           ```
           ➜ INTERPRETAR: 38→1, 42→2, 46→1 (ignorar colunas vazias)
           
           **NUNCA usar tamanho "UN" para roupas!**
        
        3. **VALIDAR TAMANHOS POR CATEGORIA**:
           - VESTIDOS/BLUSAS/CASACOS: 34-48 ou XS-XL
           - CALÇAS/JEANS: 24-32 ou W28-W36  
           - MALHAS/T-SHIRTS: XS-XXL
           - "UN" só para acessórios especiais (cintos, carteiras)
        
        ### PROBLEMA 2: CORES EM FALTA (NULL em vez das cores visíveis)
        **SOLUÇÃO - EXTRAIR SEMPRE CÓDIGO E NOME**:
        
        1. **LOCALIZAR INFORMAÇÃO DE COR**:
           - Coluna "Color" ou "Cor"
           - Junto ao código do produto
           - Em linha separada
        
        2. **EXTRAIR AMBOS CÓDIGO E NOME**:
           - "X0707 Asparago" → color_code="X0707", color_name="Asparago"
           - "22222 Nero" → color_code="22222", color_name="Nero"
           - "94028 Blu marino" → color_code="94028", color_name="Blu marino"
           - Se só há código, investigar nome na linha/contexto
           - Se só há nome, investigar código na linha/contexto
        
        3. **PADRÕES DE CÓDIGOS COMUNS**:
           - Alfanuméricos: X0707, M9990, C3831, V9414
           - Numéricos: 22222, 94028, 02056, 03243
           - **NUNCA deixar color_name como NULL se há informação visível**
        
        ### PROBLEMA 3: AGRUPAMENTO INCORRETO
        **SOLUÇÃO - AGRUPAR VARIANTES POR COR**:
        
        Se vir códigos como CF5271MA96E.1, CF5271MA96E.2:
        ➜ AGRUPAR em UM produto CF5271MA96E com múltiplas cores
        
        ##  ALGORITMO PASSO-A-PASSO:
        
        ### PASSO 1: IDENTIFICAR ESTRUTURA
        1. Localizar tabela principal de produtos
        2. Identificar colunas de tamanhos (números ou letras)
        3. Identificar onde estão as cores (código + nome)
        4. Identificar onde estão as quantidades
        
        ### PASSO 2: EXTRAIR CADA PRODUTO
        Para cada linha/bloco de produto:
        1. **Código**: CF5015E0624, CF5271MA96E, etc.
        2. **Nome**: Traduzir para português se necessário
        3. **Cores**: Extrair código E nome da cor
        4. **Tamanhos**: Mapear posicionalmente com quantidades
        5. **Preços**: Extrair preços visíveis
        
        ### PASSO 3: VALIDAR E AGRUPAR
        1. Verificar tamanhos realistas
        2. Verificar cores completas
        3. Agrupar produtos com mesmo código base
        4. Incluir apenas tamanhos com quantidade > 0
        
        ## EXEMPLOS PRÁTICOS:
        
        ### Exemplo 1: Linha Simples
        ```
        CF5015E0624 | X0707 Asparago | 40 | 42 |    | 44 |
                    |                | 1  | 1  |    |    |
        ```
        ➜ RESULTADO: sizes: [{{"size":"40","quantity":1}}, {{"size":"42","quantity":1}}]
        
        ### Exemplo 2: Múltiplas Cores
        ```
        CF5271MA96E | M9990 Bege  | XS:1 | S:1 | M:1 | L:3 |
        CF5271MA96E | 22222 Preto | XS:1 | S:1 | M:1 |     |
        ```
        ➜ AGRUPAR em UM produto com 2 cores

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
        7. **NUNCA usar tamanho "UN"** para roupas (vestidos, blusas, calças, etc.)
        8. **SEMPRE extrair código E nome** da cor se visível
        9. **MAPEAR tamanhos por posição** nas colunas
        10. **SÓ incluir tamanhos com quantidade > 0**
        11. **AGRUPAR produtos** com mesmo código base
        12. **TRADUZIR nomes** para português

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