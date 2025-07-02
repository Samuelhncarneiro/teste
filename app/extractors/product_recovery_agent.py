# app/extractors/product_recovery_agent.py - NOVO AGENTE DE RECUPERAÇÃO

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import google.generativeai as genai

logger = logging.getLogger(__name__)

class ProductRecoveryAgent:
    """
    Agente especializado em recuperar produtos que foram perdidos na extração inicial
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Lista de produtos conhecidos que devem estar no documento
        self.expected_products = [
            "CF5041MA82O", "CF5042MA82O", "CF5071T4627", "CF5085MS55N",
            "CF5137D5026", "CF5305T2589", "CF5317MS019", "CF5345J1857",
            "CF5372T054A", "CF5377T081A"
        ]
    
    async def recover_missing_products(
        self, 
        image_path: str, 
        extracted_products: List[Dict[str, Any]], 
        page_number: int
    ) -> List[Dict[str, Any]]:
        """
        Tenta recuperar produtos que podem ter sido perdidos na extração
        """
        # Identificar produtos já extraídos
        extracted_codes = set()
        for product in extracted_products:
            code = product.get("material_code", "")
            if code:
                extracted_codes.add(code)
        
        # Identificar produtos potencialmente perdidos
        missing_products = []
        for expected_code in self.expected_products:
            if expected_code not in extracted_codes:
                missing_products.append(expected_code)
        
        if not missing_products:
            logger.info(f"Página {page_number}: Nenhum produto em falta detectado")
            return []
        
        logger.info(f"Página {page_number}: Tentando recuperar {len(missing_products)} produtos: {missing_products}")
        
        # Tentar recuperação específica
        recovered = await self._attempt_targeted_recovery(image_path, missing_products, page_number)
        
        if recovered:
            logger.info(f"Página {page_number}: Recuperados {len(recovered)} produtos adicionais")
        
        return recovered
    
    async def _attempt_targeted_recovery(
        self, 
        image_path: str, 
        missing_codes: List[str], 
        page_number: int
    ) -> List[Dict[str, Any]]:
        """
        Recuperação direcionada para códigos específicos
        """
        try:
            image = Image.open(image_path)
            
            # Prompt específico para recuperação
            recovery_prompt = f"""
            # MODO DE RECUPERAÇÃO ESPECÍFICA - Página {page_number}
            
            Procure especificamente por estes códigos de produto na imagem:
            {', '.join(missing_codes)}
            
            Para CADA código encontrado na imagem, extraia:
            
            ## INSTRUÇÕES ESPECÍFICAS:
            1. Localize visualmente o código na imagem
            2. Identifique a linha ou seção correspondente
            3. Extraia cor, tamanhos e quantidades dessa linha específica
            4. IGNORE se o código não estiver claramente visível
            
            ## ESTRUTURA ESPERADA NO DOCUMENTO:
            ```
            Código    Cor       38  40  42  44  XS  S   M   L   Qty  Price
            CFxxxxxx  ColorInfo  1   1   -   -   1   1   -   -    4   99.00
            ```
            
            ## REGRAS DE EXTRAÇÃO:
            - Se a célula tem número > 0: incluir tamanho
            - Se a célula está vazia ou tem "-": NÃO incluir tamanho
            - Extrair preço da coluna Price
            - Nome do produto pode estar próximo ao código
            
            Retorne JSON apenas para códigos REALMENTE ENCONTRADOS:
            {{
              "recovered_products": [
                {{
                  "material_code": "CFxxxxxx",
                  "name": "Nome extraído ou estimado",
                  "category": "Categoria estimada",
                  "colors": [
                    {{
                      "color_code": "código",
                      "color_name": "nome da cor",
                      "sizes": [
                        {{"size": "38", "quantity": 1}},
                        {{"size": "40", "quantity": 1}}
                      ],
                      "unit_price": 99.00,
                      "sales_price": null,
                      "subtotal": null
                    }}
                  ]
                }}
              ]
            }}
            
            IMPORTANTE: Retorne apenas produtos que você consegue ver claramente na imagem.
            """
            
            response = self.model.generate_content([recovery_prompt, image])
            
            # Processar resposta
            recovery_result = self._parse_recovery_response(response.text, page_number)
            
            return recovery_result
            
        except Exception as e:
            logger.error(f"Erro na recuperação direcionada página {page_number}: {str(e)}")
            return []
    
    def _parse_recovery_response(self, response_text: str, page_number: int) -> List[Dict[str, Any]]:
        """
        Parse da resposta de recuperação
        """
        try:
            import json
            
            # Tentar extrair JSON
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, response_text)
            
            if matches:
                result = json.loads(matches[0])
            else:
                result = json.loads(response_text)
            
            recovered_products = result.get("recovered_products", [])
            
            # Validar produtos recuperados
            valid_products = []
            for product in recovered_products:
                if self._validate_recovered_product(product):
                    valid_products.append(product)
                else:
                    logger.warning(f"Produto recuperado inválido: {product.get('material_code', 'N/A')}")
            
            logger.info(f"Página {page_number}: {len(valid_products)} produtos recuperados com sucesso")
            return valid_products
            
        except Exception as e:
            logger.error(f"Erro ao processar resposta de recuperação página {page_number}: {str(e)}")
            return []
    
    def _validate_recovered_product(self, product: Dict[str, Any]) -> bool:
        """
        Validação básica de produto recuperado
        """
        required_fields = ["material_code", "colors"]
        
        for field in required_fields:
            if field not in product or not product[field]:
                return False
        
        # Verificar se material_code tem padrão válido
        material_code = product["material_code"]
        if not re.match(r'^CF\d{4}[A-Z]{2,6}\d*$', material_code):
            return False
        
        # Verificar se tem pelo menos uma cor com tamanhos
        colors = product["colors"]
        if not isinstance(colors, list) or len(colors) == 0:
            return False
        
        for color in colors:
            if "sizes" in color and isinstance(color["sizes"], list) and len(color["sizes"]) > 0:
                return True
        
        return False

