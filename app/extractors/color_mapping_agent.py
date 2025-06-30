# app/extractors/ai_color_mapping_agent.py
import logging
import json
import re
from typing import Dict, Any, List, Optional
import google.generativeai as genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.data.reference_data import COLOR_MAP

logger = logging.getLogger(__name__)

class ColorMappingAgent:

    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.color_map = COLOR_MAP
        
        # Estatísticas para logging
        self.stats = {
            "total_colors_processed": 0,
            "successfully_mapped": 0,
            "failed_mappings": 0,
            "mappings_details": []
        }
    
    def map_product_colors(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        
        # Reset das estatísticas
        self.stats = {
            "total_colors_processed": 0,
            "successfully_mapped": 0,
            "failed_mappings": 0,
            "mappings_details": []
        }
        
        mapped_products = []
        
        for product in products:
            mapped_product = product.copy()
            
            if "colors" in product and isinstance(product["colors"], list):
                mapped_colors = []
                
                for color in product["colors"]:
                    mapped_color = self._map_single_color(color)
                    mapped_colors.append(mapped_color)
                    self.stats["total_colors_processed"] += 1
                
                mapped_product["colors"] = mapped_colors
            
            # Também mapear referências se existirem
            if "references" in product and isinstance(product["references"], list):
                mapped_references = []
                
                for ref in product["references"]:
                    mapped_ref = ref.copy()
                    
                    # Mapear color_name na referência se necessário
                    if "color_name" in ref and ref["color_name"]:
                        mapped_color_info = self._map_color_name_with_ai(ref["color_name"])
                        
                        if mapped_color_info:
                            mapped_ref["color_name"] = mapped_color_info["name"]
                            # Atualizar color_code se não estiver presente ou for inconsistente
                            if not mapped_ref.get("color_code") or mapped_ref["color_code"] != mapped_color_info["code"]:
                                mapped_ref["color_code"] = mapped_color_info["code"]
                    
                    mapped_references.append(mapped_ref)
                
                mapped_product["references"] = mapped_references
            
            mapped_products.append(mapped_product)
        
        # Log das estatísticas
        self._log_mapping_stats()
        
        return mapped_products
    
    def _map_single_color(self, color: Dict[str, Any]) -> Dict[str, Any]:
        mapped_color = color.copy()
        
        original_name = color.get("color_name", "")
        original_code = color.get("color_code", "")
        
        if original_name:
            mapped_info = self._map_color_name_with_ai(original_name)
            
            if mapped_info:
                # Verificar se o código original estava correto
                if original_code and original_code != mapped_info["code"]:
                    logger.info(f"'{original_name}' tinha código {original_code} → corrigido para {mapped_info['code']} ({mapped_info['name']})")
                elif not original_code:
                    logger.info(f"Cor mapeada: '{original_name}' (sem código) → '{mapped_info['name']}' (código {mapped_info['code']})")
                else:
                    logger.info(f"Cor confirmada: '{original_name}' → '{mapped_info['name']}' (código {mapped_info['code']})")
                
                mapped_color["color_code"] = mapped_info["code"]
                mapped_color["color_name"] = mapped_info["name"]
                
                # Log da mudança
                change_info = {
                    "original_name": original_name,
                    "original_code": original_code,
                    "mapped_name": mapped_info["name"],
                    "mapped_code": mapped_info["code"],
                    "confidence": mapped_info.get("confidence", "high")
                }
                self.stats["mappings_details"].append(change_info)
                self.stats["successfully_mapped"] += 1
                
                return mapped_color
        
        if original_code and original_code in self.color_map:
            mapped_color["color_name"] = self.color_map[original_code]
            
            if original_name != self.color_map[original_code]:
                logger.warning(f"Usado código como fallback: '{original_name}' → '{self.color_map[original_code]}' (código {original_code})")
            
            self.stats["successfully_mapped"] += 1
            return mapped_color
        
        self.stats["failed_mappings"] += 1
        logger.warning(f"Falha completa no mapeamento: '{original_name}' (código: '{original_code}')")
        
        return mapped_color
    
    def _map_color_name_with_ai(self, color_name: str) -> Optional[Dict[str, str]]:
        if not color_name or not color_name.strip():
            return None

        try:
            color_examples = {
                "001": "Branco (white, blanc, bianco, branco)",
                "002": "Vermelho (red, rouge, rosso, vermelho)", 
                "003": "Verde (green, vert, verde, open green, medium green)",
                "004": "Castanho (brown, marrom, castanho, chocolate)",
                "005": "Amarelo (yellow, jaune, giallo, amarelo)",
                "006": "Lilás (lilac, lilas, viola, lilás)",
                "007": "Rosa (pink, rose, rosa, light pink, pastel pink)",
                "008": "Azul (blue, bleu, blu, azul, navy, dark blue, light blue, pastel blue)",
                "009": "Laranja (orange, arancione, laranja)",
                "010": "Preto (black, noir, nero, preto)",
                "011": "Cinza (gray, grey, gris, grigio, cinza, charcoal, cinzento, slate, ash)",
                "012": "Bege (beige, natural, nude, bege, open beige, cream, ivory)"
            }
            
            examples_text = "\n".join([f"{code}: {desc}" for code, desc in color_examples.items()])

            available_colors = []
            for code, name in self.color_map.items():
                available_colors.append(f"{code}: {name}")
            
            colors_list = "\n".join(available_colors)
            
            prompt = f"""
                # ESPECIALISTA EM MAPEAMENTO DE CORES

                Você é um especialista em cores que deve analisar o nome "{color_name}" e encontrar a cor mais adequada.

                ## ANÁLISE SEMÂNTICA DE CORES:
                {examples_text}

                ## CORES DISPONÍVEIS:
                {colors_list}

                ## REGRAS DE ANÁLISE SEMÂNTICA:
                1. **Tons de Cinza**: "Charcoal", "Slate", "Ash", "Graphite" → sempre "011: Cinza"
                2. **Tons de Azul**: qualquer variação de azul → sempre "008: Azul"  
                3. **Tons de Verde**: qualquer variação de verde → sempre "003: Verde"
                4. **Tons de Rosa**: qualquer variação de rosa/pink → sempre "007: Rosa"
                5. **Tons Naturais**: "Natural", "Nude", "Cream" → sempre "012: Bege"

                ## EXEMPLOS CRÍTICOS:
                - "Charcoal" = cinza escuro → código "011"
                - "Navy" = azul marinho → código "008"
                - "Natural" = cor natural/bege → código "012"

                ## COR A ANALISAR: "{color_name}"

                Analise semanticamente esta cor e retorne:
                ```json
                {{
                "code": "XXX",
                "name": "Nome Português",
                "confidence": "high",
                "reasoning": "Explicação da análise semântica"
                }}
                ```

                IMPORTANTE: Analise o SIGNIFICADO da cor, não apenas palavras-chave!
                """

            response = self.model.generate_content(prompt)
            response_text = response.text
            
            mapping_info = self._extract_json_from_response(response_text)
            
            if mapping_info and self._validate_mapping(mapping_info):
                return mapping_info
            else:
                logger.warning(f"Resposta inválida do Gemini para cor '{color_name}': {response_text[:100]}...")
                
                # Fallback inteligente
                fallback_mapping = self._get_fallback_mapping(color_name)
                if fallback_mapping:
                    logger.info(f"Usado mapeamento de fallback para '{color_name}' → {fallback_mapping['name']} ({fallback_mapping['code']})")
                    return fallback_mapping
                
                return None
                    
        except Exception as e:
            logger.error(f"Erro ao mapear cor '{color_name}' com IA: {str(e)}")
            
            # Fallback em caso de erro
            fallback_mapping = self._get_fallback_mapping(color_name)
            if fallback_mapping:
                logger.info(f"Usado mapeamento de fallback para '{color_name}' → {fallback_mapping['name']} ({fallback_mapping['code']})")
                return fallback_mapping
            
            return None
                    
        except Exception as e:
            logger.error(f"Erro ao mapear cor '{color_name}' com IA: {str(e)}")
            
            fallback_mapping = self._get_fallback_mapping(color_name)
            if fallback_mapping:
                logger.info(f"Usado mapeamento de fallback para '{color_name}' → {fallback_mapping['name']} ({fallback_mapping['code']})")
                return fallback_mapping
            
            return None

    def _get_fallback_mapping(self, color_name: str) -> Optional[Dict[str, str]]:
        fallback_mappings = {
            # Casos problemáticos específicos
            "charcoal": {"code": "011", "name": "Cinza"},
            "natural": {"code": "012", "name": "Bege"},
            "navy": {"code": "008", "name": "Azul"},
            "navy blue": {"code": "008", "name": "Azul"},
            "dark blue": {"code": "008", "name": "Azul"},
            "light blue": {"code": "008", "name": "Azul"},
            "pastel blue": {"code": "008", "name": "Azul"},
            "open green": {"code": "003", "name": "Verde"},
            "medium green": {"code": "003", "name": "Verde"},
            "light pink": {"code": "007", "name": "Rosa"},
            "pastel pink": {"code": "007", "name": "Rosa"},
            "light/pastel pink": {"code": "007", "name": "Rosa"},
            "open beige": {"code": "012", "name": "Bege"},
            "open red": {"code": "002", "name": "Vermelho"},
            
            # Cores básicas
            "white": {"code": "001", "name": "Branco"},
            "black": {"code": "010", "name": "Preto"},
            "red": {"code": "002", "name": "Vermelho"},
            "green": {"code": "003", "name": "Verde"},
            "blue": {"code": "008", "name": "Azul"},
            "pink": {"code": "007", "name": "Rosa"},
            "gray": {"code": "011", "name": "Cinza"},
            "grey": {"code": "011", "name": "Cinza"},
            "beige": {"code": "012", "name": "Bege"},
            
            # Variações portuguesas
            "branco": {"code": "001", "name": "Branco"},
            "preto": {"code": "010", "name": "Preto"},
            "vermelho": {"code": "002", "name": "Vermelho"},
            "verde": {"code": "003", "name": "Verde"},
            "azul": {"code": "008", "name": "Azul"},
            "rosa": {"code": "007", "name": "Rosa"},
            "cinza": {"code": "011", "name": "Cinza"},
            "cinzento": {"code": "011", "name": "Cinza"},
            "bege": {"code": "012", "name": "Bege"}
        }
        
        color_lower = color_name.lower().strip()
        
        # Procurar correspondência exata
        if color_lower in fallback_mappings:
            return fallback_mappings[color_lower]
        
        for key, mapping in fallback_mappings.items():
            if key in color_lower or color_lower in key:
                return mapping
        
        return None

    def _extract_json_from_response(self, response_text: str) -> Optional[Dict[str, str]]:
        try:
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, response_text)
            
            if matches:
                json_str = matches[0]
            else:
                json_pattern = r'(\{[\s\S]*?\})'
                matches = re.findall(json_pattern, response_text)
                
                if matches:
                    json_candidates = []
                    for potential_json in matches:
                        try:
                            parsed = json.loads(potential_json)
                            if isinstance(parsed, dict) and "code" in parsed:
                                json_candidates.append((len(potential_json), potential_json))
                        except:
                            continue
                    
                    if json_candidates:
                        json_candidates.sort(reverse=True)
                        json_str = json_candidates[0][1]
                    else:
                        return None
                else:
                    return None
            
            result = json.loads(json_str)
            return result
            
        except Exception as e:
            logger.warning(f"Erro ao extrair JSON: {str(e)}")
            return None
    
    def _validate_mapping(self, mapping_info: Dict[str, str]) -> bool:
        if not all(key in mapping_info for key in ["code", "name"]):
            return False
        
        code = mapping_info["code"]
        if code not in self.color_map:
            return False
        
        expected_name = self.color_map[code]
        provided_name = mapping_info["name"]
        
        if expected_name.lower() != provided_name.lower():
            logger.warning(f"Nome inconsistente para código {code}: esperado '{expected_name}', recebido '{provided_name}'")
            mapping_info["name"] = expected_name
        
        return True
    
    def _log_mapping_stats(self):
        total = self.stats["total_colors_processed"]
        if total == 0:
            return
        
        successful = self.stats["successfully_mapped"]
        failed = self.stats["failed_mappings"]
        
        logger.info("=" * 50)
        logger.info(f"RESUMO DO MAPEAMENTO DE CORES")
        logger.info(f"   Total processadas: {total}")
        logger.info(f"   Mapeadas com sucesso: {successful}")
        logger.info(f"   Não mapeadas: {failed}")
        logger.info(f"   Taxa de sucesso: {successful/total*100:.1f}%")
        logger.info("=" * 50)
    
    def get_mapping_report(self) -> Dict[str, Any]:
        return {
            "statistics": self.stats,
            "available_colors_count": len(self.color_map),
            "processing_method": "gemini_ai_mapping"
        }