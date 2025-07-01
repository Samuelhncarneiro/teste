import logging
import re
from typing import Dict, Any, List, Optional
from app.utils.file_utils import extract_text_from_pdf
from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class UniversalAnalyzer:
    """
    Sistema universal que SE INTEGRA com o código existente.
    NÃO substitui nada, apenas ADICIONA inteligência.
    """
    
    def __init__(self):
        self.confidence_threshold = 0.3
        
    def analyze_document(self, document_path: str) -> Dict[str, Any]:
        """
        Análise universal que COMPLEMENTA a análise existente
        """
        try:
            # Extrair texto
            if document_path.lower().endswith('.pdf'):
                text = extract_text_from_pdf(document_path)
            else:
                with open(document_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Análise automática de padrões
            patterns = self._detect_all_patterns(text)
            
            # Determinar se deve usar sistema universal ou clássico
            use_universal = self._should_use_universal(patterns)
            
            return {
                "use_universal": use_universal,
                "patterns": patterns,
                "adaptive_prompt": self._generate_adaptive_prompt(patterns) if use_universal else None,
                "confidence": self._calculate_confidence(patterns)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise universal: {e}")
            return {"use_universal": False, "error": str(e)}
    
    def _detect_all_patterns(self, text: str) -> Dict[str, Any]:
        """Detecta padrões automaticamente"""
        return {
            "product_codes": self._detect_product_patterns(text),
            "size_systems": self._detect_size_patterns(text),
            "color_patterns": self._detect_color_patterns(text),
            "layout_type": self._detect_layout_type(text),
            "supplier_hints": self._detect_supplier_hints(text)
        }
    
    def _detect_product_patterns(self, text: str) -> List[Dict]:
        """Detecta padrões de códigos de produto"""
        import re
        
        patterns = []
        
        # Padrão HUGO BOSS style
        boss_pattern = r'\b[A-Z]{2,4}\d{6,12}[A-Z0-9]*\b'
        boss_matches = re.findall(boss_pattern, text)
        if len(boss_matches) >= 3:
            patterns.append({
                "type": "hugo_boss_style",
                "regex": boss_pattern,
                "examples": boss_matches[:3],
                "confidence": min(0.9, len(boss_matches) / 10)
            })
        
        # Padrão numérico longo (Marella/Dedimax)
        numeric_pattern = r'\b\d{10,15}\b'
        numeric_matches = re.findall(numeric_pattern, text)
        if len(numeric_matches) >= 3:
            patterns.append({
                "type": "numeric_long",
                "regex": numeric_pattern,
                "examples": numeric_matches[:3],
                "confidence": min(0.8, len(numeric_matches) / 10)
            })
        
        return sorted(patterns, key=lambda p: p["confidence"], reverse=True)
    
    def _detect_size_patterns(self, text: str) -> List[Dict]:
        """Detecta sistemas de tamanhos"""
        patterns = []
        
        # Internacional (XS, S, M, L, XL)
        intl_pattern = r'\b(XS|S|M|L|XL|XXL|XXXL)\b'
        intl_matches = re.findall(intl_pattern, text)
        if len(intl_matches) >= 4:
            patterns.append({
                "type": "international",
                "sizes": sorted(list(set(intl_matches))),
                "confidence": min(0.9, len(set(intl_matches)) / 6)
            })
        
        # Europeu (38, 40, 42, 44)
        eur_pattern = r'\b(3[4-9]|4[0-9]|5[0-6])\b'
        eur_matches = re.findall(eur_pattern, text)
        if len(eur_matches) >= 4:
            patterns.append({
                "type": "european_numeric",
                "sizes": sorted(list(set(eur_matches))),
                "confidence": min(0.8, len(set(eur_matches)) / 8)
            })
        
        # Jeans (26, 28, 30, 32)
        jeans_pattern = r'\b(2[4-9]|3[0-6])\b'
        jeans_matches = re.findall(jeans_pattern, text)
        if len(jeans_matches) >= 3:
            patterns.append({
                "type": "jeans_waist",
                "sizes": sorted(list(set(jeans_matches))),
                "confidence": min(0.7, len(set(jeans_matches)) / 6)
            })
        
        logger.info(f"📏 Tamanhos: {len(patterns)} sistemas detectados")
        return sorted(patterns, key=lambda p: p["confidence"], reverse=True)
    
    def _detect_color_patterns(self, text: str) -> List[Dict]:
        """✅ MÉTODO ADICIONADO - Detecta padrões de cores"""
        patterns = []
        
        # Códigos numéricos de cores (001, 002, 018, etc.)
        numeric_color_pattern = r'\b\d{3,4}\b(?=\s*[-:]?\s*[A-Za-z])'
        numeric_matches = re.findall(numeric_color_pattern, text)
        if len(numeric_matches) >= 3:
            patterns.append({
                "type": "numeric_codes",
                "examples": numeric_matches[:5],
                "confidence": min(0.8, len(numeric_matches) / 10)
            })
        
        # Códigos alfanuméricos (V9414, M9847, etc.)
        alpha_color_pattern = r'\b[A-Z]\d{4}\b'
        alpha_matches = re.findall(alpha_color_pattern, text)
        if len(alpha_matches) >= 2:
            patterns.append({
                "type": "alpha_numeric",
                "examples": alpha_matches[:5],
                "confidence": min(0.7, len(alpha_matches) / 8)
            })
        
        # Nomes de cores em português/inglês
        color_names_pattern = r'\b(negro|nero|black|white|branco|red|vermelho|blue|azul|green|verde|rosa|pink|cinza|gray|bege|beige)\b'
        color_names = re.findall(color_names_pattern, text, re.IGNORECASE)
        if len(color_names) >= 2:
            patterns.append({
                "type": "color_names",
                "examples": list(set(color_names))[:5],
                "confidence": min(0.6, len(set(color_names)) / 8)
            })
        
        logger.info(f"🎨 Cores: {len(patterns)} padrões detectados")
        return sorted(patterns, key=lambda p: p["confidence"], reverse=True)
    
    def _detect_supplier_hints(self, text: str) -> Dict[str, Any]:
        """✅ MÉTODO ADICIONADO - Detecta pistas sobre o fornecedor"""
        
        # Lista de fornecedores conhecidos
        known_suppliers = [
            "MARELLA", "DEDIMAX", "LIU.JO", "LIUJO", "HUGO BOSS", "BOSS",
            "PAUL & SHARK", "TOMMY HILFIGER", "GANT", "RALPH LAUREN",
            "WEEKEND MAXMARA", "TWINSET", "BRAX", "MEYER"
        ]
        
        supplier_scores = {}
        for supplier in known_suppliers:
            count = len(re.findall(supplier, text, re.IGNORECASE))
            if count > 0:
                supplier_scores[supplier] = count
        
        if supplier_scores:
            best_supplier = max(supplier_scores.items(), key=lambda x: x[1])
            confidence = min(0.9, best_supplier[1] / 5)
            
            logger.info(f"🏢 Fornecedor detectado: {best_supplier[0]} (confiança: {confidence:.2f})")
            
            return {
                "detected_supplier": best_supplier[0],
                "confidence": confidence,
                "all_matches": supplier_scores
            }
        
        # Se não encontrar fornecedor conhecido
        logger.info("🏢 Nenhum fornecedor conhecido detectado")
        return {
            "detected_supplier": "UNKNOWN",
            "confidence": 0.1,
            "all_matches": {}
        }
    
    def _detect_layout_type(self, text: str) -> Dict[str, Any]:
        """Detecta tipo de layout"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return {"type": "empty", "confidence": 0.0}
        
        # Detectar estrutura tabular
        tabular_lines = 0
        for line in lines:
            spaces = len(re.findall(r'\s{3,}', line))
            if spaces >= 2:
                tabular_lines += 1
        
        tabular_ratio = tabular_lines / len(lines)
        
        if tabular_ratio > 0.4:
            layout_type = "tabular"
            confidence = tabular_ratio
        else:
            layout_type = "sequential"
            confidence = 1 - tabular_ratio
        
        logger.info(f"📋 Layout: {layout_type} (confiança: {confidence:.2f})")
        
        return {
            "type": layout_type,
            "confidence": confidence,
            "tabular_ratio": tabular_ratio,
            "total_lines": len(lines)
        }
    
    def _calculate_confidence(self, patterns: Dict[str, Any]) -> float:
        
        confidences = []
        
        # Confiança dos códigos de produto
        if patterns["product_codes"]:
            confidences.append(patterns["product_codes"][0]["confidence"])
        
        # Confiança dos sistemas de tamanho
        if patterns["size_systems"]:
            confidences.append(patterns["size_systems"][0]["confidence"])
        
        # Confiança do layout
        if patterns["layout_type"]:
            confidences.append(patterns["layout_type"]["confidence"])
        
        # Confiança do fornecedor
        if patterns["supplier_hints"]:
            confidences.append(patterns["supplier_hints"]["confidence"])
        
        # Confiança das cores
        if patterns["color_patterns"]:
            confidences.append(patterns["color_patterns"][0]["confidence"])
        
        if not confidences:
            return 0.0
        
        # Média ponderada
        overall_confidence = sum(confidences) / len(confidences)
        
        logger.info(f"📊 Confiança geral: {overall_confidence:.2f}")
        
        return overall_confidence

    def _should_use_universal(self, patterns: Dict[str, Any]) -> bool:
        """Decide se deve usar sistema universal ou clássico"""
        
        # Critérios para usar universal:
        # 1. Confiança alta nos padrões detectados
        # 2. Padrões não reconhecidos pelo sistema clássico
        # 3. Layout complexo
        
        product_confidence = patterns["product_codes"][0]["confidence"] if patterns["product_codes"] else 0
        size_confidence = patterns["size_systems"][0]["confidence"] if patterns["size_systems"] else 0
        layout_confidence = patterns["layout_type"]["confidence"]
        
        overall_confidence = (product_confidence + size_confidence + layout_confidence) / 3
        
        # Usar universal se confiança for boa OU se for documento complexo
        return overall_confidence > 0.5 or patterns["layout_type"]["type"] == "tabular"
    
    def _generate_adaptive_prompt(self, patterns: Dict[str, Any]) -> str:
        """Gera prompt adaptativo baseado nos padrões"""
        
        sections = ["# EXTRAÇÃO ADAPTATIVA AUTOMÁTICA\n"]
        
        # Seção de produtos
        if patterns["product_codes"]:
            best_pattern = patterns["product_codes"][0]
            sections.append(f"""
            ## CÓDIGOS DE PRODUTO DETECTADOS:
            - Padrão: {best_pattern['type']}
            - Regex: {best_pattern['regex']}
            - Exemplos: {', '.join(best_pattern['examples'])}
            - Use APENAS códigos que seguem este padrão
            """)
        
        # Seção de tamanhos
        if patterns["size_systems"]:
            best_size = patterns["size_systems"][0]
            sections.append(f"""
            ## SISTEMA DE TAMANHOS DETECTADO:
            - Tipo: {best_size['type']}
            - Tamanhos encontrados: {', '.join(best_size['sizes'])}
            - Mapear posicionalmente com quantidades
            """)
        
        # Estratégia de layout
        layout = patterns["layout_type"]
        if layout["type"] == "tabular":
            sections.append("""
            ## ESTRATÉGIA: EXTRAÇÃO TABULAR
            - Processar linha por linha
            - Mapear colunas por posição
            - Tamanhos em linha, quantidades abaixo
            """)
        else:
            sections.append("""
            ## ESTRATÉGIA: EXTRAÇÃO SEQUENCIAL
            - Cada linha é um item
            - Dados organizados horizontalmente
            """)
        
        return "\n".join(sections)
