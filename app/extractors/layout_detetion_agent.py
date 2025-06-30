# app/extractors/layout_detetion_agent.py
import os
import json
import logging
import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import google.generativeai as genai
from PIL import Image
import fitz

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.utils.file_utils import convert_pdf_to_images, optimize_image

logger = logging.getLogger(__name__)

class LayoutDetetionAgent:
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    async def analyze_document_structure(self, document_path: str) -> Dict[str, Any]:
        try:
            technical_analysis = self._analyze_pdf_technical_structure(document_path)
            
            visual_analysis = await self._analyze_visual_patterns(document_path)
            
            content_analysis = self._analyze_text_patterns(document_path)
            
            final_layout = self._determine_layout_strategy(
                technical_analysis, visual_analysis, content_analysis
            )
            
            return final_layout
            
        except Exception as e:
            logger.exception(f"Erro na análise de estrutura: {str(e)}")
            return self._get_fallback_analysis()
    
    def _analyze_pdf_technical_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Análise técnica da estrutura do PDF - detecta padrões matemáticos
        """
        try:
            analysis = {
                "coordinate_patterns": {},
                "text_alignment": {},
                "spacing_consistency": {},
                "column_detection": {},
                "row_detection": {},
                "table_indicators": []
            }
            
            pdf_document = fitz.open(pdf_path)
            
            # Analisar primeira página em detalhe
            page = pdf_document.load_page(0)
            blocks = page.get_text("dict")
            
            # Extrair coordenadas de todo o texto
            all_coordinates = []
            text_lines = []
            
            for block in blocks.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        line_bbox = line.get("bbox", [0, 0, 0, 0])
                        line_text = ""
                        
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                line_text += text + " "
                                bbox = span.get("bbox", [0, 0, 0, 0])
                                all_coordinates.append({
                                    "x1": bbox[0], "y1": bbox[1],
                                    "x2": bbox[2], "y2": bbox[3],
                                    "text": text,
                                    "font_size": span.get("size", 12)
                                })
                        
                        if line_text.strip():
                            text_lines.append({
                                "bbox": line_bbox,
                                "text": line_text.strip()
                            })
            
            # 1. DETECÇÃO DE COLUNAS
            analysis["column_detection"] = self._detect_columns_mathematically(all_coordinates)
            
            # 2. DETECÇÃO DE LINHAS/TABELAS
            analysis["row_detection"] = self._detect_rows_mathematically(text_lines)
            
            # 3. ANÁLISE DE ALINHAMENTO
            analysis["text_alignment"] = self._analyze_text_alignment(all_coordinates)
            
            # 4. ANÁLISE DE ESPAÇAMENTO
            analysis["spacing_consistency"] = self._analyze_spacing_patterns(text_lines)
            
            # 5. INDICADORES DE TABELA
            analysis["table_indicators"] = self._detect_table_indicators(all_coordinates, text_lines)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Erro na análise técnica: {str(e)}")
            return {"error": str(e)}
    
    def _detect_columns_mathematically(self, coordinates: List[Dict]) -> Dict[str, Any]:
        """
        Detecta colunas usando análise matemática de coordenadas X
        """
        if not coordinates:
            return {"column_count": 0, "confidence": 0.0}
        
        # Extrair todas as posições X de início
        x_positions = [coord["x1"] for coord in coordinates]
        
        # Agrupar posições X similares (clustering simples)
        x_positions.sort()
        clusters = []
        tolerance = 10  # pixels
        
        for x in x_positions:
            added_to_cluster = False
            for cluster in clusters:
                if abs(x - cluster["center"]) <= tolerance:
                    cluster["positions"].append(x)
                    cluster["center"] = sum(cluster["positions"]) / len(cluster["positions"])
                    cluster["count"] += 1
                    added_to_cluster = True
                    break
            
            if not added_to_cluster:
                clusters.append({
                    "center": x,
                    "positions": [x],
                    "count": 1
                })
        
        # Filtrar clusters significativos (com pelo menos 3 ocorrências)
        significant_clusters = [c for c in clusters if c["count"] >= 3]
        
        # Calcular confiança baseada na regularidade
        total_elements = len(coordinates)
        elements_in_clusters = sum(c["count"] for c in significant_clusters)
        confidence = elements_in_clusters / total_elements if total_elements > 0 else 0
        
        return {
            "column_count": len(significant_clusters),
            "confidence": confidence,
            "column_positions": [c["center"] for c in significant_clusters],
            "regularity_score": confidence
        }
    
    def _detect_rows_mathematically(self, text_lines: List[Dict]) -> Dict[str, Any]:
        """
        Detecta linhas regulares usando análise de coordenadas Y
        """
        if not text_lines:
            return {"row_count": 0, "confidence": 0.0}
        
        # Extrair posições Y e alturas
        y_positions = []
        line_heights = []
        
        for line in text_lines:
            bbox = line["bbox"]
            y_center = (bbox[1] + bbox[3]) / 2
            height = bbox[3] - bbox[1]
            
            y_positions.append(y_center)
            line_heights.append(height)
        
        # Calcular espaçamento entre linhas
        y_positions.sort()
        spacings = []
        
        for i in range(1, len(y_positions)):
            spacing = y_positions[i] - y_positions[i-1]
            spacings.append(spacing)
        
        if not spacings:
            return {"row_count": len(text_lines), "confidence": 0.0}
        
        # Analisar regularidade do espaçamento
        avg_spacing = sum(spacings) / len(spacings)
        spacing_variance = sum((s - avg_spacing) ** 2 for s in spacings) / len(spacings)
        spacing_std = spacing_variance ** 0.5
        
        # Confiança baseada na consistência do espaçamento
        cv = spacing_std / avg_spacing if avg_spacing > 0 else 1  # Coeficiente de variação
        regularity_confidence = max(0, 1 - cv)  # Quanto menor a variação, maior a confiança
        
        # Analisar altura das linhas
        avg_height = sum(line_heights) / len(line_heights)
        height_consistency = 1 - (max(line_heights) - min(line_heights)) / avg_height if avg_height > 0 else 0
        
        return {
            "row_count": len(text_lines),
            "confidence": (regularity_confidence + height_consistency) / 2,
            "avg_spacing": avg_spacing,
            "spacing_regularity": regularity_confidence,
            "height_consistency": height_consistency
        }
    
    def _analyze_text_alignment(self, coordinates: List[Dict]) -> Dict[str, Any]:
        """
        Analisa padrões de alinhamento do texto
        """
        if not coordinates:
            return {"left_aligned": 0, "right_aligned": 0, "centered": 0}
        
        # Agrupar por posições X similares
        x_groups = {}
        tolerance = 5
        
        for coord in coordinates:
            x = coord["x1"]
            grouped = False
            
            for group_x in x_groups.keys():
                if abs(x - group_x) <= tolerance:
                    x_groups[group_x].append(coord)
                    grouped = True
                    break
            
            if not grouped:
                x_groups[x] = [coord]
        
        # Contar elementos por grupo
        alignment_strength = {}
        for x_pos, group in x_groups.items():
            if len(group) >= 3:  # Grupos significativos
                alignment_strength[x_pos] = len(group)
        
        total_aligned = sum(alignment_strength.values())
        total_elements = len(coordinates)
        
        return {
            "alignment_groups": len(alignment_strength),
            "alignment_ratio": total_aligned / total_elements if total_elements > 0 else 0,
            "strongest_alignment": max(alignment_strength.values()) if alignment_strength else 0,
            "alignment_confidence": total_aligned / total_elements if total_elements > 0 else 0
        }
    
    def _analyze_spacing_patterns(self, text_lines: List[Dict]) -> Dict[str, Any]:
        """
        Analisa padrões de espaçamento para detectar estrutura
        """
        if len(text_lines) < 2:
            return {"consistency": 0, "pattern_detected": False}
        
        # Calcular espaçamentos horizontais dentro de cada linha
        horizontal_gaps = []
        
        for line in text_lines:
            text = line["text"]
            # Contar sequências de espaços
            space_sequences = re.findall(r'\s{2,}', text)
            if space_sequences:
                horizontal_gaps.extend([len(seq) for seq in space_sequences])
        
        # Calcular espaçamentos verticais
        y_positions = [line["bbox"][1] for line in text_lines]
        y_positions.sort()
        
        vertical_gaps = []
        for i in range(1, len(y_positions)):
            gap = y_positions[i] - y_positions[i-1]
            vertical_gaps.append(gap)
        
        # Analisar consistência
        def calculate_consistency(gaps):
            if not gaps:
                return 0
            avg = sum(gaps) / len(gaps)
            variance = sum((g - avg) ** 2 for g in gaps) / len(gaps)
            cv = (variance ** 0.5) / avg if avg > 0 else 1
            return max(0, 1 - cv)
        
        h_consistency = calculate_consistency(horizontal_gaps)
        v_consistency = calculate_consistency(vertical_gaps)
        
        return {
            "horizontal_consistency": h_consistency,
            "vertical_consistency": v_consistency,
            "overall_consistency": (h_consistency + v_consistency) / 2,
            "pattern_detected": (h_consistency + v_consistency) / 2 > 0.6
        }
    
    def _detect_table_indicators(self, coordinates: List[Dict], text_lines: List[Dict]) -> List[Dict]:
        """
        Detecta indicadores específicos de estrutura tabular
        """
        indicators = []
        
        # 1. Indicador: Números organizados em grid
        number_pattern = r'\b\d+\b'
        number_positions = []
        
        for coord in coordinates:
            if re.search(number_pattern, coord["text"]):
                number_positions.append((coord["x1"], coord["y1"]))
        
        if len(number_positions) > 10:  # Muitos números
            # Verificar se estão organizados em grid
            x_positions = [pos[0] for pos in number_positions]
            y_positions = [pos[1] for pos in number_positions]
            
            # Contar posições X únicas (colunas)
            unique_x = len(set(int(x/10)*10 for x in x_positions))  # Agrupamento de 10px
            unique_y = len(set(int(y/10)*10 for y in y_positions))  # Agrupamento de 10px
            
            if unique_x >= 3 and unique_y >= 3:
                indicators.append({
                    "type": "number_grid",
                    "confidence": min(0.9, (unique_x * unique_y) / 50),
                    "columns": unique_x,
                    "rows": unique_y
                })
        
        # 2. Indicador: Cabeçalhos regulares
        potential_headers = []
        for line in text_lines[:5]:  # Primeiras 5 linhas
            text = line["text"].upper()
            # Procurar por padrões de cabeçalho
            if any(word in text for word in ["SIZE", "COLOR", "QTY", "PRICE", "QUANTITY", "MODELO", "COR", "TAMANHO"]):
                potential_headers.append(line)
        
        if potential_headers:
            indicators.append({
                "type": "table_headers",
                "confidence": min(0.8, len(potential_headers) / 3),
                "headers_found": len(potential_headers)
            })
        
        # 3. Indicador: Estrutura repetitiva
        line_patterns = []
        for line in text_lines:
            # Criar "assinatura" da linha baseada no padrão de texto/números
            pattern = re.sub(r'[A-Za-z]+', 'TEXT', line["text"])
            pattern = re.sub(r'\d+', 'NUM', pattern)
            pattern = re.sub(r'\s+', ' ', pattern).strip()
            line_patterns.append(pattern)
        
        # Contar padrões repetidos
        pattern_counts = {}
        for pattern in line_patterns:
            if len(pattern) > 10:  # Padrões significativos
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        repeated_patterns = {p: c for p, c in pattern_counts.items() if c >= 3}
        
        if repeated_patterns:
            max_repetition = max(repeated_patterns.values())
            indicators.append({
                "type": "repetitive_structure",
                "confidence": min(0.9, max_repetition / 10),
                "pattern_repetitions": max_repetition
            })
        
        return indicators
    
    async def _analyze_visual_patterns(self, document_path: str) -> Dict[str, Any]:
        """
        Análise visual usando IA - completamente genérica
        """
        try:
            # Converter primeira página para imagem
            image_paths = convert_pdf_to_images(document_path, os.path.dirname(document_path), pages=[0])
            
            if not image_paths:
                return {"error": "Não foi possível converter para imagem"}
            
            # Otimizar imagem
            optimized_path = optimize_image(image_paths[0], os.path.dirname(image_paths[0]))
            image = Image.open(optimized_path)
            
            # Prompt genérico para análise visual
            visual_prompt = """
            # ANÁLISE VISUAL DE LAYOUT - DETECTOR GENÉRICO
            
            Analise esta imagem de documento comercial e identifique padrões estruturais:
            
            ## PADRÕES DE LAYOUT A DETECTAR:
            
            1. **GRID_TABULAR**: Dados organizados em grade clara com linhas e colunas
               - Elementos alinhados em matriz
               - Cabeçalhos visíveis
               - Células bem definidas
            
            2. **LIST_VERTICAL**: Lista vertical de itens
               - Itens empilhados verticalmente
               - Cada linha é um item completo
               - Estrutura de lista clara
            
            3. **LIST_HORIZONTAL**: Dados organizados horizontalmente
               - Elementos fluem da esquerda para direita
               - Agrupamentos horizontais
            
            4. **HYBRID_MIXED**: Combinação de elementos
               - Algumas seções tabulares
               - Outras seções em lista
               - Estrutura mista
            
            5. **FORM_FIELDS**: Estrutura de formulário
               - Campos e valores
               - Labels e inputs
            
            6. **FREE_TEXT**: Texto livre sem estrutura clara
               - Parágrafos de texto
               - Sem organização tabular
            
            ## ELEMENTOS ESTRUTURAIS A IDENTIFICAR:
            - Presença de linhas divisórias
            - Alinhamento de elementos
            - Padrões repetitivos
            - Organização de números/códigos
            - Densidade de informação
            - Consistência visual
            
            ## CARACTERÍSTICAS DOS DADOS:
            - Como produtos/itens são apresentados
            - Organização de códigos e quantidades
            - Padrão de cores/variações
            - Estrutura de preços
            
            Responda em JSON:
            ```json
            {
              "primary_layout": "GRID_TABULAR|LIST_VERTICAL|LIST_HORIZONTAL|HYBRID_MIXED|FORM_FIELDS|FREE_TEXT",
              "confidence": 0.0-1.0,
              "secondary_patterns": ["lista", "de", "padrões", "secundários"],
              "structural_elements": {
                "has_clear_grid": true/false,
                "has_headers": true/false,
                "has_dividing_lines": true/false,
                "alignment_quality": "high|medium|low",
                "data_density": "high|medium|low",
                "repetitive_patterns": true/false
              },
              "data_organization": {
                "item_presentation": "individual_rows|grouped_sections|scattered",
                "code_pattern": "systematic|mixed|none",
                "number_alignment": "columnar|inline|mixed",
                "price_organization": "dedicated_area|inline|scattered"
              },
              "extraction_hints": {
                "best_approach": "table_scan|line_by_line|section_by_section|adaptive",
                "complexity_level": "simple|moderate|complex",
                "special_considerations": ["lista", "de", "considerações"]
              }
            }
            ```
            """
            
            # Gerar análise visual
            response = self.model.generate_content([visual_prompt, image])
            
            # Extrair JSON da resposta
            visual_analysis = self._extract_json_from_text(response.text)
            
            return visual_analysis or {"error": "Falha na análise visual"}
            
        except Exception as e:
            logger.warning(f"Erro na análise visual: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_text_patterns(self, pdf_path: str) -> Dict[str, Any]:
        """
        Análise de padrões no texto extraído
        """
        try:
            pdf_document = fitz.open(pdf_path)
            full_text = ""
            
            # Extrair texto de até 3 páginas
            for page_num in range(min(3, len(pdf_document))):
                page = pdf_document.load_page(page_num)
                full_text += page.get_text()
            
            analysis = {
                "product_indicators": self._detect_product_patterns(full_text),
                "table_indicators": self._detect_table_patterns_text(full_text),
                "structure_indicators": self._detect_structure_patterns(full_text)
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Erro na análise de texto: {str(e)}")
            return {"error": str(e)}
    
    def _detect_product_patterns(self, text: str) -> Dict[str, Any]:
        """
        Detecta padrões que indicam produtos
        """
        # Padrões comuns de códigos de produto
        code_patterns = [
            r'\b[A-Z]{2,}\d{4,}\b',  # Ex: CF5015E0624
            r'\b\d{8,}\b',           # Códigos numéricos longos
            r'\b[A-Z]\d+[A-Z]*\b',   # Ex: T3216, MA82O
        ]
        
        product_codes = []
        for pattern in code_patterns:
            matches = re.findall(pattern, text)
            product_codes.extend(matches)
        
        # Padrões de tamanhos
        size_patterns = [
            r'\b(XS|S|M|L|XL|XXL|XXXL)\b',
            r'\b\d{2}\b',  # Tamanhos numéricos
            r'\b(2|4|6|8|10|12|14|16)\b'  # Tamanhos de roupa
        ]
        
        sizes_found = []
        for pattern in size_patterns:
            matches = re.findall(pattern, text)
            sizes_found.extend(matches)
        
        # Padrões de preços
        price_patterns = [
            r'\b\d+[,\.]\d{2}\b',
            r'\b\d+\.\d{2}\b',
            r'\€\s*\d+[,\.]\d{2}'
        ]
        
        prices_found = []
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            prices_found.extend(matches)
        
        return {
            "product_codes_found": len(set(product_codes)),
            "sizes_found": len(set(sizes_found)),
            "prices_found": len(set(prices_found)),
            "product_density": len(set(product_codes)) / max(1, len(text.split('\n'))),
            "has_product_structure": len(set(product_codes)) > 3 and len(set(sizes_found)) > 3
        }
    
    def _detect_table_patterns_text(self, text: str) -> Dict[str, Any]:
        """
        Detecta padrões tabulares no texto
        """
        lines = text.split('\n')
        
        # Detectar linhas com múltiplas colunas (muitos espaços)
        tabular_lines = 0
        for line in lines:
            space_sequences = len(re.findall(r'\s{3,}', line))
            if space_sequences >= 2:
                tabular_lines += 1
        
        # Detectar cabeçalhos
        header_keywords = ['size', 'color', 'qty', 'price', 'model', 'quantity', 'tamanho', 'cor', 'preço', 'modelo']
        header_lines = 0
        
        for line in lines[:10]:  # Primeiras 10 linhas
            line_lower = line.lower()
            if sum(1 for keyword in header_keywords if keyword in line_lower) >= 2:
                header_lines += 1
        
        return {
            "tabular_line_ratio": tabular_lines / max(1, len(lines)),
            "header_lines_found": header_lines,
            "has_table_structure": tabular_lines > len(lines) * 0.3,
            "table_confidence": min(1.0, tabular_lines / max(1, len(lines)) + header_lines * 0.2)
        }
    
    def _detect_structure_patterns(self, text: str) -> Dict[str, Any]:
        """
        Detecta padrões estruturais gerais
        """
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Analisar consistência no comprimento das linhas
        line_lengths = [len(line) for line in non_empty_lines]
        
        if line_lengths:
            avg_length = sum(line_lengths) / len(line_lengths)
            length_variance = sum((l - avg_length) ** 2 for l in line_lengths) / len(line_lengths)
            length_consistency = 1 - min(1, (length_variance ** 0.5) / avg_length) if avg_length > 0 else 0
        else:
            length_consistency = 0
        
        # Detectar repetição de padrões
        pattern_repetitions = {}
        for line in non_empty_lines:
            # Criar padrão baseado na estrutura
            pattern = re.sub(r'[A-Za-z]+', 'W', line)  # Palavras
            pattern = re.sub(r'\d+', 'N', pattern)     # Números
            pattern = re.sub(r'\s+', ' ', pattern).strip()
            
            if len(pattern) > 5:
                pattern_repetitions[pattern] = pattern_repetitions.get(pattern, 0) + 1
        
        max_repetitions = max(pattern_repetitions.values()) if pattern_repetitions else 0
        
        return {
            "line_consistency": length_consistency,
            "pattern_repetitions": max_repetitions,
            "structure_score": (length_consistency + min(1, max_repetitions / 10)) / 2,
            "is_structured": length_consistency > 0.5 or max_repetitions > 3
        }
    
    def _determine_layout_strategy(
        self, 
        technical: Dict[str, Any], 
        visual: Dict[str, Any], 
        content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combina todas as análises para determinar a melhor estratégia
        """
        # Scores para cada tipo de layout
        layout_scores = {
            "GRID_TABULAR": 0.0,
            "LIST_VERTICAL": 0.0,
            "LIST_HORIZONTAL": 0.0,
            "HYBRID_MIXED": 0.0,
            "FORM_FIELDS": 0.0,
            "FREE_TEXT": 0.0
        }
        
        # PONTUAÇÃO BASEADA NA ANÁLISE TÉCNICA
        if not technical.get("error"):
            # Colunas detectadas
            column_info = technical.get("column_detection", {})
            if column_info.get("column_count", 0) >= 5:
                layout_scores["GRID_TABULAR"] += 0.3 * column_info.get("confidence", 0)
            elif column_info.get("column_count", 0) >= 2:
                layout_scores["LIST_VERTICAL"] += 0.2 * column_info.get("confidence", 0)
            
            # Linhas regulares
            row_info = technical.get("row_detection", {})
            if row_info.get("confidence", 0) > 0.7:
                layout_scores["GRID_TABULAR"] += 0.2
                layout_scores["LIST_VERTICAL"] += 0.1
            
            # Alinhamento
            alignment = technical.get("text_alignment", {})
            if alignment.get("alignment_confidence", 0) > 0.8:
                layout_scores["GRID_TABULAR"] += 0.2
            
            # Indicadores de tabela
            table_indicators = technical.get("table_indicators", [])
            for indicator in table_indicators:
                if indicator.get("type") == "number_grid":
                    layout_scores["GRID_TABULAR"] += 0.3 * indicator.get("confidence", 0)
                elif indicator.get("type") == "table_headers":
                    layout_scores["GRID_TABULAR"] += 0.2 * indicator.get("confidence", 0)
                elif indicator.get("type") == "repetitive_structure":
                    layout_scores["LIST_VERTICAL"] += 0.2 * indicator.get("confidence", 0)
        
        # PONTUAÇÃO BASEADA NA ANÁLISE VISUAL
        if not visual.get("error"):
            primary_layout = visual.get("primary_layout", "")
            confidence = visual.get("confidence", 0)
            
            if primary_layout in layout_scores:
                layout_scores[primary_layout] += 0.4 * confidence
            
            # Elementos estruturais
            structural = visual.get("structural_elements", {})
            if structural.get("has_clear_grid"):
                layout_scores["GRID_TABULAR"] += 0.2
            if structural.get("repetitive_patterns"):
                layout_scores["LIST_VERTICAL"] += 0.1
                layout_scores["GRID_TABULAR"] += 0.1
        
        # PONTUAÇÃO BASEADA NO CONTEÚDO
        if not content.get("error"):
            # Indicadores de produto
            product_info = content.get("product_indicators", {})
            if product_info.get("has_product_structure"):
                layout_scores["GRID_TABULAR"] += 0.1
                layout_scores["LIST_VERTICAL"] += 0.1
            
            # Indicadores de tabela
            table_info = content.get("table_indicators", {})
            if table_info.get("has_table_structure"):
                layout_scores["GRID_TABULAR"] += 0.2 * table_info.get("table_confidence", 0)
            
            # Estrutura geral
            structure_info = content.get("structure_indicators", {})
            if structure_info.get("is_structured"):
                score = structure_info.get("structure_score", 0)
                layout_scores["GRID_TABULAR"] += 0.1 * score
                layout_scores["LIST_VERTICAL"] += 0.1 * score
        
        # DETERMINAR MELHOR LAYOUT
        best_layout = max(layout_scores, key=layout_scores.get)
        best_score = layout_scores[best_layout]
        
        # Mapear para estratégias de extração
        strategy_mapping = {
            "GRID_TABULAR": "table_extraction",
            "LIST_VERTICAL": "list_extraction", 
            "LIST_HORIZONTAL": "horizontal_list_extraction",
            "HYBRID_MIXED": "hybrid_extraction",
            "FORM_FIELDS": "form_extraction",
            "FREE_TEXT": "adaptive_extraction"
        }
        
        # Se score muito baixo, usar estratégia adaptativa
        if best_score < 0.3:
            best_layout = "HYBRID_MIXED"
            strategy = "adaptive_extraction"
            confidence = 0.3
        else:
            strategy = strategy_mapping[best_layout]
            confidence = min(0.95, best_score)
        
        # Gerar instruções específicas
        extraction_instructions = self._generate_extraction_instructions(
            best_layout, strategy, technical, visual, content
        )
        
        logger.info(f"Layout detectado: {best_layout} (confiança: {confidence:.2f})")
        logger.info(f"Estratégia selecionada: {strategy}")
        
        return {
            "layout_type": best_layout,
            "extraction_strategy": strategy,
            "confidence": confidence,
            "layout_scores": layout_scores,
            "technical_analysis": technical,
            "visual_analysis": visual,
            "content_analysis": content,
            "extraction_instructions": extraction_instructions
        }
    
    def _generate_extraction_instructions(
        self,
        layout_type: str,
        strategy: str,
        technical: Dict[str, Any],
        visual: Dict[str, Any], 
        content: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Gera instruções específicas baseadas na análise completa
        """
        instructions = {}
        
        # Instruções base por tipo de layout
        if layout_type == "GRID_TABULAR":
            instructions.update({
                "approach": "Processar como tabela estruturada - linha por linha, coluna por coluna",
                "product_identification": "Primeira coluna normalmente contém códigos/nomes de produtos",
                "size_extraction": "Tamanhos organizados em colunas horizontais - verificar cabeçalhos",
                "color_extraction": "Cores podem estar em linha dedicada ou coluna específica",
                "quantity_extraction": "Quantidades nas intersecções - células vazias = não disponível",
                "price_extraction": "Preços em coluna dedicada, normalmente no final"
            })
            
            # Ajustes baseados na análise técnica
            column_count = technical.get("column_detection", {}).get("column_count", 0)
            if column_count > 8:
                instructions["special_note"] = f"Muitas colunas detectadas ({column_count}) - podem ser tamanhos"
            
        elif layout_type == "LIST_VERTICAL":
            instructions.update({
                "approach": "Processar como lista vertical - cada linha é um item",
                "product_identification": "Cada linha de produtos tem código e nome no início",
                "size_extraction": "Tamanhos listados horizontalmente após descrição",
                "color_extraction": "Código/nome de cor próximo ao nome do produto",
                "quantity_extraction": "Quantidades por tamanho na mesma linha",
                "price_extraction": "Preço normalmente no final de cada linha"
            })
            
        elif layout_type == "LIST_HORIZONTAL":
            instructions.update({
                "approach": "Processar horizontalmente - dados fluem da esquerda para direita",
                "product_identification": "Produtos organizados em blocos horizontais",
                "size_extraction": "Tamanhos podem estar em sequência horizontal",
                "color_extraction": "Cores agrupadas por produto ou seção",
                "quantity_extraction": "Quantidades próximas aos respectivos tamanhos",
                "price_extraction": "Preços em posição fixa por produto"
            })
            
        elif layout_type == "HYBRID_MIXED":
            instructions.update({
                "approach": "Estratégia mista - adaptar por seção do documento",
                "product_identification": "Verificar padrão de cada seção separadamente",
                "size_extraction": "Combinar estratégias tabular e lista conforme necessário",
                "color_extraction": "Adaptar método por seção - pode variar",
                "quantity_extraction": "Verificar formato por grupo de produtos",
                "price_extraction": "Localização pode variar - verificar padrões"
            })
            
        else:  # FORM_FIELDS ou FREE_TEXT
            instructions.update({
                "approach": "Extração adaptativa - usar múltiplas estratégias",
                "product_identification": "Detectar produtos por padrões automáticos",
                "size_extraction": "Procurar tamanhos conhecidos próximos a produtos",
                "color_extraction": "Identificar códigos e nomes descritivos",
                "quantity_extraction": "Localizar números próximos a tamanhos",
                "price_extraction": "Identificar valores monetários"
            })
        
        # Adicionar considerações específicas baseadas nas análises
        considerations = []
        
        # Da análise técnica
        if technical.get("table_indicators"):
            for indicator in technical["table_indicators"]:
                if indicator.get("type") == "number_grid":
                    considerations.append("Grade numérica detectada - provavelmente quantidades em tabela")
                elif indicator.get("type") == "table_headers":
                    considerations.append("Cabeçalhos de tabela encontrados - usar para mapear colunas")
        
        # Da análise visual
        if visual.get("structural_elements"):
            elements = visual["structural_elements"]
            if elements.get("has_dividing_lines"):
                considerations.append("Linhas divisórias presentes - respeitar separações")
            if elements.get("data_density") == "high":
                considerations.append("Alta densidade de dados - cuidado com sobreposições")
        
        # Da análise de conteúdo
        if content.get("product_indicators"):
            product_info = content["product_indicators"]
            if product_info.get("product_density", 0) > 0.1:
                considerations.append("Alta densidade de códigos de produto detectada")
        
        if considerations:
            instructions["special_considerations"] = "; ".join(considerations)
        
        return instructions
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extrai JSON de texto com tratamento de erros
        """
        try:
            # Procurar por bloco de código JSON
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, text)
            
            if matches:
                return json.loads(matches[0])
            
            # Tentar interpretar como JSON direto
            try:
                return json.loads(text)
            except:
                # Procurar por objeto JSON na string
                json_pattern = r'(\{[\s\S]*\})'
                matches = re.findall(json_pattern, text)
                
                if matches:
                    for potential_json in matches:
                        try:
                            result = json.loads(potential_json)
                            if isinstance(result, dict):
                                return result
                        except:
                            continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Erro ao extrair JSON: {str(e)}")
            return None
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        return {
            "layout_type": "HYBRID_MIXED",
            "extraction_strategy": "adaptive_extraction",
            "confidence": 0.3,
            "layout_scores": {},
            "technical_analysis": {"error": "Falha na análise técnica"},
            "visual_analysis": {"error": "Falha na análise visual"},
            "content_analysis": {"error": "Falha na análise de conteúdo"},
            "extraction_instructions": {
                "approach": "Extração adaptativa conservadora",
                "product_identification": "Detectar produtos por códigos conhecidos",
                "size_extraction": "Procurar padrões de tamanhos conhecidos",
                "color_extraction": "Identificar códigos numéricos e nomes de cores",
                "quantity_extraction": "Localizar números próximos a tamanhos",
                "price_extraction": "Identificar valores monetários",
                "special_considerations": "Usar múltiplas estratégias devido à incerteza na detecção"
            }
        }