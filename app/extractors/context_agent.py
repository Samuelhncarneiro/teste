# app/extractors/context_agent.py
import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
import google.generativeai as genai
from PIL import Image
import fitz

from app.config import GEMINI_API_KEY, GEMINI_MODEL, CONVERTED_DIR
from app.utils.file_utils import convert_pdf_to_images, extract_text_from_pdf, optimize_image
from app.data.reference_data import get_supplier_code, SUPPLIER_MAP
from app.utils.supplier_utils import match_supplier_name, normalize_supplier_name, get_normalized_supplier

logger = logging.getLogger(__name__)

class ContextAgent:
    """
    Agente avançado para análise de contexto que identifica layout, fornecedor, 
    marca e metadados do documento, incluindo informações sobre a localização
    e estrutura dos produtos para auxiliar o agente de extração.
    """
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        """
        Inicializa o agente de contexto avançado
        
        Args:
            api_key: Chave de API do Gemini (default: valor do .env)
        """
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    async def analyze_document(self, document_path: str) -> Dict[str, Any]:
        """
        Analisa o documento completo para extrair contexto e informações sobre layout
        
        Args:
            document_path: Caminho para o documento
            
        Returns:
            Dict: Informações contextuais e de layout do documento
        """
        # Extrair nome do arquivo para uso na análise
        filename = os.path.basename(document_path)
        fallback_info = {
            "document_type": "Documento de pedido",
            "supplier": "", 
            "brand": "",
            "customer": "",
            "reference_number": "",
            "date": "",
            "season": "",
            "file_name": filename,
            "layout_info": {
                "has_table_structure": True,
                "product_sections": [],
                "headers_detected": []
            }
        }
        
        # Verificar se é um PDF
        is_pdf = document_path.lower().endswith('.pdf')
        if not is_pdf:
            logger.warning(f"Documento não é PDF, usando informações de fallback com análise limitada")
            return self._ensure_supplier_and_brand(fallback_info)
        
        try:
            # Extrair texto do PDF para análise inicial
            pdf_text = extract_text_from_pdf(document_path)
            
            # Analisar PDF com PyMuPDF para informações estruturais
            doc_structure = self._analyze_pdf_structure(document_path)
            
            # Preparar imagem da primeira página para análise visual
            first_page_image = await self._prepare_first_page_image(document_path)
            
            if first_page_image:
                # Realizar análise completa com texto e imagem
                context_info = await self._analyze_with_image_and_text(
                    document_path, 
                    first_page_image, 
                    pdf_text, 
                    fallback_info,
                    doc_structure
                )
            else:
                # Análise baseada apenas em texto
                context_info = await self._analyze_text_only(
                    pdf_text, 
                    fallback_info,
                    doc_structure
                )
            
            # Garantir informações de fornecedor e marca
            context_info = self._ensure_supplier_and_brand(context_info)
            
            logger.info(f"Análise de contexto completa: {len(context_info)} campos extraídos")
            return context_info
            
        except Exception as e:
            logger.exception(f"Erro na análise de contexto avançada: {str(e)}")
            return self._ensure_supplier_and_brand(fallback_info)
    
    def _analyze_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analisa a estrutura do PDF para detectar tabelas, seções e layout
        
        Args:
            pdf_path: Caminho para o PDF
            
        Returns:
            Dict: Informações de estrutura do documento
        """
        try:
            structure_info = {
                "page_count": 0,
                "has_tables": False,
                "detected_tables": [],
                "text_blocks": [],
                "potential_headers": []
            }
            
            # Abrir o documento PDF
            pdf_document = fitz.open(pdf_path)
            structure_info["page_count"] = len(pdf_document)
            
            # Analisar apenas as primeiras 2 páginas para entender a estrutura
            pages_to_analyze = min(2, len(pdf_document))
            
            for page_num in range(pages_to_analyze):
                page = pdf_document.load_page(page_num)
                
                # Extrair blocos de texto
                blocks = page.get_text("blocks")
                structure_info["text_blocks"].extend([
                    {"page": page_num, "bbox": block[:4], "text": block[4]} 
                    for block in blocks
                ])
                
                # Analisar tabelas - uma abordagem simples baseada em texto
                text = page.get_text()
                
                # Detectar cabeçalhos potenciais (primeira linha de cada bloco)
                for block in blocks:
                    block_text = block[4]
                    first_line = block_text.split('\n')[0] if '\n' in block_text else block_text
                    if len(first_line.strip()) > 0 and len(first_line) < 100:  # Provavelmente um cabeçalho
                        structure_info["potential_headers"].append({"page": page_num, "text": first_line.strip()})
                
                # Detectar tabelas por padrões de alinhamento no texto
                lines = text.split('\n')
                
                # Padrões que sugerem uma tabela
                potential_table_lines = []
                for line in lines:
                    # Contar número de sequências de espaços com mais de 3 espaços
                    space_sequences = len(re.findall(r'\s{3,}', line))
                    if space_sequences >= 2 and len(line.strip()) > 10:
                        potential_table_lines.append(line)
                
                # Se encontramos linhas que parecem de tabela
                if len(potential_table_lines) > 3:  # Pelo menos 3 linhas para considerar uma tabela
                    structure_info["has_tables"] = True
                    structure_info["detected_tables"].append({
                        "page": page_num,
                        "sample_lines": potential_table_lines[:3],
                        "estimated_rows": len(potential_table_lines)
                    })
            
            return structure_info
            
        except Exception as e:
            logger.warning(f"Erro ao analisar estrutura do PDF: {str(e)}")
            return {
                "page_count": 0,
                "has_tables": True,  # Assumir que tem tabelas por padrão
                "error": str(e)
            }
    
    async def _prepare_first_page_image(self, document_path: str) -> Optional[Image.Image]:
        """
        Prepara a imagem da primeira página para análise
        
        Args:
            document_path: Caminho para o documento
            
        Returns:
            Optional[Image.Image]: Imagem da primeira página ou None se falhar
        """
        try:
            # Converter apenas a primeira página
            image_paths = convert_pdf_to_images(document_path, CONVERTED_DIR, pages=[0])
            
            if not image_paths or len(image_paths) == 0:
                logger.warning(f"Não foi possível converter a primeira página para imagem")
                return None
                
            # Otimizar a imagem
            first_page_img = image_paths[0]
            optimized_path = optimize_image(first_page_img, os.path.dirname(first_page_img))
            
            # Carregar a imagem
            return Image.open(optimized_path)
            
        except Exception as e:
            logger.warning(f"Erro ao preparar imagem da primeira página: {str(e)}")
            return None
    
    async def _analyze_with_image_and_text(
        self, 
        document_path: str,
        image: Image.Image, 
        pdf_text: str, 
        fallback_info: Dict[str, Any],
        doc_structure: Dict[str, Any]
    ) -> Dict[str, Any]:

        filename = os.path.basename(document_path)
        
        # Incluir nome do arquivo e informações estruturais na análise
        filename_hint = f"Nome do arquivo: {filename}"
        structure_hint = self._format_structure_hint(doc_structure)
        
        # Limitar o texto para o prompt
        limited_text = pdf_text[:2000] if pdf_text else ""
        
        # Construir prompt avançado para análise de contexto e layout
        context_prompt = f"""
        # ANÁLISE AVANÇADA DE DOCUMENTO COMERCIAL

        Você é um especialista em análise de documentos comerciais com foco em pedidos, notas de encomenda e faturas.
        
        ## Objetivos da Análise:
        1. Extrair metadados do documento (emissor, cliente, referências, etc.)
        2. Identificar o layout e a estrutura das informações de produtos
        3. Fornecer orientações precisas para o agente de extração sobre onde encontrar os dados

        ## Metadados a Extrair:
        - Tipo de documento (nota de encomenda, pedido, orçamento, etc.)
        - Fornecedor/Emissor (empresa que emite o documento) - PRIORIDADE MÁXIMA (Extrai de acordo com o arquivo de referência)
        - Cliente/Destinatário
        - Número de referência/pedido
        - Data do documento
        - Marca dos produtos - PRIORIDADE ALTA
        - Temporada/Coleção (se aplicável)
        - Condições de pagamento

        ## Análise de Layout:
        - Estrutura geral do documento (tabular, formulário, misto)
        - Localização das informações de produtos (páginas, seções)
        - Padrão de apresentação das cores e tamanhos
        - Como identificar códigos de produto vs. códigos de cor
        - Formato de preços e quantidades
        
        ## Informações Contextuais:
        {filename_hint}
        
        ## Informações de Estrutura Detectadas:
        {structure_hint}
        
        ## Texto Extraído (Parcial):
        {limited_text}
        
        Analise cuidadosamente a imagem e o texto fornecido e retorne sua análise completa em formato JSON:
        
        ```json
        {{
          "document_type": "Tipo de documento",
          "supplier": "Nome do fornecedor/emissor",
          "customer": "Nome do cliente/destinatário",
          "reference_number": "Número de referência do documento",
          "date": "Data do documento",
          "brand": "Marca dos produtos",
          "season": "Temporada/Coleção",
          "payment_terms": "Condições de pagamento",
          "layout_info": {{
            "general_structure": "Descrição da estrutura geral do documento (tabular, formulário, etc.)",
            "product_location": "Onde encontrar os produtos no documento (página inicial, todas as páginas, etc.)",
            "product_identifier": "Como identificar os produtos (código, nome, etc.)",
            "color_pattern": "Como as cores são apresentadas (linhas separadas, colunas, etc.)",
            "size_pattern": "Como os tamanhos são apresentados (colunas, grupos, etc.)",
            "quantity_format": "Como as quantidades são apresentadas",
            "price_format": "Como os preços são apresentados",
            "table_headers": ["Lista", "de", "cabeçalhos", "detectados"],
            "special_instructions": "Instruções especiais para o agente de extração"
          }}
        }}
        ```
        
        Forneça informações DETALHADAS na seção layout_info - isso será crítico para o agente de extração.
        """
        
        try:
            # Gerar resposta com base na imagem e no prompt
            context_response = self.model.generate_content([context_prompt, image])
            context_text = context_response.text
            
            # Extrair JSON da resposta
            context_info = self._extract_json_from_text(context_text)
            
            # Verificar se extraímos informações válidas
            if not context_info or not isinstance(context_info, dict):
                logger.warning("JSON inválido na resposta de análise com imagem. Tentando análise apenas com texto.")
                return await self._analyze_text_only(pdf_text, fallback_info, doc_structure)
            
            # Garantir campos obrigatórios
            self._ensure_required_fields(context_info, fallback_info)
            
            # Complementar com análise de estrutura
            if "layout_info" not in context_info or not context_info["layout_info"]:
                context_info["layout_info"] = self._generate_layout_info(doc_structure)
            
            # Adicionar nome do arquivo para uso posterior
            context_info["file_name"] = filename
            
            logger.info(f"Análise com imagem e texto concluída com sucesso")
            return context_info
            
        except Exception as e:
            logger.exception(f"Erro ao analisar com imagem e texto: {str(e)}")
            # Fallback para análise apenas com texto
            return await self._analyze_text_only(pdf_text, fallback_info, doc_structure)
    
    async def _analyze_text_only(
        self, 
        pdf_text: str, 
        fallback_info: Dict[str, Any],
        doc_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analisa o documento usando apenas o texto quando a análise de imagem falha
        
        Args:
            pdf_text: Texto extraído do PDF
            fallback_info: Informações de fallback
            doc_structure: Estrutura do documento analisada
            
        Returns:
            Dict: Informações contextuais do documento
        """
        try:
            # Limitar o tamanho do texto
            limited_text = pdf_text[:3000] if pdf_text else ""
            
            # Incluir nome do arquivo e informações estruturais na análise
            filename_hint = f"Nome do arquivo: {fallback_info.get('file_name', '')}"
            structure_hint = self._format_structure_hint(doc_structure)
            
            # Prompt simplificado para análise de texto com foco em contexto e layout
            prompt = f"""
            # ANÁLISE DE DOCUMENTO COMERCIAL (APENAS TEXTO)

            Analise o texto extraído deste documento PDF e extraia informações sobre contexto e layout.
            
            ## Metadados a Extrair:
            - Tipo de documento (nota de encomenda, pedido, orçamento, etc.)
            - Fornecedor/Emissor (empresa que emite o documento) - PRIORIDADE MÁXIMA
            - Cliente/Destinatário
            - Número de referência/pedido
            - Data do documento
            - Marca dos produtos - PRIORIDADE ALTA
            - Temporada/Coleção (se aplicável)
            
            ## Análise de Layout (a partir do texto):
            - Estrutura geral do documento (tabular, formulário, misto)
            - Como identificar códigos de produto vs. códigos de cor
            - Formato de preços e quantidades
            
            ## Informações Contextuais:
            {filename_hint}
            
            ## Informações de Estrutura Detectadas:
            {structure_hint}
            
            ## Texto do PDF (Parcial):
            {limited_text}
            
            Retorne sua análise em formato JSON:
            
            ```json
            {{
              "document_type": "Tipo de documento",
              "supplier": "Nome do fornecedor/emissor"(O nome do fornecedor terá de ser igual ao do arquivo de referência),
              "customer": "Nome do cliente/destinatário",
              "reference_number": "Número de referência do documento",
              "date": "Data do documento",
              "brand": "Marca dos produtos",
              "season": "Temporada/Coleção",
              "layout_info": {{
                "general_structure": "Descrição da estrutura geral",
                "product_identifier": "Como identificar os produtos",
                "color_pattern": "Como as cores são apresentadas",
                "size_pattern": "Como os tamanhos são apresentados",
                "special_instructions": "Instruções especiais para o agente de extração"
              }}
            }}
            ```
            """
            
            # Gerar resposta
            response = self.model.generate_content(prompt)
            context_text = response.text
            
            # Extrair JSON
            context_info = self._extract_json_from_text(context_text)
            
            # Verificar se extraímos informações válidas
            if not context_info or not isinstance(context_info, dict):
                logger.warning("JSON inválido na resposta de análise com texto. Usando fallback.")
                context_info = fallback_info.copy()
                context_info["layout_info"] = self._generate_layout_info(doc_structure)
                return context_info
            
            # Garantir campos obrigatórios
            self._ensure_required_fields(context_info, fallback_info)
            
            # Complementar com análise de estrutura
            if "layout_info" not in context_info or not context_info["layout_info"]:
                context_info["layout_info"] = self._generate_layout_info(doc_structure)
            
            # Adicionar nome do arquivo para uso posterior
            context_info["file_name"] = fallback_info.get("file_name", "")
            
            logger.info(f"Análise apenas com texto concluída com sucesso")
            return context_info
            
        except Exception as e:
            logger.exception(f"Erro ao analisar texto para contexto: {str(e)}")
            
            # Usar fallback com informações de layout baseadas na estrutura
            result = fallback_info.copy()
            result["layout_info"] = self._generate_layout_info(doc_structure)
            return result
    
    def _format_structure_hint(self, doc_structure: Dict[str, Any]) -> str:
        if not doc_structure:
            return "Não foi possível analisar a estrutura do documento."
        
        hints = []
        
        # Informação básica
        hints.append(f"- Documento com {doc_structure.get('page_count', '?')} página(s)")
        
        # Informação sobre tabelas
        if doc_structure.get('has_tables', False):
            tables = doc_structure.get('detected_tables', [])
            if tables:
                # Detectar padrão posicional em exemplos
                sample_lines = []
                for table in tables:
                    sample_lines.extend(table.get('sample_lines', []))
                
                if sample_lines:
                    # Verificar se há padrão posicional
                    has_positional_pattern = False
                    for line in sample_lines:
                        size_count = 0
                        for size in ['XS', 'S', 'M', 'L', 'XL', '38', '40', '42', '44', '46', '48']:
                            if size in line:
                                size_count += 1
                        
                        if size_count >= 3:
                            has_positional_pattern = True
                            break
                    
                    if has_positional_pattern:
                        hints.append("**PADRÃO POSICIONAL DETECTADO**:")
                        hints.append("- USAR MAPEAMENTO POSICIONAL: tamanho só existe se tem quantidade")
                        hints.append("- Células vazias = tamanho não disponível")
        
        return "\n".join(hints)
    
    def _generate_layout_info(self, doc_structure: Dict[str, Any]) -> Dict[str, Any]:
        layout_info = {
            "general_structure": "Tabular",
            "product_location": "Todo o documento",
            "product_identifier": "Código e nome do produto",
            "color_pattern": "Múltiplas cores por produto em linhas separadas",
            "size_pattern": "Tamanhos distribuídos em colunas",
            "quantity_format": "Número inteiro em cada célula da tabela",
            "price_format": "Valor numérico com separador decimal",
            "table_headers": [],
            "special_instructions": "Verificar cuidadosamente células vazias que indicam tamanho indisponível"
        }
        
        # Extrair cabeçalhos potenciais
        headers = doc_structure.get('potential_headers', [])
        if headers:
            header_texts = [h.get('text', '') for h in headers[:10]]  # Limitar a 10 cabeçalhos
            layout_info["table_headers"] = [h for h in header_texts if h]
        
        # Ajustar com base em tabelas detectadas
        if doc_structure.get('has_tables', False):
            tables = doc_structure.get('detected_tables', [])
            if tables:
                # Analisar estrutura a partir das amostras de linhas
                sample_lines = []
                for table in tables:
                    sample_lines.extend(table.get('sample_lines', []))
                
                if sample_lines:
                    # Detectar padrão de tamanhos (se tiver várias colunas alinhadas)
                    spaces_count = 0
                    for line in sample_lines:
                        spaces_count += len(re.findall(r'\s{3,}', line))
                    
                    avg_spaces = spaces_count / len(sample_lines) if sample_lines else 0
                    
                    if avg_spaces > 5:
                        layout_info["size_pattern"] = "Múltiplos tamanhos em colunas separadas"
                        layout_info["special_instructions"] += ". A tabela tem muitas colunas, provavelmente para diferentes tamanhos."
        
        return layout_info
    
    def _ensure_supplier_and_brand(self, context_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Garante que as informações de fornecedor e marca estão presentes e consistentes.
        Utiliza correspondência por similaridade para normalizar fornecedores.
        
        Args:
            context_info: Informações de contexto
            
        Returns:
            Dict: Informações de contexto com fornecedor e marca garantidos
        """
        filename = context_info.get("file_name", "")
        
        # ETAPA 1: Garantir que temos um fornecedor
        # Tentar extrair o fornecedor do nome do arquivo se não for encontrado no documento
        if not context_info.get("supplier") or context_info["supplier"] in ["", "Não identificado"]:
            # Tentar extrair do nome do arquivo
            if filename:
                # Lista de possíveis palavras a remover
                words_to_remove = ["nota", "encomenda", "pedido", "order", "orçamento", 
                                "fatura", "invoice", "pdf", "doc", "documento", 
                                "ficheiro", "file", "de", "do", "da", "das", "dos"]
                
                # Limpar o nome do arquivo
                clean_name = filename.lower()
                for word in words_to_remove:
                    clean_name = re.sub(r'\b' + word + r'\b', '', clean_name)
                
                clean_name = clean_name.replace("_", " ").replace("-", " ")
                
                # Dividir em palavras e remover espaços extras
                name_parts = [part.strip() for part in clean_name.split() if part.strip()]
                
                # Verificar contra lista de fornecedores conhecidos
                potential_supplier = None
                try:
                    # Verificar SUPPLIER_MAP apenas se estiver disponível
                    if SUPPLIER_MAP:
                        for supplier_code, supplier_name in SUPPLIER_MAP.items():
                            supplier_lower = supplier_name.lower()
                            if any(part in supplier_lower or supplier_lower in part for part in name_parts):
                                potential_supplier = supplier_name
                                break
                except:
                    logger.warning("SUPPLIER_MAP não disponível para verificação")
                
                # Se encontrou um fornecedor potencial
                if potential_supplier:
                    context_info["supplier"] = potential_supplier
                    logger.info(f"Fornecedor extraído do nome do arquivo: {potential_supplier}")
                elif name_parts:
                    # Usar a parte mais longa como potencial nome
                    longest_part = max(name_parts, key=len)
                    if len(longest_part) > 2:
                        context_info["supplier"] = longest_part.title()
                        logger.info(f"Usando parte do nome do arquivo como fornecedor: {longest_part.title()}")
                    else:
                        context_info["supplier"] = "Fornecedor não identificado"
                else:
                    context_info["supplier"] = "Fornecedor não identificado"
        
        # ETAPA 2: Normalizar o fornecedor extraído com correspondência por similaridade
        supplier = context_info.get("supplier", "")
        if supplier and supplier != "Fornecedor não identificado":
            try:
                # Usar a função match_supplier_name do supplier_utils.py
                normalized_supplier = match_supplier_name(supplier)
                
                if normalized_supplier != supplier:
                    logger.info(f"Fornecedor normalizado de '{supplier}' para '{normalized_supplier}'")
                    context_info["supplier"] = normalized_supplier
            except Exception as e:
                logger.warning(f"Erro ao normalizar fornecedor: {str(e)}")
        
        # ETAPA 3: Garantir que a marca está presente
        # A regra mais importante: se não tiver marca definida, usar o fornecedor como marca
        if not context_info.get("brand") or context_info["brand"] in ["", "Não identificado"]:
            # Usar o fornecedor como marca
            if context_info.get("supplier") and context_info["supplier"] not in ["", "Fornecedor não identificado"]:
                context_info["brand"] = context_info["supplier"]
                logger.info(f"Marca definida como fornecedor: {context_info['brand']}")
            else:
                context_info["brand"] = "Marca não identificada"
        
        # ETAPA 4: Garantir consistência
        # Se temos uma marca mas não temos fornecedor, usar a marca como fornecedor
        if (not context_info.get("supplier") or context_info["supplier"] in ["", "Fornecedor não identificado"]) and context_info.get("brand"):
            context_info["supplier"] = context_info["brand"]
            
            # Tentar normalizar a marca como fornecedor
            try:
                normalized_supplier = match_supplier_name(context_info["supplier"])
                
                if normalized_supplier != context_info["supplier"]:
                    logger.info(f"Marca usada como fornecedor e normalizada de '{context_info['supplier']}' para '{normalized_supplier}'")
                    context_info["supplier"] = normalized_supplier
            except Exception as e:
                logger.warning(f"Erro ao normalizar marca como fornecedor: {str(e)}")
            
            logger.info(f"Fornecedor definido como marca: {context_info['supplier']}")
        
        return context_info
    
    def _ensure_required_fields(self, context_info: Dict[str, Any], fallback_info: Dict[str, Any]) -> None:
        """
        Garante que os campos obrigatórios estejam presentes no contexto,
        preenchendo com valores de fallback se necessário
        
        Args:
            context_info: Informações de contexto extraídas
            fallback_info: Informações de fallback para usar se campos estiverem ausentes
        """
        # Lista de campos obrigatórios
        required_fields = [
            "document_type", "supplier", "brand", "customer",
            "reference_number", "date", "season", "file_name"
        ]
        
        # Garantir que todos os campos obrigatórios existam
        for field in required_fields:
            if field not in context_info or not context_info[field]:
                context_info[field] = fallback_info.get(field, "")
        
        # Garantir que layout_info existe
        if "layout_info" not in context_info or not context_info["layout_info"]:
            context_info["layout_info"] = fallback_info.get("layout_info", {})

    def _ensure_supplier_and_brand(self, context_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Garante que as informações de fornecedor e marca estão presentes e consistentes
        
        Args:
            context_info: Informações de contexto
            
        Returns:
            Dict: Informações de contexto com fornecedor e marca garantidos
        """
        filename = context_info.get("file_name", "")
        
        # Tentar extrair o fornecedor do nome do arquivo se não for encontrado no documento
        if not context_info.get("supplier") or context_info["supplier"] in ["", "Não identificado"]:
            # Tentar extrair do nome do arquivo
            if filename:
                # Lista de possíveis palavras a remover
                words_to_remove = ["nota", "encomenda", "pedido", "order", "orçamento", 
                                   "fatura", "invoice", "pdf", "doc", "documento", 
                                   "ficheiro", "file", "de", "do", "da", "das", "dos"]
                
                # Limpar o nome do arquivo
                clean_name = filename.lower()
                for word in words_to_remove:
                    clean_name = re.sub(r'\b' + word + r'\b', '', clean_name)
                
                clean_name = clean_name.replace("_", " ").replace("-", " ")
                
                # Dividir em palavras e remover espaços extras
                name_parts = [part.strip() for part in clean_name.split() if part.strip()]
                
                # Verificar contra lista de fornecedores conhecidos
                potential_supplier = None
                if hasattr(self, 'SUPPLIER_MAP') and self.SUPPLIER_MAP:
                    for supplier_code, supplier_name in self.SUPPLIER_MAP.items():
                        supplier_lower = supplier_name.lower()
                        if any(part in supplier_lower or supplier_lower in part for part in name_parts):
                            potential_supplier = supplier_name
                            break
                
                # Se encontrou um fornecedor potencial
                if potential_supplier:
                    context_info["supplier"] = potential_supplier
                    logger.info(f"Fornecedor extraído do nome do arquivo: {potential_supplier}")
                elif name_parts:
                    # Usar a parte mais longa como potencial nome
                    longest_part = max(name_parts, key=len)
                    if len(longest_part) > 2:
                        context_info["supplier"] = longest_part.title()
                        logger.info(f"Usando parte do nome do arquivo como fornecedor: {longest_part.title()}")
                    else:
                        context_info["supplier"] = "Fornecedor não identificado"
                else:
                    context_info["supplier"] = "Fornecedor não identificado"
        
        # Verificar se o fornecedor está na lista de fornecedores conhecidos
        supplier = context_info.get("supplier", "")
        if supplier and hasattr(self, 'get_supplier_code') and callable(self.get_supplier_code):
            supplier_code = self.get_supplier_code(supplier)
            if supplier_code and supplier_code in self.SUPPLIER_MAP:
                # Usar o nome padronizado
                context_info["supplier"] = self.SUPPLIER_MAP[supplier_code]
        
        # Se não tiver marca, usar o fornecedor como marca
        if not context_info.get("brand") or context_info["brand"] in ["", "Não identificado"]:
            context_info["brand"] = context_info.get("supplier", "Marca não identificada")
        
        return context_info
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extrai um objeto JSON de um texto que pode conter código markdown
        
        Args:
            text: Texto potencialmente contendo JSON
            
        Returns:
            Dict ou None: Objeto JSON extraído ou None se falhar
        """
        try:
            # Verificar se tem bloco de código JSON
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, text)
            
            if matches:
                # Usar o primeiro bloco JSON encontrado
                return json.loads(matches[0])
            
            # Tentar interpretar a string inteira como JSON
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Tentar encontrar qualquer objeto JSON na string
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
            
            # Se chegou aqui, não encontrou JSON válido
            return None
            
        except Exception as e:
            logger.warning(f"Erro ao extrair JSON do texto: {str(e)}")
            return None
    
    def format_context_for_extraction(self, context_info: Dict[str, Any]) -> str:
        """
        Formata as informações de contexto para uso pelo agente de extração
        
        Args:
            context_info: Informações de contexto extraídas
            
        Returns:
            str: Contexto formatado para o agente de extração
        """
        # Seções do contexto formatado
        sections = []
        
        # 1. Informações básicas do documento
        basic_info = []
        for key in ["document_type", "supplier", "brand", "customer", 
                   "reference_number", "date", "season"]:
            if key in context_info and context_info[key]:
                # Converter chaves snake_case para Título Legível
                readable_key = key.replace('_', ' ').title()
                basic_info.append(f"{readable_key}: {context_info[key]}")
        
        if basic_info:
            sections.append("## Informações do Documento")
            sections.append("\n".join(basic_info))
        
        # 2. Informações de layout
        if "layout_info" in context_info and context_info["layout_info"]:
            layout_info = context_info["layout_info"]
            sections.append("\n## Informações de Layout")
            
            for key, value in layout_info.items():
                if key == "table_headers" and isinstance(value, list):
                    if value:
                        headers_str = ", ".join([f'"{h}"' for h in value])
                        sections.append(f"Cabeçalhos Detectados: {headers_str}")
                elif value:
                    # Converter chave para formato legível
                    readable_key = key.replace('_', ' ').title()
                    sections.append(f"{readable_key}: {value}")
        
        # 3. Instruções específicas para o extrator
        sections.append("\n## Instruções para Extração")
        
        # Adicionar instruções baseadas no layout
        if "layout_info" in context_info and context_info["layout_info"]:
            layout = context_info["layout_info"]
            
            # Instruções sobre identificação de produtos
            product_identifier = layout.get("product_identifier", "")
            if product_identifier:
                sections.append(f"- Identificação de Produtos: {product_identifier}")
            
            # Instruções sobre padrão de cores
            color_pattern = layout.get("color_pattern", "")
            if color_pattern:
                sections.append(f"- Padrão de Cores: {color_pattern}")
            
            # Instruções sobre tamanhos
            size_pattern = layout.get("size_pattern", "")
            if size_pattern:
                sections.append(f"- Padrão de Tamanhos: {size_pattern}")
            
            # Instruções especiais
            special_instructions = layout.get("special_instructions", "")
            if special_instructions:
                sections.append(f"- Atenção Especial: {special_instructions}")
        
        # Adicionar instruções gerais
        sections.append("- Extrair apenas produtos com dados completos (código, cores, tamanhos)")
        sections.append("- Ignorar linhas de totais ou resumos")
        sections.append("- Verificar células vazias que indicam tamanhos indisponíveis")
        
        # Juntar todas as seções com quebras de linha duplas entre elas
        return "\n\n".join(sections)