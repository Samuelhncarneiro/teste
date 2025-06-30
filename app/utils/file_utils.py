# app/utils/file_utils.py
import os
import fitz  # PyMuPDF
import logging
from PIL import Image
from typing import List, Optional

logger = logging.getLogger(__name__)

def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 150, pages: Optional[List[int]] = None) -> List[str]:
    """
    Converte um PDF em imagens, uma por página.
    
    Args:
        pdf_path: Caminho para o arquivo PDF
        output_dir: Diretório onde as imagens serão salvas
        dpi: Resolução das imagens em DPI
        pages: Lista opcional de índices de páginas a converter (0-indexed). Se None, converte todas.
    
    Returns:
        List[str]: Lista de caminhos para as imagens geradas
    """
    try:
        # Abrir o documento PDF
        pdf_document = fitz.open(pdf_path)
        
        image_paths = []
        
        # Determinar quais páginas converter
        if pages is None:
            page_indices = range(len(pdf_document))
        else:
            page_indices = [p for p in pages if 0 <= p < len(pdf_document)]
        
        # Iterar por cada página
        for page_idx in page_indices:
            page = pdf_document.load_page(page_idx)
            
            # Ajustar zoom com base no DPI (2.0 = 192 DPI, 1.5 = 144 DPI)
            zoom_factor = dpi / 96  # 96 DPI é o padrão
            
            # Renderizar página como imagem com zoom
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
            
            # Salvar imagem
            output_path = os.path.join(output_dir, f"{os.path.basename(pdf_path)}_page_{page_idx+1}.png")
            pix.save(output_path)
            image_paths.append(output_path)
            
        return image_paths
    
    except Exception as e:
        logger.error(f"Erro ao converter PDF para imagens: {str(e)}")
        raise

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrai texto de um arquivo PDF."""
    try:
        pdf_document = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
            
        return text
    except Exception as e:
        logger.error(f"Erro ao extrair texto do PDF {pdf_path}: {str(e)}")
        return ""

def optimize_image(image_path: str, output_dir: str, max_dimension: int = 1200, quality: int = 85) -> str:
    """Otimiza uma imagem para processamento de visão computacional."""
    try:
        with Image.open(image_path) as img:
            # Converter para RGB se for RGBA
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            # Redimensionar se necessário
            if img.width > max_dimension or img.height > max_dimension:
                ratio = min(max_dimension / img.width, max_dimension / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                logger.info(f"Redimensionando imagem de {img.width}x{img.height} para {new_size[0]}x{new_size[1]}")
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Salvar com compressão otimizada
            output_path = os.path.join(output_dir, f"opt_{os.path.basename(image_path)}")
            img.save(output_path, "JPEG", quality=quality, optimize=True)
            
            return output_path
    except Exception as e:
        logger.error(f"Erro ao otimizar imagem: {str(e)}")
        return image_path