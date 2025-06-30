# app/data/reference_data.py

# Mapeamento de códigos de cor para nomes de cor
COLOR_MAP = {
    "001": "Branco",
    "002": "Vermelho",
    "003": "Verde",
    "004": "Castanho",
    "005": "Amarelo",
    "006": "Lilás",
    "007": "Rosa",
    "008": "Azul",
    "009": "Laranja",
    "010": "Preto",
    "011": "Cinza",
    "012": "Bege",
    "013": "Camel",
    "014": "Coral",
    "015": "Chocolate",
    "016": "Creme",
    "017": "Dourado",
    "018": "Gelo",
    "019": "Grená",
    "020": "Turquesa",
    "021": "Prata",
    "022": "Púrpura",
    "023": "Roxo",
    "024": "Violeta",
    "025": "Salmão",
    "026": "Bronze",
    "027": "Cereja",
    "028": "Fucsia",
    "029": "Marfim",
    "030": "Tijolo"
}

COLOR_CODE_MAP = {v: k for k, v in COLOR_MAP.items()}

SIZE_MAP = {
    "XS": "001",
    "S": "002",
    "M": "003",
    "L": "004",
    "XL": "005",
    "XXL": "006",
    "XXXL": "007",
    "31": "008",
    "32": "009",
    "33": "010",
    "34": "011",
    "35": "012",
    "36": "013",
    "37": "014",
    "38": "015",
    "39": "016",
    "40": "017",
    "42": "018",
    "44": "019",
    "46": "020",
    "48": "021",
    "50": "022",
    "52": "023",
    "54": "024",
    "56": "025",
    "58": "026",
    "2": "027",
    "4": "028",
    "6": "029",
    "8": "030",
    "10": "031",
    "12": "032",
    "14": "033",
    "16": "034",
    "TU": "035",
    "28": "036",
    "29": "037",
    "30": "038",
    "26": "039",
    "27": "040",
    "39-40": "041",
    "41-42": "042",
    "43-44": "043",
    "41": "044",
    "43": "045"
}

# Lista de categorias
CATEGORIES = [
    "CAMISAS",
    "CASACOS",
    "VESTIDOS",
    "BLUSAS",
    "CALÇAS",
    "CALÇÃO",
    "MALHAS",
    "SAIAS",
    "T-SHIRTS",
    "POLOS",
    "JEANS",
    "SWEATSHIRTS",
    "BLAZERS E FATOS",
    "BLUSÕES E PARKAS",
    "CALÇADO",
    "TOP",
    "ACESSÓRIOS"
]

# Lista de marcas
BRANDS = [
    "HUGO BOSS",
    "PAUL & SHARK",
    "LIU.JO",
    "LOVE MOSCHINO",
    "BRAX",
    "MEYER",
    "TWINSET",
    "WEEKEND/MAXMARA",
    "MARELLA",
    "TOMMY HILFIGER",
    "GANT",
    "LEBEK",
    "BOUTIQUE MOSCHINO",
    "COCCINELLI",
    "DIELMAR",
    "ESCORPION",
    "NAULOVER",
    "MICAELA LUISA",
    "RALPH LAUREN",
    "KOTTAS & MANTSIOS",
    "CORTY",
    "BENNETT"
]

# Mapeamento de fornecedores com códigos e marcações
SUPPLIER_DATA = {
    "01": {"nome": "GANT", "marcacao": 1.65},
    "02": {"nome": "HUGO BOSS", "marcacao": 2.73},
    "03": {"nome": "VANDOMA- António M. Sousa", "marcacao": 4.0},
    "04": {"nome": "ESCORPION", "marcacao": 2.6},
    "05": {"nome": "MAXMARA", "marcacao": 2.6},
    "06": {"nome": "MARELLA", "marcacao": 2.7},
    "07": {"nome": "PAUL & SHARK- DAMA", "marcacao": 2.5},
    "08": {"nome": "MARCOTEX", "marcacao": 2.7},
    "09": {"nome": "FLORENTINO COLECCION", "marcacao": 2.6},
    "10": {"nome": "MEYER- HOSEN", "marcacao": 2.7},
    "11": {"nome": "DECENIO", "marcacao": 2.5},
    "12": {"nome": "MIGUEL BELLIDO", "marcacao": 3.0},
    "13": {"nome": "LINDENMANN", "marcacao": 3.0},
    "15": {"nome": "TOMMY HILFIGER", "marcacao": 2.45},
    "17": {"nome": "LEBEK FASHION", "marcacao": 3.0},
    "18": {"nome": "NAULOVER", "marcacao": 2.6},
    "20": {"nome": "DIELMAR", "marcacao": 2.6},
    "22": {"nome": "RALPH LAUREN", "marcacao": 2.7},
    "23": {"nome": "LVX", "marcacao": None},
    "24": {"nome": "LIU.JO", "marcacao": 2.6},
    "25": {"nome": "PENNYBLACK", "marcacao": 2.6},
    "26": {"nome": "CB BENNETT", "marcacao": 3.0},
    "27": {"nome": "TOMMY HILFIGER- ACESS", "marcacao": None},
    "28": {"nome": "REFIVE", "marcacao": None},
    "29": {"nome": "MICHAELA LOUISA", "marcacao": None},
    "30": {"nome": "COCCINELLE", "marcacao": 2.61},
    "31": {"nome": "LOVE MOSCHINO- SINV", "marcacao": 2.6},
    "32": {"nome": "BOUTIQUE MOSCHINO-AEFFE", "marcacao": 2.6},
    "33": {"nome": "TWINSET- ANDRÉ COSTA", "marcacao": 2.5},
    "35": {"nome": "MORPHOPOLIS OFICINA DE ARQUITECTURA", "marcacao": None}
}

# Mapeamentos simplificados para consulta rápida
SUPPLIER_MAP = {k: v["nome"] for k, v in SUPPLIER_DATA.items()}
SUPPLIER_CODE_MAP = {v: k for k, v in SUPPLIER_MAP.items()}
MARKUP_MAP = {k: v["marcacao"] for k, v in SUPPLIER_DATA.items() if v["marcacao"] is not None}

def get_color_name(color_code):
    """
    Retorna o nome da cor baseado no código
    
    Args:
        color_code: Código da cor
        
    Returns:
        str: Nome da cor ou o próprio código se não encontrado
    """
    if color_code in COLOR_MAP:
        return COLOR_MAP[color_code]
    return color_code

def get_color_code(color_name):
    """
    Retorna o código da cor baseado no nome
    
    Args:
        color_name: Nome da cor
        
    Returns:
        str: Código da cor ou None se não encontrado
    """
    color_name_upper = color_name.upper()
    # Busca exata
    if color_name in COLOR_CODE_MAP:
        return COLOR_CODE_MAP[color_name]
    
    # Busca por correspondência parcial
    for name, code in COLOR_CODE_MAP.items():
        if name.upper() in color_name_upper or color_name_upper in name.upper():
            return code
    
    return None

def get_size_code(size):
    """
    Retorna o código associado a um tamanho
    
    Args:
        size: Tamanho do produto
        
    Returns:
        str: Código do tamanho ou None se não encontrado
    """
    size_upper = str(size).upper()
    if size_upper in SIZE_MAP:
        return SIZE_MAP[size_upper]
    
    # Busca por correspondência parcial
    for s, code in SIZE_MAP.items():
        if s.upper() == size_upper:
            return code
    
    return None

def get_category(category_name):
    """
    Verifica se uma categoria existe e retorna-a padronizada
    
    Args:
        category_name: Nome da categoria a verificar
        
    Returns:
        str: Nome da categoria padronizado ou None se não existir
    """
    category_upper = str(category_name).upper()
    
    if category_upper in CATEGORIES:
        return category_upper
    
    # Busca parcial
    for category in CATEGORIES:
        if category in category_upper or category_upper in category:
            return category
    
    return None

def get_supplier_code(supplier_name):
    """
    Retorna o código do fornecedor baseado no nome
    
    Args:
        supplier_name: Nome do fornecedor
        
    Returns:
        str: Código do fornecedor ou None se não encontrado
    """
    supplier_upper = supplier_name.upper()
    
    # Busca exata
    if supplier_name in SUPPLIER_CODE_MAP:
        return SUPPLIER_CODE_MAP[supplier_name]
    
    # Busca parcial
    for name, code in SUPPLIER_CODE_MAP.items():
        if name.upper() in supplier_upper or supplier_upper in name.upper():
            return code
    
    return None

def get_supplier_by_code(code):
    """
    Retorna o nome do fornecedor baseado no código
    
    Args:
        code: Código do fornecedor
        
    Returns:
        str: Nome do fornecedor ou None se não encontrado
    """
    code_str = str(code).zfill(2) if len(str(code)) == 1 else str(code)
    
    if code_str in SUPPLIER_MAP:
        return SUPPLIER_MAP[code_str]
    
    return None

def get_markup(supplier_code):
    """
    Retorna o valor de marcação (margem) para um fornecedor
    
    Args:
        supplier_code: Código do fornecedor
        
    Returns:
        float: Valor da marcação ou None se não encontrado
    """
    code_str = str(supplier_code).zfill(2) if len(str(supplier_code)) == 1 else str(supplier_code)
    
    if code_str in SUPPLIER_DATA:
        return SUPPLIER_DATA[code_str]["marcacao"]
    
    return None

def get_brand_categories():
    """
    Retorna a lista de categorias disponíveis
    
    Returns:
        list: Lista de categorias
    """
    return CATEGORIES

def get_brand_names():
    """
    Retorna a lista de marcas disponíveis
    
    Returns:
        list: Lista de marcas
    """
    return BRANDS

def get_suppliers():
    """
    Retorna a lista de fornecedores disponíveis
    
    Returns:
        dict: Dicionário com códigos e nomes de fornecedores
    """
    return SUPPLIER_MAP

def normalize_color_name(color_name):
    """
    Normaliza um nome de cor para o padrão do sistema
    
    Args:
        color_name: Nome da cor a ser normalizada
        
    Returns:
        str: Nome da cor normalizado ou o original se não encontrado
    """
    # Tenta encontrar o código da cor
    color_code = get_color_code(color_name)
    
    # Se encontrou o código, retorna o nome padronizado
    if color_code:
        return get_color_name(color_code)
    
    return color_name

def normalize_size(size):
    """
    Normaliza um tamanho para o padrão do sistema
    
    Args:
        size: Tamanho a ser normalizado
        
    Returns:
        str: Tamanho normalizado ou o original se não encontrado
    """
    size_upper = str(size).upper().strip()
    
    # Verifica se o tamanho está no mapeamento
    if size_upper in SIZE_MAP:
        return size_upper
    
    # Algumas normalizações específicas
    if size_upper == "EXTRA SMALL":
        return "XS"
    elif size_upper == "SMALL":
        return "S"
    elif size_upper == "MEDIUM":
        return "M"
    elif size_upper == "LARGE":
        return "L"
    elif size_upper == "EXTRA LARGE":
        return "XL"
    elif size_upper in ["2XL", "XX LARGE", "EXTRA EXTRA LARGE"]:
        return "XXL"
    elif size_upper in ["3XL", "XXX LARGE"]:
        return "XXXL"
    
    return size