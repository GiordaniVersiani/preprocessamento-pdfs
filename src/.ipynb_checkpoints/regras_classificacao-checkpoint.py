import re
from typing import Dict, Any

def atualizar_contexto_estrutural(contexto_atual: Dict[str, Any], tipo_elemento: str, texto_elemento: str) -> Dict[str, Any]:
    """
    Atualiza o dicionário de contexto estrutural (Capítulo, Seção, etc.) 
    com base no tipo de bloco encontrado.
    """
    novo_contexto = contexto_atual.copy()
    
    if tipo_elemento == 'titulo_1': 
        novo_contexto['capitulo'] = texto_elemento
        novo_contexto['secao'] = None
        novo_contexto['subsecao'] = None
    elif tipo_elemento == 'titulo_2':
        novo_contexto['secao'] = texto_elemento
        novo_contexto['subsecao'] = None
    elif tipo_elemento == 'titulo_3':
        novo_contexto['subsecao'] = texto_elemento
        
    return novo_contexto

def classificar_elemento_unstructured(el) -> str:
    """
    Traduz a categoria do 'unstructured' (ex: 'Title', 'Text')
    para o nosso 'tipo' de bloco (ex: 'titulo_1', 'paragrafo', 'ignorar').
    """
    categoria = str(el.category)
    
    # 1. O que queremos IGNORAR
    if categoria in ['Header', 'Footer', 'PageNumber']:
        return 'ignorar'

    # 2. O que queremos CLASSIFICAR
    if categoria == 'Title':
        texto = el.text
        if re.match(r"^CAPÍTULO\s[IVXLC]+", texto):
             return "titulo_1"
        if re.match(r"^\d\.\d\s", texto):
             return "titulo_2"
        return "titulo_desconhecido" 
    
    if categoria == 'Table':
        return 'tabela'
    
    if categoria == 'ListItem':
        return 'item_lista'

    # 3. O que queremos MANTER (O Padrão)
    return 'paragrafo'