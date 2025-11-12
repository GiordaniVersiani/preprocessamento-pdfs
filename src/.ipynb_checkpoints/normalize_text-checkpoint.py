import re
import string
from unidecode import unidecode

def normalize_text(
    raw_text: str,
    acronyms: dict = None,
    standardization_map: dict = None
) -> str:
    """
    Normaliza o texto, realizando uma série de etapas de limpeza e padronização.

    Esta função é o foco e foi aprimorada para incluir:
    1.  Remoção de quebras de linha indevidas (hifenização no final da linha).
    2.  Expansão de siglas de forma mais robusta.
    3.  Padronização de terminologia específica (ex: 'discente' -> 'aluno').
    4.  Conversão para minúsculas e remoção de acentos.
    5.  Remoção de pontuação.
    6.  Normalização de espaços em branco.

    Args:
        raw_text (str): O texto bruto a ser normalizado.
        acronyms (dict): Dicionário com siglas a serem expandidas.
                         Ex: {"PPC": "Projeto Pedagógico de Curso"}.
        standardization_map (dict): Dicionário para padronizar termos.
                                     Ex: {"discente": "aluno", "docente": "professor"}.

    Returns:
        str: O texto normalizado e limpo.
    """
    if acronyms is None:
        acronyms = {}
    if standardization_map is None:
        standardization_map = {}

    # Etapa 1: Correção de hifenização e quebras de linha
    # Remove o hífen no final de uma linha e junta a palavra com a continuação na próxima linha.
    # Ex: "gradua-\nção" -> "graduação"
    normalized_text = re.sub(r'-\n\s*', '', raw_text)
    # Substitui quebras de linha por espaço para unificar o texto em um fluxo contínuo.
    normalized_text = normalized_text.replace('\n', ' ')

    # Etapa 2: Expansão de siglas e padronização de termos
    # Unifica os dois dicionários para uma única passada de substituição.
    # A padronização vem primeiro para evitar conflitos com siglas.
    combined_map = {**standardization_map, **acronyms}

    # Ordena as chaves por comprimento, da maior para a menor, para evitar substituições parciais
    # Ex: substituir "BCC" antes de uma hipotética sigla "BC".
    sorted_keys = sorted(combined_map.keys(), key=len, reverse=True)

    for key in sorted_keys:
        # Usa \b (word boundary) para garantir que estamos substituindo a palavra/sigla inteira.
        # re.escape trata caracteres especiais que possam existir na chave (ex: pontos em U.S.A.).
        pattern = r'\b' + re.escape(key) + r'\b'
        # Usa re.IGNORECASE para capturar variações como 'ppc' ou 'PPC'.
        normalized_text = re.sub(pattern, combined_map[key], normalized_text, flags=re.IGNORECASE)

    # Etapa 3: Conversão para minúsculas
    # Padroniza todo o texto para caixa baixa, facilitando comparações e processamento.
    normalized_text = normalized_text.lower()

    # Etapa 4: Remoção de acentos
    # Translitera caracteres acentuados para suas versões não acentuadas (ex: 'ção' -> 'cao').
    normalized_text = unidecode(normalized_text)

    # Etapa 5: Remoção de pontuação
    # Cria um tradutor que remove todos os caracteres de pontuação definidos em string.punctuation.
    # Esta é uma abordagem "agressiva". Em alguns cenários, pode ser útil manter hífens ou outros sinais.
    translator = str.maketrans('', '', string.punctuation)
    normalized_text = normalized_text.translate(translator)

    # Etapa 6: Normalização de espaços em branco
    # Substitui múltiplos espaços, tabulações, etc., por um único espaço e remove espaços no início/fim.
    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()

    return normalized_text
