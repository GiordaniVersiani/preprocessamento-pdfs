import re
import json
import pdfplumber

def detect_structure(pdf_path: str, normalized_text: str) -> dict:
    """
    Detecta a estrutura de um texto normalizado, identificando capítulos, seções, artigos, parágrafos e incisos,
    utilizando análise de layout do PDF para maior precisão.

    Args:
        pdf_path (str): O caminho para o arquivo PDF original.
        normalized_text (str): O texto normalizado a ser analisado (usado como fallback ou para validação).

    Returns:
        dict: Um dicionário representando a estrutura do documento.
              Exemplo:
              {
                "estrutura": [
                  {
                    "tipo": "capitulo",
                    "titulo": "TÍTULO II",
                    "secoes": [
                      {
                        "tipo": "secao",
                        "titulo": "CAPÍTULO III",
                        "artigos": [
                          {
                            "tipo": "artigo",
                            "titulo": "Art. 21",
                            "paragrafos": [
                              {
                                "numero": "§ 6º",
                                "texto": "As disciplinas eletivas..."
                              }
                            ]
                          }
                        ]
                      }
                    ]
                  }
                ]
              }
    """
    structure = {"estrutura": []}
    current_chapter = None
    current_section = None
    current_article = None

    # Usar pdfplumber para extrair texto com informações de layout
    text_with_layout = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_with_layout.extend(page_text.split('\n'))

    if not text_with_layout:
        text_with_layout = normalized_text.split('\n')

    for line_raw in text_with_layout:
        line = line_raw.strip()
        if not line:
            continue

        match_capitulo = re.match(r"^(título [ivxlcdm]+)", line, re.IGNORECASE)
        if match_capitulo:
            capitulo_titulo = match_capitulo.group(1).upper()
            current_chapter = {"tipo": "capitulo", "titulo": capitulo_titulo, "secoes": [], "artigos": [], "paragrafos": []}
            structure["estrutura"].append(current_chapter)
            current_section = None
            current_article = None
            continue

        match_secao = re.match(r"^(capítulo [ivxlcdm]+)", line, re.IGNORECASE)
        if match_secao:
            secao_titulo = match_secao.group(1).upper()
            new_section = {"tipo": "secao", "titulo": secao_titulo, "artigos": [], "paragrafos": []}
            if current_chapter:
                current_chapter["secoes"].append(new_section)
                current_section = new_section
            else:
                structure["estrutura"].append(new_section)
                current_section = new_section
            current_article = None
            continue

        match_artigo = re.match(r"^(art\.?\s*\d+º?)(.*)", line, re.IGNORECASE)
        if match_artigo:
            artigo_titulo = match_artigo.group(1).capitalize()
            artigo_texto_inicial = match_artigo.group(2).strip()
            new_article = {"tipo": "artigo", "titulo": artigo_titulo, "paragrafos": []}
            if artigo_texto_inicial:
                new_article["paragrafos"].append({"numero": None, "texto": artigo_texto_inicial})

            if current_section:
                current_section["artigos"].append(new_article)
                current_article = new_article
            elif current_chapter:
                current_chapter["artigos"].append(new_article)
                current_article = new_article
            else:
                structure["estrutura"].append(new_article)
                current_article = new_article
            continue

        match_paragrafo = re.match(r"^(§\s*\d+º|parágrafo único|\b[ivxlcdm]+\s*[-–—])\s*(.*)", line, re.IGNORECASE)
        if match_paragrafo:
            paragraph_number = match_paragrafo.group(1).strip()
            paragraph_text = match_paragrafo.group(2).strip()
            if current_article:
                current_article["paragrafos"].append({"numero": paragraph_number, "texto": paragraph_text})
            elif current_section:
                current_section["paragrafos"].append({"numero": paragraph_number, "texto": paragraph_text})
            elif current_chapter:
                current_chapter["paragrafos"].append({"numero": paragraph_number, "texto": paragraph_text})
            else:
                structure["estrutura"].append({"tipo": "paragrafo", "numero": paragraph_number, "texto": paragraph_text})
            continue

        if current_article and current_article["paragrafos"]:
            current_article["paragrafos"][-1]["texto"] += " " + line
        elif current_article:
            current_article["paragrafos"].append({"numero": None, "texto": line})
        elif current_section and current_section["paragrafos"]:
            current_section["paragrafos"][-1]["texto"] += " " + line
        elif current_section:
            current_section["paragrafos"].append({"numero": None, "texto": line})
        elif current_chapter and current_chapter["paragrafos"]:
            current_chapter["paragrafos"][-1]["texto"] += " " + line
        elif current_chapter:
            current_chapter["paragrafos"].append({"numero": None, "texto": line})
        else:
            # Se não há estrutura pai, adiciona como um parágrafo de nível superior
            if structure["estrutura"] and structure["estrutura"][-1].get("tipo") == "paragrafo" and structure["estrutura"][-1].get("numero") is None:
                structure["estrutura"][-1]["texto"] += " " + line
            else:
                structure["estrutura"].append({"tipo": "paragrafo", "numero": None, "texto": line})

    return structure