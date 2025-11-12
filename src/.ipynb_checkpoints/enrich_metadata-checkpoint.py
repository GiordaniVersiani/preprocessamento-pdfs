import json
import os
import re
from datetime import datetime
from pypdf import PdfReader

def enrich_metadata(structured_data: dict, pdf_path: str = None, custom_metadata: dict = None) -> dict:
    """
    Enriquece a estrutura do documento com metadados, extraindo-os do PDF e combinando com metadados personalizados.

    Args:
        structured_data (dict): A estrutura do documento processada.
        pdf_path (str, optional): O caminho para o arquivo PDF original, usado para inferir nome do documento e extrair metadados.
        custom_metadata (dict, optional): Um dicionário com metadados personalizados para adicionar ou sobrescrever.
                                         Pode incluir: doc_id, nome_doc, versao, data_publicacao,
                                         pagina_inicial, pagina_final.

    Returns:
        dict: A estrutura do documento enriquecida com metadados.
    """
    enriched_data = structured_data.copy()

    metadata = {
        "doc_id": "",
        "nome_doc": "Documento Desconhecido",
        "versao": "1ª versão",
        "data_publicacao": datetime.now().strftime("%Y-%m-%d"),
        "pagina_inicial": 1,
        "pagina_final": None,
    }

    if pdf_path and os.path.exists(pdf_path):
        try:
            reader = PdfReader(pdf_path)
            pdf_info = reader.metadata
            
            if pdf_info:
                if "/Title" in pdf_info: metadata["nome_doc"] = pdf_info["/Title"]
                elif "/Subject" in pdf_info: metadata["nome_doc"] = pdf_info["/Subject"]
                else: metadata["nome_doc"] = os.path.basename(pdf_path)

                if "/CreationDate" in pdf_info:
                    date_str = pdf_info["/CreationDate"]
                    match = re.search(r"\d{4}(\d{2})(\d{2})", date_str)
                    if match: metadata["data_publicacao"] = f"{match.group(0)[:4]}-{match.group(1)}-{match.group(2)}"
                elif "/ModDate" in pdf_info:
                    date_str = pdf_info["/ModDate"]
                    match = re.search(r"\d{4}(\d{2})(\d{2})", date_str)
                    if match: metadata["data_publicacao"] = f"{match.group(0)[:4]}-{match.group(1)}-{match.group(2)}"

            metadata["pagina_final"] = len(reader.pages)

        except Exception as e:
            print(f"Aviso: Não foi possível extrair metadados do PDF {pdf_path} usando pypdf: {e}")
            metadata["nome_doc"] = os.path.basename(pdf_path)
    else:
        metadata["nome_doc"] = os.path.basename(pdf_path) if pdf_path else "Documento Desconhecido"

    if custom_metadata:
        metadata.update(custom_metadata)

    final_enriched_data = metadata
    final_enriched_data.update(enriched_data)

    return final_enriched_data