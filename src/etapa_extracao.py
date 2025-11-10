import fitz
import json
import warnings
import traceback
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace

# Importa as regras de classificação do outro arquivo .py
from regras_classificacao import (
    atualizar_contexto_estrutural,
    classificar_elemento_unstructured
)

# Ignora warnings de performance
warnings.filterwarnings("ignore", category=UserWarning, module='unstructured')

def processar_documento(pdf_path: str, jsonl_output_path: str):
    """
    Função principal que executa a Etapa 1 (Extração) e Etapa 3 (Estrutura).
    Lê um PDF, o processa com o unstructured, aplica as regras de classificação
    e salva o resultado em um arquivo JSONL.
    """
    pdf_nome = Path(pdf_path).name
    doc_id_base = Path(pdf_path).stem
    
    print(f"\n--- Processando: {pdf_nome} ---")
    
    try:
        # Pega o total de páginas (com fitz, rápido)
        total_paginas_doc = 0
        try:
            with fitz.open(pdf_path) as doc_fitz:
                total_paginas_doc = doc_fitz.page_count
        except Exception as e:
            print(f"Aviso: Não foi possível contar as páginas com fitz: {e}")

        #O CORAÇÃO DA EXTRAÇÃO (ETAPA 1)
        print("Iniciando particionamento com 'unstructured' (estratégia hi_res)...")
        elementos = partition_pdf(
            pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False
        )
        print("Particionamento concluído.")

        #INICIALIZA O ESTADO (ETAPA 3)
        metadados_doc = {
            "total_paginas": total_paginas_doc,
            "instituicao": None, "campus": None, "curso": None,
            "ano": None, "hash_arquivo": None
        }
        contexto_atual = {
            "capitulo": None, "secao": None, "subsecao": None, "artigo": None
        }
        
        #LOOP DE ESTRUTURAÇÃO E ESCRITA
        with open(jsonl_output_path, 'w', encoding='utf-8') as outfile:
            
            bloco_counter = 0
            for el in elementos:
                bloco_counter += 1
                
                # Aplica as regras de classificação (Etapa 3)
                tipo_bloco = classificar_elemento_unstructured(el)

                if tipo_bloco == 'ignorar':
                    continue
                
                texto_limpo = clean_extra_whitespace(el.text)
                if not texto_limpo: continue

                if "titulo" in tipo_bloco:
                    contexto_atual = atualizar_contexto_estrutural(contexto_atual, tipo_bloco, texto_limpo)
                
                # Monta o objeto JSONL (Schema Rico)
                bloco_id = f"{doc_id_base}-p{el.metadata.page_number or 0}-b{bloco_counter}"
                json_linha = {
                    "doc_id": doc_id_base, "nome_doc": pdf_nome,
                    "metadados_doc": metadados_doc,
                    "estrutura_global": {},
                    "bloco_id": bloco_id,
                    "pagina": el.metadata.page_number,
                    "tipo": tipo_bloco,
                    "contexto_estrutural": contexto_atual.copy(),
                    "texto_bruto": texto_limpo,
                    "texto_normalizado": None,
                    "tabela_dados": None, "tabela_resumo": None,
                    "info_bloco": {
                        "bbox": el.metadata.coordinates.to_dict() if el.metadata.coordinates else None,
                        "fonte_estilo": None 
                    }
                }
                
                if tipo_bloco == 'tabela':
                    bloco_id = f"{doc_id_base}-p{el.metadata.page_number or 0}-t{bloco_counter}"
                    json_linha['bloco_id'] = bloco_id
                    json_linha['tabela_dados'] = el.metadata.text_as_html
                    json_linha['texto_bruto'] = f"[PLACEHOLDER_TABELA: {texto_limpo[:50]}...]"

                outfile.write(json.dumps(json_linha, ensure_ascii=False) + "\n")

        print(f"Processamento concluído. JSONL salvo em: {jsonl_output_path}")

    except Exception as e:
        print(f"ERRO FATAL ao processar {pdf_nome}: {e}")
        traceback.print_exc()