import json
import os
import glob
import pandas as pd
import warnings
import traceback
from pathlib import Path
from io import StringIO
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, Any

# --- Configuração do Modelo (google-t5/t5-small) ---
MODEL_NAME = "google-t5/t5-small" 
TABLE_PIPELINE = None

def inicializar_modelo_tabela():
    global TABLE_PIPELINE
    if TABLE_PIPELINE is None:
        try:
            print(f" [Etapa 4] Carregando modelo '{MODEL_NAME}' (pode levar um momento)...")
            
            model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
            tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
            
            # --- ESTA É A LINHA CORRIGIDA ---
            # O modelo T5 padrão usa a tarefa "text2text-generation".
            TABLE_PIPELINE = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                framework="pt"
            )
            print(" [Etapa 4] Modelo de Tabela-para-Texto carregado.")
            
        except Exception as e:
            print(f"!!! ERRO [Etapa 4] Não foi possível carregar o modelo '{MODEL_NAME}'.")
            print(f"    Erro: {e}")
            raise

def converter_html_para_texto(html_tabela: str) -> str:
    if TABLE_PIPELINE is None:
        inicializar_modelo_tabela()
        
    try:
        df = pd.read_html(StringIO(html_tabela))[0]
        tabela_string = df.to_string() 
        prompt = f"summarize the following table: {tabela_string}"
        resultado = TABLE_PIPELINE(prompt, max_length=150, min_length=10, truncation=True)
        
        if resultado and len(resultado) > 0:
            return resultado[0]['generated_text']
        else:
            return "Não foi possível gerar um resumo para esta tabela."

    except ImportError:
        print("!!! ERRO [Etapa 4] 'lxml' ou 'html5lib' não encontrados.")
        return "Erro de dependência na conversão de HTML."
    except Exception as e:
        print(f"Aviso: Falha ao processar tabela HTML: {e}")
        return f"Tabela mal formatada: {str(e)[:100]}"

def enriquecer_tabelas(jsonl_directory: str):
    print(f"\n--- Iniciando Etapa 4: Enriquecimento de Tabelas ---")
    jsonl_files = glob.glob(os.path.join(jsonl_directory, "*.jsonl"))
    if not jsonl_files:
        print(f" [Etapa 4] Nenhum arquivo .jsonl encontrado em {jsonl_directory} para enriquecer.")
        return
    try:
        inicializar_modelo_tabela()
    except Exception as e:
        return 
    for file_path in jsonl_files:
        print(f" [Etapa 4] Processando arquivo: {os.path.basename(file_path)}")
        temp_file_path = file_path + ".temp"
        linhas_processadas = 0
        tabelas_convertidas = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f_in, \
                 open(temp_file_path, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    linhas_processadas += 1
                    try:
                        data = json.loads(line)
                        if data.get("tipo") != "tabela" or data.get("tabela_resumo"):
                            f_out.write(line)
                            continue
                        html_tabela = data.get("tabela_dados")
                        if not html_tabela or not isinstance(html_tabela, str) or "<table" not in html_tabela:
                            f_out.write(line)
                            continue
                        resumo_tabela = converter_html_para_texto(html_tabela)
                        data["tabela_resumo"] = resumo_tabela
                        data["texto_normalizado"] = resumo_tabela
                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                        tabelas_convertidas += 1
                    except json.JSONDecodeError:
                        f_out.write(line) 
            os.replace(temp_file_path, file_path)
            print(f"   -> Concluído. {tabelas_convertidas} tabelas convertidas em {linhas_processadas} linhas.")
        except Exception as e:
            print(f"!!! ERRO FATAL [Etapa 4] ao processar {file_path}: {e}")
            traceback.print_exc()
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    print("--- Etapa 4 (Tabelas) Concluída ---")