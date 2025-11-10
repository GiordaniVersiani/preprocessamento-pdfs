import json
import os
import glob
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter # <-- Importa o fatiador

# --- Classe StatsTracker (Sem mudanças) ---
class StatsTracker:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.documentos_criados = 0
        self.blocos_vazios = 0
        self.blocos_ignorados = 0
        self.problemas = []
    def add_doc(self): self.documentos_criados += 1
    def log_bloco_vazio(self, contexto: Dict[str, Any]):
        self.blocos_vazios += 1
        msg = f"Bloco vazio (sem texto): {contexto.get('bloco_id', 'ID Desconhecido')}"
        self.problemas.append(msg)
    def log_bloco_ignorado(self, tipo: str): self.blocos_ignorados += 1
    def get_report(self) -> Dict[str, Any]:
        return {
            "arquivo": self.file_name,
            "documentos_criados_para_rag": self.documentos_criados,
            "blocos_ignorados (ruído)": self.blocos_ignorados,
            "inconsistencias": {"blocos_vazios_ou_placeholder": self.blocos_vazios,},
            "lista_de_problemas": self.problemas
        }

# --- FUNÇÃO PRINCIPAL DE CARREGAMENTO (COM "SMART CHUNKING") ---
def load_and_process_jsons(json_folder_path: str = "data/output_blocos/") -> Tuple[List[Document], Dict[str, Any]]:
    print(f"--- [RAG Loader v_SMART_CHUNK] Iniciando Tarefa 1: Carregamento e Fatiamento ---")
    
    jsonl_paths = glob.glob(os.path.join(json_folder_path, "*.jsonl"))
    if not jsonl_paths:
        print(f"!!! ERRO [RAG Loader]: Nenhum arquivo .jsonl encontrado em '{json_folder_path}'")
        return [], {}

    all_final_chunks = [] # A lista final de chunks que irão para o FAISS
    full_report = {}
    tipos_para_ignorar = ["ignorar", "rodape", "cabecalho", "titulo_desconhecido"]

    # --- ETAPA DE FATIAMENTO (CHUNKING) ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, # Tamanho do Pedaço
        chunk_overlap=400, # Sobreposição
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""] # Tenta quebrar em parágrafos
    )
    
    print(f" [RAG Loader] Processando e fatiando {len(jsonl_paths)} arquivos .jsonl...")
    for file_path in jsonl_paths:
        file_name = os.path.basename(file_path)
        print(f" [RAG Loader] Processando arquivo: {file_name}")
        tracker = StatsTracker(file_name)
        
        # Buffers para agrupar o texto do documento inteiro
        mega_document_texto_busca = ""
        mega_document_texto_resposta = ""
        base_metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                
                primeira_linha = True
                for line in f:
                    try:
                        linha_json = json.loads(line)
                        tipo_bloco = linha_json.get("tipo", "paragrafo")
                        if tipo_bloco in tipos_para_ignorar:
                            tracker.log_bloco_ignorado(tipo_bloco)
                            continue

                        # --- LÓGICA DE AGRUPAMENTO ---
                        
                        # 1. Pega o Texto de Busca (Normalizado ou Bruto da Tabela)
                        texto_busca = None
                        if tipo_bloco == 'tabela':

                            texto_busca = linha_json.get("tabela_resumo") 
                            if not texto_busca:
                                 texto_busca = linha_json.get("texto_bruto") # Fallback
                        else:
                            texto_busca = linha_json.get("texto_normalizado") # Usa o normalizado do parágrafo
                        
                        if not texto_busca:
                            texto_busca = linha_json.get("texto_bruto")
                        
                        # 2. Pega o Texto de Resposta (Sempre Bruto, mas para tabelas, use o resumo)
                        # --- MUDANÇA AQUI ---
                        if tipo_bloco == 'tabela':
                            texto_resposta = linha_json.get("tabela_resumo") # Usar o resumo para a resposta também
                            if not texto_resposta:
                                 texto_resposta = linha_json.get("texto_bruto") # Fallback
                        else:
                            texto_resposta = linha_json.get("texto_bruto")
                        
                        # Agora o 'if' abaixo vai funcionar,
                        # pois 'texto_busca' e 'texto_resposta' terão o conteúdo real da tabela
                        if not texto_busca or not texto_resposta or "[PLACEHOLDER_" in texto_busca:
                            tracker.log_bloco_vazio(linha_json)
                            continue
                        
                        # 4. Adiciona o texto aos "mega-documentos"
                        mega_document_texto_busca += texto_busca + "\n\n"
                        mega_document_texto_resposta += texto_resposta + "\n\n"
                        
                    except json.JSONDecodeError:
                        print(f"Aviso: Linha mal formatada pulada em {file_name}")
            
            # --- FIM DO ARQUIVO: FATIA OS "MEGA-DOCUMENTOS" ---
            if mega_document_texto_busca:
                # Fatiamos os dois textos (busca e resposta) em paralelo
                chunks_de_busca = text_splitter.split_text(mega_document_texto_busca)
                chunks_de_resposta = text_splitter.split_text(mega_document_texto_resposta)
                
                # Garante que temos o mesmo número de chunks
                if len(chunks_de_busca) != len(chunks_de_resposta):
                    # Se der errado, usa a busca como resposta (Fallback)
                    chunks_de_resposta = chunks_de_busca
                
                for i in range(len(chunks_de_busca)):
                    chunk_busca = chunks_de_busca[i]
                    chunk_resposta = chunks_de_resposta[i]
                    
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata["texto_bruto_resposta"] = chunk_resposta 
                    
                    doc = Document(page_content=chunk_busca, metadata=chunk_metadata)
                    all_final_chunks.append(doc)
                    tracker.add_doc()
            
            report = tracker.get_report()
            full_report[file_name] = report
            print(f"   -> Concluído. {report['documentos_criados_para_rag']} chunks semânticos criados.")

        except Exception as e:
            print(f"!!! ERRO [RAG Loader]: Falha ao processar {file_name}. Erro: {e}")
            full_report[file_name] = {"erro_leitura": str(e)}
            continue
    
    print(f"--- [RAG Loader] Processamento Concluído ---")
    print(f"Total de {len(all_final_chunks)} documentos (chunks agrupados) prontos para o vector store.")
    return all_final_chunks, full_report