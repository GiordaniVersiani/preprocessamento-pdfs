import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- FUNÇÃO 1: Carregar o Modelo de Embedding ---
def get_embedding_model():
    """
    Retorna o modelo de embeddings em português.
    """
    model_name = "neuralmind/bert-base-portuguese-cased" 
    print(f"🔤 Carregando modelo de embeddings: {model_name}")

    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print("  [RAG VectorStore] Modelo de embedding carregado (na CPU).")
    return embeddings

# --- FUNÇÕES 2 e 3: Criar ou Carregar o Índice ---
def _create_and_save_faiss_index(
    documents: List[Document], 
    embedding_model: HuggingFaceEmbeddings, 
    index_path: str
) -> FAISS:
    print(f"  [RAG VectorStore] Criando novo índice FAISS em: '{index_path}'")
    print("    -> Esta etapa pode demorar alguns minutos na primeira vez...")
    index = FAISS.from_documents(documents, embedding_model)
    index.save_local(index_path)
    print(f"    -> Índice FAISS criado e salvo com sucesso.")
    return index

def _load_faiss_index(
    embedding_model: HuggingFaceEmbeddings, 
    index_path: str
) -> FAISS:
    print(f"  [RAG VectorStore] Carregando índice FAISS existente de: '{index_path}'")
    index = FAISS.load_local(
        index_path, 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
    )
    print(f"    -> Índice FAISS carregado com sucesso.")
    return index

# --- FUNÇÃO 4: Ponto de Entrada ---
def get_vector_store(
    documents: List[Document], 
    embedding_model: HuggingFaceEmbeddings, 
    index_path: str = "faiss_index"
) -> FAISS: # <-- MUDANÇA: Retorna FAISS, não um retriever
    
    print(f"--- [RAG VectorStore] Iniciando Tarefa 2: Criação do Vector Store ---")
    
    if os.path.exists(index_path):
        vector_store = _load_faiss_index(embedding_model, index_path)
    else:
        vector_store = _create_and_save_faiss_index(documents, embedding_model, index_path)
        
    print(f"--- [RAG VectorStore] Tarefa 2 Concluída. Vector Store está pronto. ---")
    
    # --- MUDANÇA ---
    return vector_store
    # --- FIM DA MUDANÇA ---
