import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any

# --- Função 1: Carregar o Modelo ---
# (Esta lógica foi movida da Célula 6 para cá)

def get_semantic_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Carrega o modelo de embedding, tentando usar a GPU (cuda)
    se disponível.
    """
    print(f"   -> Carregando modelo de embedding: '{model_name}'...")
    print("      (Isso pode demorar um pouco na primeira vez)")
    
    # Tenta usar a GPU (cuda) se disponível, senão usa a CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"      (Usando dispositivo: {device})")
    
    model = SentenceTransformer(model_name, device=device)
    return model

# --- Função 2: O "Worker" de Deduplicação ---
# (Esta é a função que a Célula 6 irá chamar)

def deduplicate_semantically(
    blocks: List[Dict[str, Any]], 
    model: SentenceTransformer,
    global_seen_embeddings: List[torch.Tensor], 
    threshold: float = 0.85, 
    min_length: int = 50
) -> (List[Dict[str, Any]], List[torch.Tensor]):
    """
    Processa uma lista de blocos (de um arquivo) e remove duplicatas
    semânticas comparando com um cache global de embeddings.
    """
    
    clean_blocks = []
    new_embeddings_to_add = [] # Embeddings únicos deste arquivo

    # Divide os blocos em "curtos" (para manter) e "longos" (para analisar)
    blocks_to_keep = [b for b in blocks if len(b.get("texto_normalizado", "")) < min_length]
    blocks_to_process = [b for b in blocks if len(b.get("texto_normalizado", "")) >= min_length]
    
    clean_blocks.extend(blocks_to_keep) # Mantém todos os blocos curtos
    
    if not blocks_to_process:
        # Nenhum bloco longo para processar
        return clean_blocks, global_seen_embeddings

    # Gera embeddings para todos os blocos "longos" DE UMA VEZ
    texts_to_check = [b.get("texto_normalizado") for b in blocks_to_process]
    new_embeddings = model.encode(texts_to_check, convert_to_tensor=True, show_progress_bar=True)
    
    if not global_seen_embeddings:
        # Este é o primeiro arquivo, todos os blocos são únicos
        print("   -> (Primeiro arquivo, todos os blocos são únicos por padrão)")
        clean_blocks.extend(blocks_to_process)
        global_seen_embeddings.extend(new_embeddings)
    else:
        # Compara os N novos embeddings com os M embeddings da memória
        existing_embeddings_tensor = torch.stack(global_seen_embeddings)
        
        # Calcula a similaridade de cossenos
        similarities = util.cos_sim(new_embeddings, existing_embeddings_tensor)
        
        # Verifica, para cada novo bloco, se ele é duplicado
        for i in range(len(blocks_to_process)):
            # Verifica se QUALQUER similaridade para este bloco [i] é > threshold
            is_duplicate = torch.any(similarities[i] > threshold)
            
            if not is_duplicate:
                # Bloco é único
                clean_blocks.append(blocks_to_process[i])
                # Adiciona à lista temporária para não comparar com ele mesmo
                new_embeddings_to_add.append(new_embeddings[i])
            # else: Bloco é um duplicado semântico e é descartado
    
    # Adiciona os novos embeddings únicos ao cache global
    if new_embeddings_to_add:
        global_seen_embeddings.extend(new_embeddings_to_add)
        
    return clean_blocks, global_seen_embeddings