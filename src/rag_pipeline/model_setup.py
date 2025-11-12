import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from typing import Optional
import traceback 

def get_llm(
    # --- MODELO EDITADO ---
    model_id: str = "google/gemma-2b-it", 
    # --- FIM DA EDIÇÃO ---
    max_new_tokens: int = 256, 
    temperature: float = 0.7
) -> Optional[HuggingFacePipeline]:
    
    print(f"--- [RAG ModelSetup] Iniciando Tarefa 3: Carregando LLM: '{model_id}' ---")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("  [RAG ModelSetup] Tokenizador carregado.")
    except Exception as e:
        print(f"!!! ERRO [RAG ModelSetup]: Falha ao carregar o tokenizador.")
        traceback.print_exc() 
        return None

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        print("  [RAG ModelSetup] Configurando quantização de 8-bit COM offload para CPU...")

        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "auto",
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs
        )
        print("  [RAG ModelSetup] Modelo LLM carregado (em 8-bit, com offload GPU/CPU).")
    except Exception as e:
        print(f"!!! ERRO [RAG ModelSetup]: Falha ao carregar o modelo.")
        traceback.print_exc() 
        return None

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1
    )
    
    llm_pipeline = HuggingFacePipeline(pipeline=pipe)
    
    print(f"--- [RAG ModelSetup] Tarefa 3 Concluída. LLM está pronto. ---")
    return llm_pipeline