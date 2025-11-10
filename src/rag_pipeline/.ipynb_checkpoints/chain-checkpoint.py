# ============================================================
# chain.py — Versão estável (fase “Lúcio e Guisso”)
# ============================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List


# ============================================================
# PROMPT PRINCIPAL (Gemma / LLaMA compatível)
# ============================================================
RAG_PROMPT_TEMPLATE = """<bos><start_of_turn>user
Você é um assistente especializado nos documentos oficiais do curso de Ciência da Computação do IFNMG.

Use APENAS as informações do contexto abaixo para responder à pergunta.
Se a informação estiver **implícita**, faça uma dedução lógica e avise o usuário.
Se a informação **não estiver presente**, diga claramente: "A informação não foi encontrada nos documentos."

Regras para a resposta:
- Responda em português formal e direto.
- Evite repetir o enunciado da pergunta.
- Sempre que possível, mencione o documento (ex: "Fonte: PPCBCC2019.jsonl") e o trecho relevante.
- Nunca invente nomes, valores ou prazos que não existam no contexto.
- Quando houver números, interprete se representam tempo (anos, semestres) ou carga horária (horas).
- Se o contexto mencionar "prazo", prefira unidades de tempo (anos ou semestres), nunca horas.
- Se houver menção a porcentagem ou nota mínima de aprovação, inclua o valor.

CONTEXTO:
{context}

PERGUNTA:
{question}
<end_of_turn>
<start_of_turn>model
"""
MAX_CONTEXT_CHARS = 7000  # evita estouro de contexto
# ============================================================
# Função para formatar documentos para o prompt
# ============================================================
def _format_docs(docs: List[Document]) -> str:
    """
    Formata os documentos para o prompt, preservando o texto original.
    Mantém apenas o campo 'texto_bruto_resposta' para melhor escrita final.
    """
    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get('source_file', 'N/A')
        pagina = doc.metadata.get('pagina', 'N/A')
        trecho = doc.metadata.get('texto_bruto_resposta', '').strip()

        formatted_docs.append(
            f"--- Trecho do Documento: {source} (Página: {pagina}) ---\n{trecho}"
        )

    return "\n\n".join(formatted_docs)


# ============================================================
# Função principal — criação da cadeia RAG
# ============================================================
def create_rag_chain(retriever, llm):
    """
    Cria a cadeia completa do RAG (Retrieval-Augmented Generation)
    usada no experimento 'Lúcio e Guisso'.
    """
    print("--- [RAG Chain] Iniciando Tarefa 4: Montando o RAG Chain ---")

    # 1️⃣ Cria o prompt template
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # 2️⃣ Pipeline paralelo — busca + formatação
    setup_and_retrieval = RunnableParallel(
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough()
        }
    )

    # 3️⃣ Encadeia busca + LLM + parser final
    rag_chain = setup_and_retrieval | prompt | llm | StrOutputParser()

    print("--- [RAG Chain] Tarefa 4 Concluída. O RAG Chain está pronto. ---")
    return rag_chain
