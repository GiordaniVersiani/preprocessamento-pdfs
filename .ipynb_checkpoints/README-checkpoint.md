# Projeto Modelo – Pré-processamento de PDFs

Este repositório serve como **exemplo de estrutura** para os trabalhos em equipes.

Projeto Integrador 2025 do Curso de Ciência da Computação do IFNMG Campus Montes Claros

## Estrutura

- `src/`: funções Python reutilizáveis
- `notebooks/`: exemplos em Jupyter Notebook
- `data/input/`: documentos PDF de entrada
- `data/output/`: resultados gerados em JSON

## Como rodar
1. Abra o anaconda prompt.
2. Navegue ate a pasta onde esta o projeto.
3. Rode este comando no terminal:
   ```bash
   conda env create -f environment.yml
   ```
4. Ative o ambiente com o comando:
   ```bash
   conda activate ia_projeto
   ```
5. Abra o jupyter lab com o comando no terminal:
    ```bash
   jupyter lab
   ```

## Realizar Testes
1. Para extrair pdfs, Execute o Notebook **"extracao_pdf_EQP4"**.
2. Para rodar o modelo rag, Execute o Notebook **"experimentoEquipe4"**.
