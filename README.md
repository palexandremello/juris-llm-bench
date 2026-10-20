# juris-llm-bench

Segundo projeto do **juris-pdf**: compara LLMs locais (via [Ollama](https://ollama.com)) na extração de dados
dos textos que o juris-pdf gera e confere as respostas com um gabarito.

## Fluxo

```
juris-pdf ──► dados/saida/*.txt ──► cada modelo lê cada texto ──► JSON com os campos ──► compara com gabarito.csv
                                    (prompt + campos do config.yaml)                       ──► resumo.csv, por_campo.csv, erros.csv
```

Um benchmark é uma pasta com três coisas (veja [`exemplos/decisoes`](exemplos/decisoes)):

| arquivo | o que é | quem edita |
|---|---|---|
| `config.yaml` | modelos, campos, prompt e onde estão os textos | quem monta o benchmark |
| `textos/` | um `.txt` por documento (ou aponte direto para `dados/saida` do juris-pdf) | — |
| `gabarito.csv` | resposta correta de cada documento, uma coluna por campo | jurista / anotador (Excel, Sheets) |

## Uso

```bash
ollama serve &
uv sync
uv run juris-llm-bench exemplos/decisoes/config.yaml               # roda os modelos e mostra as notas
uv run juris-llm-bench exemplos/decisoes/config.yaml --so-avaliar  # só recalcula (ex.: depois de corrigir o gabarito)
uv run pytest -q                                                    # testes (Ollama simulado, sem GPU)
```

No Google Colab (GPU grátis), use [`notebooks/colab.ipynb`](notebooks/colab.ipynb). O código também funciona
em qualquer máquina com GPU e Ollama. Para usar um servidor remoto, defina `OLLAMA_HOST=http://servidor:11434`.

Se o processo for interrompido, rode de novo: os documentos já respondidos são pulados.

## Campos e correção

Cada campo do `config.yaml` tem um tipo. O tipo define o formato que o modelo é obrigado a responder e também
como a resposta é corrigida:

| tipo | correto quando | no gabarito.csv |
|---|---|---|
| `texto` | igual, ignorando maiúsculas, acentos e pontuação (ou ≥ 90% parecido) | `TJSP` |
| `numero` | diferença ≤ 0,01 | `15000.00` |
| `data` | mesma data | `2025-03-14` ou `14/03/2025` |
| `sim_nao` | igual | `sim` / `não` |
| `[a, b, c]` | mesma opção (classificação, enriquecimento) | `consumidor` |
| `lista` | nota parcial: F1 entre os itens | `Fulano; Beltrano` |

Célula vazia no gabarito significa "não consta no documento" (o modelo deve responder `null`). Documentos sem
linha no gabarito são processados, mas não entram na nota.

## Resultados (`resultados/`)

- `resumo.csv`: por modelo, o `acerto` (média das notas dos campos), `% json_valido` e `segundos_por_doc`.
- `por_campo.csv`: acerto de cada modelo em cada campo.
- `erros.csv`: cada campo errado, com previsto × esperado.
- `previsoes_<modelo>.csv`: o que o modelo respondeu, no formato do gabarito.
- `<modelo>.jsonl`: respostas brutas.

## Com os seus documentos

1. Copie `exemplos/decisoes` e aponte `textos:` para a saída do juris-pdf.
2. Ajuste `campos` e `prompt`.
3. Rode sem gabarito.
4. Abra um `previsoes_<modelo>.csv`, corrija à mão e salve como `gabarito.csv`. Dá menos trabalho do que digitar
   do zero, mas revise de verdade: o gabarito herda os erros do modelo que você usou.
5. Rode com `--so-avaliar`.

## Cuidados

- **Tamanho do documento.** Se o texto passar de `num_ctx` (padrão 16384 tokens), o Ollama corta o texto sem
  avisar. O programa avisa antes de rodar. Processos inteiros podem precisar de 32768.
- **Texto bruto do Stirling.** O texto tem linhas quebradas e o rodapé do PJe repetido em toda página. O prompt
  do exemplo pede ao modelo que ignore isso. Não limpamos o texto porque às vezes o rodapé é o único lugar com o
  nome do magistrado e a data.
- Os documentos do exemplo são **fictícios**.
