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
docker compose up -d                                                # sobe o Ollama em localhost:11434
uv sync
uv run juris-llm-bench exemplos/decisoes/config.yaml               # roda os modelos e mostra as notas
uv run juris-llm-bench exemplos/decisoes/config.yaml --so-avaliar  # só recalcula (ex.: depois de corrigir o gabarito)
uv run pytest -q                                                    # testes (Ollama simulado, sem Docker nem GPU)
```

Os modelos do `config.yaml` são baixados na primeira execução (~2-3 GB cada) e ficam guardados no volume
`ollama` do Docker. `docker compose down` para o Ollama sem apagar os modelos.

### GPU no Docker

O `docker-compose.yml` usa a GPU NVIDIA. Se o `docker compose up` der
`could not select device driver "nvidia"`, falta instalar o
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(uma vez só, Ubuntu):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

Para conferir: `docker compose exec ollama nvidia-smi`. Sem GPU NVIDIA, apague o bloco `deploy:` do
`docker-compose.yml`: funciona na CPU, só que bem mais devagar.

### Sem Docker

No Google Colab (GPU grátis), use [`notebooks/colab.ipynb`](notebooks/colab.ipynb). Em outra máquina, instale o
[Ollama](https://ollama.com/download) e rode `ollama serve`. Para usar um servidor remoto, defina
`OLLAMA_HOST=http://servidor:11434`.

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

O exemplo já vem com os resultados de uma execução numa RTX 5060 Ti, para
referência. No exemplo, o melhor foi o `qwen3:4b` (0,96 de acerto, ~1,8 s por documento).

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
