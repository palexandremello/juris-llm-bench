"""Benchmark: roda cada modelo em cada texto, compara com o gabarito e resume.

Tudo é definido num config.yaml (ver exemplos/decisoes/config.yaml):

    textos:   pasta com .txt (pode ser direto o dados/saida do juris-pdf)
    gabarito: CSV com uma coluna "id" (nome do .txt sem extensão) + uma coluna por campo
    modelos:  tags do Ollama
    prompt:   instrução enviada ao modelo; {texto} é substituído pelo documento
    campos:   nome -> tipo (texto, numero, data, lista, sim_nao ou lista de opções)

O tipo de cada campo define o schema JSON exigido do modelo E como ele é corrigido:

    texto      igual, ignorando maiúsculas/acentos/pontuação (ou ≥ 90% parecido)
    numero     diferença ≤ 0,01  ("R$ 1.234,56" vale)
    data       mesma data (aaaa-mm-dd ou dd/mm/aaaa)
    sim_nao    igual
    [opções]   mesma opção
    lista      F1 entre os itens (no CSV, itens separados por ";")
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import yaml

from .ollama import OllamaClient

OPCOES_PADRAO = {"temperature": 0, "seed": 42, "num_ctx": 16384}
CARACTERES_POR_TOKEN = 3.5  # aproximação para português


# --- config ------------------------------------------------------------------

def carregar_config(caminho: str | Path) -> dict:
    caminho = Path(caminho).resolve()
    cfg = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    pasta = caminho.parent
    cfg["textos"] = pasta / cfg["textos"]
    cfg["gabarito"] = pasta / cfg["gabarito"]
    cfg["resultados"] = pasta / cfg.get("resultados", "resultados")
    cfg["opcoes"] = {**OPCOES_PADRAO, **cfg.get("opcoes", {})}
    if "{texto}" not in cfg["prompt"]:
        raise ValueError("o prompt precisa conter {texto}")
    return cfg


def schema(campos: dict) -> dict:
    """JSON Schema a partir dos tipos simples. Todo campo aceita null (informação ausente no texto)."""
    tipos = {"texto": {"type": ["string", "null"]}, "numero": {"type": ["number", "null"]},
             "data": {"type": ["string", "null"]}, "sim_nao": {"type": ["boolean", "null"]},
             "lista": {"type": "array", "items": {"type": "string"}}}
    props = {}
    for nome, tipo in campos.items():
        props[nome] = {"enum": [*tipo, None]} if isinstance(tipo, list) else tipos[tipo]
    return {"type": "object", "properties": props, "required": list(campos)}


def ler_textos(pasta: Path) -> dict[str, str]:
    """{id: texto}. Aceita pasta plana ou a estrutura do juris-pdf (saida/<doc>/<doc>.txt)."""
    return {p.stem: p.read_text(encoding="utf-8", errors="replace") for p in sorted(pasta.rglob("*.txt"))}


def ler_gabarito(arquivo: Path, campos: dict) -> dict[str, dict]:
    if not arquivo.exists():
        return {}
    df = pd.read_csv(arquivo, dtype=str, keep_default_na=False)
    gab = {}
    for _, linha in df.iterrows():
        gab[linha["id"]] = {c: _do_csv(linha.get(c, ""), t) for c, t in campos.items()}
    return gab


def _do_csv(valor: str, tipo) -> object:
    valor = valor.strip()
    if tipo == "lista":
        return [v.strip() for v in valor.split(";") if v.strip()]
    if valor == "":
        return None
    if tipo == "sim_nao":
        return _normalizar(valor) in ("sim", "s", "true", "1")
    if tipo == "numero":
        return _numero(valor)
    return valor


def _para_csv(valor, tipo) -> str:
    if valor is None:
        return ""
    if tipo == "lista":
        return "; ".join(map(str, valor)) if isinstance(valor, list) else str(valor)
    if tipo == "sim_nao" and isinstance(valor, bool):
        return "sim" if valor else "não"
    return str(valor)


# --- rodar -------------------------------------------------------------------

def _arquivo_modelo(cfg: dict, modelo: str) -> Path:
    return cfg["resultados"] / (re.sub(r"[^\w.-]", "_", modelo) + ".jsonl")


def _respostas(arquivo: Path) -> dict[str, dict]:
    """Última resposta de cada documento (um run interrompido continua de onde parou)."""
    if not arquivo.exists():
        return {}
    linhas = [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: r for r in linhas}


def rodar(cfg: dict, cliente: OllamaClient | None = None) -> None:
    cliente = cliente or OllamaClient()
    textos = ler_textos(cfg["textos"])
    if not textos:
        raise FileNotFoundError(f"nenhum .txt em {cfg['textos']}")
    cfg["resultados"].mkdir(parents=True, exist_ok=True)
    fmt = schema(cfg["campos"])

    for id_, texto in textos.items():
        tokens = len(cfg["prompt"]) + len(texto)
        if tokens / CARACTERES_POR_TOKEN > cfg["opcoes"]["num_ctx"] * 0.9:
            print(f"⚠ {id_}: ~{tokens / CARACTERES_POR_TOKEN:.0f} tokens não cabem em num_ctx={cfg['opcoes']['num_ctx']}; "
                  "o Ollama vai cortar o texto. Aumente `opcoes: num_ctx` no config.")

    instalados = set(cliente.modelos())
    for modelo in cfg["modelos"]:
        arquivo = _arquivo_modelo(cfg, modelo)
        feitos = {i for i, r in _respostas(arquivo).items() if r.get("erro") is None}
        pendentes = [i for i in textos if i not in feitos]
        print(f"{modelo}: {len(pendentes)} documentos pendentes")
        if not pendentes:
            continue
        if modelo not in instalados and f"{modelo}:latest" not in instalados:
            print(f"  baixando {modelo}...")
            cliente.baixar(modelo)

        with arquivo.open("a", encoding="utf-8") as f:
            for n, id_ in enumerate(pendentes, 1):
                t = time.perf_counter()
                reg = {"id": id_}
                try:
                    r = cliente.chat(modelo, cfg["prompt"].replace("{texto}", textos[id_]), fmt, cfg["opcoes"])
                    bruto = r["message"]["content"]
                    try:
                        reg["resposta"] = json.loads(bruto)
                    except json.JSONDecodeError:
                        reg["resposta"], reg["bruto"] = None, bruto
                    reg["tokens_entrada"] = r.get("prompt_eval_count")
                    reg["erro"] = None
                except Exception as e:  # noqa: BLE001 — registra e segue para o próximo documento
                    reg["resposta"], reg["erro"] = None, str(e)
                reg["segundos"] = round(time.perf_counter() - t, 2)
                f.write(json.dumps(reg, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{n}/{len(pendentes)}] {id_}  {reg['segundos']}s" + (f"  ERRO: {reg['erro']}" if reg["erro"] else ""))


# --- corrigir ----------------------------------------------------------------

def _normalizar(s) -> str:
    s = unicodedata.normalize("NFKD", str(s).casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", "", s).split())


def _textos_iguais(a, b) -> bool:
    a, b = _normalizar(a), _normalizar(b)
    return a == b or SequenceMatcher(None, a, b).ratio() >= 0.9


def _numero(v) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).replace("R$", "").replace(" ", "")
    if "," in s:  # formato brasileiro: 1.234,56
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _data(v) -> date | None:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(str(v).strip()[:10], fmt).date()
        except ValueError:
            pass
    return None


def comparar(tipo, previsto, esperado) -> float:
    """Nota de 0 a 1 de um campo."""
    if tipo == "lista":
        p, e = list(previsto or []), list(esperado or [])
        if not p and not e:
            return 1.0
        acertos, restantes = 0, list(e)
        for item in p:
            par = next((x for x in restantes if _textos_iguais(item, x)), None)
            if par is not None:
                restantes.remove(par)
                acertos += 1
        return 2 * acertos / (len(p) + len(e))
    if previsto is None or esperado is None:
        return float(previsto is None and esperado is None)
    if tipo == "numero":
        p, e = _numero(previsto), _numero(esperado)
        return float(p is not None and e is not None and abs(p - e) <= 0.01)
    if tipo == "data":
        return float(_data(previsto) is not None and _data(previsto) == _data(esperado))
    if tipo == "sim_nao":
        return float(bool(previsto) == bool(esperado))
    if isinstance(tipo, list):
        return float(_normalizar(previsto) == _normalizar(esperado))
    return float(_textos_iguais(previsto, esperado))


def avaliar(cfg: dict) -> dict[str, pd.DataFrame]:
    """Compara as respostas com o gabarito. Grava CSVs em resultados/ e devolve os DataFrames:

    resumo       uma linha por modelo
    por_campo    acerto de cada modelo em cada campo
    erros        cada campo errado: documento, previsto × esperado
    """
    campos, gab = cfg["campos"], ler_gabarito(cfg["gabarito"], cfg["campos"])
    linhas, resumo = [], []
    for modelo in cfg["modelos"]:
        respostas = _respostas(_arquivo_modelo(cfg, modelo))
        if not respostas:
            continue
        # previsões no mesmo formato do gabarito: serve de rascunho para anotar documentos novos
        prev = [{"id": i, **{c: _para_csv((r["resposta"] or {}).get(c), t) for c, t in campos.items()}}
                for i, r in respostas.items()]
        pd.DataFrame(prev).to_csv(cfg["resultados"] / f"previsoes_{_arquivo_modelo(cfg, modelo).stem}.csv", index=False)

        corrigidos = [i for i in respostas if i in gab]
        for i in corrigidos:
            resp = respostas[i]["resposta"] or {}
            for c, t in campos.items():
                nota = comparar(t, resp.get(c), gab[i][c]) if respostas[i]["resposta"] is not None else 0.0
                linhas.append({"modelo": modelo, "id": i, "campo": c, "nota": nota,
                               "previsto": _para_csv(resp.get(c), t), "esperado": _para_csv(gab[i][c], t)})
        resumo.append({
            "modelo": modelo,
            "documentos": len(corrigidos),
            "json_valido": sum(respostas[i]["resposta"] is not None for i in respostas) / len(respostas),
            "segundos_por_doc": pd.Series([r["segundos"] for r in respostas.values()]).median(),
        })

    det = pd.DataFrame(linhas, columns=["modelo", "id", "campo", "nota", "previsto", "esperado"])
    resumo = pd.DataFrame(resumo)
    if resumo.empty:
        raise FileNotFoundError(f"nenhuma resposta em {cfg['resultados']}; rode primeiro")
    resumo.insert(1, "acerto", resumo.modelo.map(det.groupby("modelo")["nota"].mean()))
    resumo = resumo.sort_values("acerto", ascending=False).reset_index(drop=True)
    por_campo = det.pivot_table(index="modelo", columns="campo", values="nota", sort=False).reindex(resumo.modelo)
    erros = det[det.nota < 1].reset_index(drop=True)

    resumo.to_csv(cfg["resultados"] / "resumo.csv", index=False)
    por_campo.to_csv(cfg["resultados"] / "por_campo.csv")
    erros.to_csv(cfg["resultados"] / "erros.csv", index=False)
    return {"resumo": resumo, "por_campo": por_campo, "erros": erros}
