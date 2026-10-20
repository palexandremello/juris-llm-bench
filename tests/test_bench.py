"""Roda o exemplo inteiro contra um Ollama simulado (sem GPU, sem rede).

"perfeito" devolve o gabarito; "fraco" erra o resultado e não devolve JSON no 1º documento.
"""

import json
import shutil
from pathlib import Path

import httpx

from juris_llm_bench import OllamaClient, avaliar, carregar_config, rodar
from juris_llm_bench.bench import comparar, ler_gabarito

EXEMPLO = Path(__file__).resolve().parents[1] / "exemplos" / "decisoes"


def ollama_falso(gabarito: dict, textos: dict):
    def responder(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "perfeito:latest"}]})
        if req.url.path == "/api/pull":
            return httpx.Response(200, json={"status": "success"})
        corpo = json.loads(req.content)
        id_ = next(i for i, t in textos.items() if t in corpo["messages"][0]["content"])
        resposta = dict(gabarito[id_])
        if corpo["model"] == "fraco":
            if id_ == "dec-001":
                return httpx.Response(200, json={"message": {"content": "não sei"}})
            resposta["resultado"] = "procedente" if resposta["resultado"] != "procedente" else "improcedente"
        return httpx.Response(200, json={"message": {"content": json.dumps(resposta, ensure_ascii=False)}})
    return httpx.MockTransport(responder)


def test_exemplo_de_ponta_a_ponta(tmp_path):
    pasta = tmp_path / "decisoes"
    shutil.copytree(EXEMPLO, pasta, ignore=shutil.ignore_patterns("resultados"))
    cfg = carregar_config(pasta / "config.yaml")
    cfg["modelos"] = ["perfeito", "fraco"]
    gab = ler_gabarito(cfg["gabarito"], cfg["campos"])
    textos = {p.stem: p.read_text() for p in (pasta / "textos").glob("*.txt")}

    rodar(cfg, OllamaClient(transport=ollama_falso(gab, textos)))
    r = avaliar(cfg)

    resumo = r["resumo"].set_index("modelo")
    assert resumo.loc["perfeito", "acerto"] == 1.0
    assert resumo.loc["fraco", "acerto"] < 1.0
    assert resumo.loc["fraco", "json_valido"] == 0.9
    assert r["por_campo"].loc["fraco", "resultado"] == 0.0
    assert set(r["erros"].modelo) == {"fraco"}
    for f in ("resumo.csv", "por_campo.csv", "erros.csv", "previsoes_perfeito.csv"):
        assert (cfg["resultados"] / f).exists()

    # as previsões do modelo perfeito, no formato do gabarito, reproduzem o gabarito
    assert ler_gabarito(cfg["resultados"] / "previsoes_perfeito.csv", cfg["campos"]) == gab

    # rodar de novo não chama o modelo para documentos já respondidos
    linhas = (cfg["resultados"] / "perfeito.jsonl").read_text().count("\n")
    rodar(cfg, OllamaClient(transport=ollama_falso(gab, textos)))
    assert (cfg["resultados"] / "perfeito.jsonl").read_text().count("\n") == linhas


def test_correcao_por_tipo():
    assert comparar("texto", "1023456-78.2024.8.26.0100", "10234567820248260100") == 1
    assert comparar("texto", "TJ-SP", "TJSP") == 1
    assert comparar("texto", "Des. Helena Castro Vianna", "Helena Castro Vianna") == 1
    assert comparar("texto", "Fulano", "Beltrano") == 0
    assert comparar("numero", "R$ 72.340,18", 72340.18) == 1
    assert comparar("numero", None, None) == 1
    assert comparar("data", "14/03/2025", "2025-03-14") == 1
    assert comparar("sim_nao", False, False) == 1
    assert comparar(["sentenca", "acordao"], "acordao", "sentenca") == 0
    assert comparar("lista", ["BANCO HORIZONTE SA", "Fulano"], ["Banco Horizonte S.A."]) == 2 / 3
    assert comparar("lista", [], []) == 1
