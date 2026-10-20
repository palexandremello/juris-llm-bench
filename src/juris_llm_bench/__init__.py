"""juris-llm-bench: compara LLMs locais (Ollama) na extração de dados de textos jurídicos."""

import argparse

from .bench import avaliar, carregar_config, rodar
from .ollama import OllamaClient

__all__ = ["avaliar", "carregar_config", "rodar", "OllamaClient"]


def main() -> None:
    ap = argparse.ArgumentParser(prog="juris-llm-bench")
    ap.add_argument("config", help="config.yaml do benchmark")
    ap.add_argument("--so-avaliar", action="store_true", help="não chama os modelos; só recalcula as notas (ex.: após editar o gabarito)")
    a = ap.parse_args()

    cfg = carregar_config(a.config)
    if not a.so_avaliar:
        rodar(cfg)
    r = avaliar(cfg)
    print("\n" + r["resumo"].to_string(index=False, float_format="%.2f"))
    print("\nAcerto por campo:\n" + r["por_campo"].to_string(float_format="%.2f"))
    print(f"\nDetalhes em {cfg['resultados']}/ (resumo.csv, por_campo.csv, erros.csv, previsoes_*.csv)")
