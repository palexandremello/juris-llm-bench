"""Cliente mínimo para a API REST do Ollama."""

from __future__ import annotations

import os

import httpx

DEFAULT_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 600.0, transport: httpx.BaseTransport | None = None):
        if not base_url.startswith("http"):
            base_url = f"http://{base_url}"
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def modelos(self) -> list[str]:
        try:
            r = self._http.get("/api/tags")
        except httpx.ConnectError:
            raise SystemExit(f"Ollama não responde em {self._http.base_url}. Suba com: docker compose up -d") from None
        r.raise_for_status()
        return [m["name"] for m in r.json()["models"]]

    def baixar(self, modelo: str) -> None:
        r = self._http.post("/api/pull", json={"model": modelo, "stream": False}, timeout=None)
        r.raise_for_status()

    def chat(self, modelo: str, prompt: str, schema: dict, opcoes: dict) -> dict:
        """Uma pergunta, resposta restrita ao JSON Schema (structured outputs do Ollama)."""
        payload = {
            "model": modelo,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema,
            "options": opcoes,
            "think": False,  # modelos com raciocínio (qwen3...) respondem direto
            "stream": False,
        }
        r = self._http.post("/api/chat", json=payload)
        r.raise_for_status()
        return r.json()
