from __future__ import annotations
from bt.config import Settings
from bt.llm.base import LLMClient
from bt.llm.ollama_client import OllamaClient
from bt.llm.hf_client import HFEndpointClient
from bt.llm.hf_hub_client import HFHubClient

def build_llm_client(s: Settings) -> LLMClient:
    if s.provider == "ollama":
        return OllamaClient(s)
    if s.provider == "hf_hub":
        return HFHubClient(s)
    if s.provider == "hf_endpoint":
        if not s.hf_endpoint_url:
            raise ValueError("hf_endpoint_url must be set when provider='hf_endpoint'")
        return HFEndpointClient(s)
    raise ValueError(f"Unknown provider: {s.provider}")
