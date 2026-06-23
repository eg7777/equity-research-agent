"""
LLM provider adapter. Exposes chat() and get_text() regardless of backend.
Switch providers via config.PROVIDER: "anthropic" | "openai" | "ollama"
"""

import asyncio
import config

PROVIDER = config.PROVIDER

if PROVIDER == "anthropic":
    import anthropic
    _client = anthropic.Anthropic()

    async def chat(*, model, system, messages, max_tokens=4096):
        cached_system = [{"type": "text", "text": system,
                          "cache_control": {"type": "ephemeral"}}]
        return await asyncio.to_thread(
            _client.messages.create,
            model=model, max_tokens=max_tokens,
            system=cached_system, messages=messages,
        )

    def get_text(resp):
        return "".join(b.text for b in resp.content if hasattr(b, "text"))

elif PROVIDER == "openai":
    import openai
    _client = openai.OpenAI()

    async def chat(*, model, system, messages, max_tokens=4096):
        msgs = [{"role": "system", "content": system}] + messages
        return await asyncio.to_thread(
            _client.chat.completions.create,
            model=model, max_tokens=max_tokens, messages=msgs,
        )

    def get_text(resp):
        return resp.choices[0].message.content or ""

elif PROVIDER == "ollama":
    import openai
    _client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    async def chat(*, model, system, messages, max_tokens=4096):
        msgs = [{"role": "system", "content": system}] + messages
        return await asyncio.to_thread(
            _client.chat.completions.create,
            model=model, max_tokens=max_tokens, messages=msgs,
        )

    def get_text(resp):
        return resp.choices[0].message.content or ""

else:
    raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}. Use anthropic, openai, or ollama.")
