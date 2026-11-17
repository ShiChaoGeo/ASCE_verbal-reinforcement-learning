"""Schema-constrained calls to the LLM providers."""

from __future__ import annotations

import json

from vrl.config import key_for, require_key


DEFAULT_PROVIDER = "openai"

TEMPERATURE = 0.0


def _without_temperature(params: dict) -> dict:
    return {k: v for k, v in params.items() if k != "temperature"}


def _rejects_temperature(exc: Exception) -> bool:
    """True when the API refused the request because of ``temperature``."""
    msg = str(exc).lower()
    return "temperature" in msg and any(
        w in msg for w in ("unsupported", "not supported", "unexpected", "invalid")
    )


DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.4",
    "gemini": "gemini-2.5-pro",
    "deepseek": "deepseek-reasoner",
}


def _call_anthropic(system: str, user: str, schema: dict, model: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    with client.messages.stream(
        model=model,
        max_tokens=32000,
        temperature=TEMPERATURE,
        system=system,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": schema},
        },
        messages=[{"role": "user", "content": user}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        raise RuntimeError(f"Model refused: {response.stop_details}")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def _openai_refusal(response) -> str | None:
    """Return refusal text when the model declines a structured request."""
    for item in response.output or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", None) == "refusal":
                return getattr(part, "refusal", None) or "Model refused the request."
    return None


def _openai_params(system: str, user: str, schema: dict, model: str) -> dict:
    return {
        "model": model,
        "instructions": system,
        "input": user,
        "temperature": TEMPERATURE,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "vrl_output",
                "strict": True,
                "schema": schema,
            }
        },
    }


def _openai_finish(response) -> dict:
    refusal = _openai_refusal(response)
    if refusal:
        raise RuntimeError(f"Model refused: {refusal}")
    if not response.output_text:
        raise RuntimeError("OpenAI response contained no output_text.")
    return json.loads(response.output_text)


def _call_openai(system: str, user: str, schema: dict, model: str) -> dict:
    """OpenAI via the Responses API."""
    from openai import OpenAI

    client = OpenAI()
    params = _openai_params(system, user, schema, model)
    try:
        return _openai_finish(client.responses.create(**params))
    except Exception as exc:
        if not _rejects_temperature(exc):
            raise
        print(f"  ({params['model']} does not accept temperature; "
              "decoding is the provider default)")
        return _openai_finish(client.responses.create(**_without_temperature(params)))


def _call_gemini(system: str, user: str, schema: dict, model: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=TEMPERATURE,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return json.loads(response.text)


def _call_deepseek(system: str, user: str, schema: dict, model: str) -> dict:
    """DeepSeek has JSON mode but no strict schema enforcement."""
    from openai import OpenAI

    client = OpenAI(
        api_key=key_for("deepseek"),
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"{user}\n\nReturn JSON matching exactly this schema:\n"
                           f"{json.dumps(schema, indent=2)}",
            },
        ],
        response_format={"type": "json_object"},
        temperature=TEMPERATURE,
    )
    return json.loads(response.choices[0].message.content)


_PROVIDERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "gemini": _call_gemini,
    "deepseek": _call_deepseek,
}

PROVIDER_NAMES = tuple(_PROVIDERS)


def call_provider(
    provider: str,
    system: str,
    user: str,
    schema: dict,
    model: str | None = None,
) -> dict:
    """Send one schema-constrained request to any supported provider."""
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Choose from {list(_PROVIDERS)}.")
    require_key(provider)
    return _PROVIDERS[provider](system, user, schema, model or DEFAULT_MODELS[provider])
