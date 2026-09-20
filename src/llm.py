"""
Thin provider-agnostic LLM wrapper. Supports Groq (default) and Gemini,
both usable on free tiers. Switch with LLM_PROVIDER env var.
"""
import os
import json


def _groq_call(system: str, prompt: str, json_mode: bool = False) -> str:
    from groq import Groq
    
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2000,
        **kwargs,
    )
    return resp.choices[0].message.content


def _gemini_call(system: str, prompt: str, json_mode: bool = False) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
        system_instruction=system,
    )
    cfg = {"temperature": 0.2}
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    resp = model.generate_content(prompt, generation_config=cfg)
    return resp.text


def call_llm(system: str, prompt: str, json_mode: bool = False) -> str:
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if provider == "groq":

        return _groq_call(system, prompt, json_mode)
    elif provider == "gemini":

        return _gemini_call(system, prompt, json_mode)
    else:

        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def call_llm_json(system: str, prompt: str) -> dict:
    """Call the LLM and robustly parse a JSON object out of the response."""
    raw = call_llm(system, prompt, json_mode=True)
    raw = raw.strip()

    # strip markdown code fences if the model added them anyway
    if raw.startswith("```"):

        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:

        # last-resort: find the first {...} block
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
        raise
