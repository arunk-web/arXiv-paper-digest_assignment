import json
from src.state import AgentState
from src.llm import call_llm_json

_SYSTEM = """You are a precise research-paper analyst. You only state things that
are supported by the provided paper text. If something is not discussed in the
text, write "Not stated in the paper" for that field instead of guessing.
Respond ONLY with a single JSON object, no prose, no markdown fences."""

_SCHEMA_HINT = """
Return JSON with exactly these keys:
{
  "summary": "<1 paragraph, plain English, 'why this paper matters'>",
  "problem_statement": "<1-3 sentences>",
  "method": ["<bullet 1>", "<bullet 2>", "..."],
  "key_results": ["<bullet 1>", "<bullet 2>", "..."],
  "limitations": ["<bullet 1>", "<bullet 2>", "..."],
  "follow_up_questions": ["<question 1>", "<question 2>", "<question 3>"]
}
"""


def _context_for_summary(state: AgentState, max_chars: int = 9000) -> str:
    # Prioritize abstract + intro + conclusion/discussion + limitations if present
    priority_keys = [
        "abstract", "1. introduction", "introduction", "conclusion",
        "conclusions", "discussion", "limitations",
    ]
    parts = []
    used = set()
    for key in priority_keys:
        for section_name, text in state.parsed_sections.items():
            if key in section_name and section_name not in used:
                parts.append(f"### {section_name}\n{text}")
                used.add(section_name)

    # fill remaining budget with whatever sections are left, in order
    for section_name, text in state.parsed_sections.items():
        if section_name not in used:
            parts.append(f"### {section_name}\n{text}")
            used.add(section_name)

    context = "\n\n".join(parts)
    return context[:max_chars]


def run(state: AgentState) -> AgentState:
    if state.halt:
        return state

    context = _context_for_summary(state)
    p = state.selected_paper

    prompt = (
        f"Paper title: {p.title}\n"
        f"Authors: {', '.join(p.authors)}\n"
        f"arXiv ID: {p.arxiv_id}\n\n"
        f"Paper text (may be partial):\n{context}\n\n"
        f"{_SCHEMA_HINT}"
    )

    try:
        parsed = call_llm_json(_SYSTEM, prompt)
    except Exception as e:
        state.errors.append(f"summarize failed: {e}")
        parsed = {
            "summary": "Summary generation failed; see errors in state.",
            "problem_statement": "Not available",
            "method": [], "key_results": [], "limitations": [],
            "follow_up_questions": [],
        }

    state.briefing = {
        "title": p.title,
        "authors": p.authors,
        "arxiv_id": p.arxiv_id,
        "published": p.published,
        "link": f"https://arxiv.org/abs/{p.arxiv_id}",
        **parsed,
    }
    if state.parse_warning:
        state.briefing["parsing_note"] = state.parse_warning

    return state
