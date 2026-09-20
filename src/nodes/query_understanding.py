import re
from src.state import AgentState

# matches "2401.12345", "2401.12345v2", or full arxiv.org URLs
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(v\d+)?)")


def run(state: AgentState) -> AgentState:
    q = state.raw_query.strip()
    match = _ARXIV_ID_RE.search(q)
    if match and ("arxiv" in q.lower() or len(q) < 25):
        # Looks like the user gave (mostly) just an ID or a URL, not a topic sentence.
        state.mode = "paper_id"
        state.parsed_arxiv_id = match.group(1)
    else:
        state.mode = "topic_search"
    return state
