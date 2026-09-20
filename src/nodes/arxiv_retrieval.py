import arxiv
from src.state import AgentState, PaperMeta


def run(state: AgentState, max_results: int = 5) -> AgentState:
    client = arxiv.Client()

    if state.mode == "paper_id":
        search = arxiv.Search(id_list=[state.parsed_arxiv_id])
    else:
        search = arxiv.Search(
            query=state.raw_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

    try:
        results = list(client.results(search))
    except Exception as e:
        state.errors.append(f"arxiv_retrieval failed: {e}")
        results = []

    candidates = []
    for r in results:
        candidates.append(PaperMeta(
            arxiv_id=r.get_short_id(),
            title=r.title.strip(),
            authors=[a.name for a in r.authors],
            abstract=r.summary.strip().replace("\n", " "),
            published=str(r.published.date()),
            pdf_url=r.pdf_url,
            categories=r.categories,
        ))
    state.candidates = candidates

    # --- failure case: zero results for a vague/narrow topic ---
    if not candidates:
        state.halt = True
        state.halt_reason = (
            "arXiv returned zero candidates. Try a broader or differently-worded "
            "topic, or provide a direct arXiv ID/URL."
        )
    return state
