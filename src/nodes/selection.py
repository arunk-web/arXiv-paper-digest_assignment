from src.state import AgentState
from src.vectorstore import get_embedder
import numpy as np


def run(state: AgentState) -> AgentState:
    if state.halt:
        return state

    if state.mode == "paper_id" or len(state.candidates) == 1:
        state.selected_paper = state.candidates[0]
        return state


    embedder = get_embedder()
    query_vec = embedder.encode([state.raw_query], convert_to_numpy=True)[0]

    abstracts = [c.abstract for c in state.candidates]
    abs_vecs = embedder.encode(abstracts, convert_to_numpy=True)


    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    scored = sorted(
        zip(state.candidates, abs_vecs),
        key=lambda pair: cos(query_vec, pair[1]),

        reverse=True,
    )
    state.candidates = [c for c, _ in scored]  
    state.selected_paper = state.candidates[0] 
    return state
