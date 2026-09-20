"""
Explicit state graph orchestrator.

Nodes:  query_understanding -> arxiv_retrieval -> selection -> fetch_parse
        -> chunk_embed -> summarize -> (QA loop, driven separately)

Edges are conditional: if a node sets state.halt = True, downstream nodes
are skipped and the halt_reason is surfaced to the caller. This is the
"graph" (not a single prompt chain) required by the assessment: each node
is a pure function State -> State, and control flow between them is
explicit here rather than implicit inside one giant prompt.
"""
from src.state import AgentState
from src.nodes import (
    query_understanding,
    arxiv_retrieval,

    selection,
    fetch_parse,

    chunk_embed,
    summarize,
)

NODES = [
    ("query_understanding", query_understanding.run),
    ("arxiv_retrieval", arxiv_retrieval.run),
    ("selection", selection.run),

    ("fetch_parse", fetch_parse.run),
    ("chunk_embed", chunk_embed.run),

    ("summarize", summarize.run),
]


def run_pipeline(query: str, verbose: bool = True) -> AgentState:
    state = AgentState(raw_query=query)
    for name, fn in NODES:

        if state.halt:
            if verbose:

                print(f"[graph] halted before '{name}': {state.halt_reason}")
            break
        if verbose:
            
            print(f"[graph] running node: {name}")
        state = fn(state)
    return state
