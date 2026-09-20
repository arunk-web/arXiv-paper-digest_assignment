import os
from src.state import AgentState, Chunk
from src.vectorstore import VectorStore, get_embedder

_CHUNK_WORDS = 220
_OVERLAP_WORDS = 40


def _chunk_text(text: str, section: str):
    words = text.split()
    if not words:
        return []
    
    step = _CHUNK_WORDS - _OVERLAP_WORDS
    out = []
    for start in range(0, len(words), step):
        piece = words[start:start + _CHUNK_WORDS]
        if not piece:

            continue
        out.append(" ".join(piece))
        if start + _CHUNK_WORDS >= len(words):
            break
    return out



def run(state: AgentState, data_dir: str = "./data/vectorstores") -> AgentState:
    if state.halt:
        return state
    

    chunks = []

    idx = 0
    for section, text in state.parsed_sections.items():
        for piece in _chunk_text(text, section):
            chunks.append(Chunk(chunk_id=f"c{idx}", text=piece, section=section, page=-1))

            idx += 1

    if not chunks:
        state.errors.append("chunk_embed: no text available to chunk")

        return state

    state.chunks = chunks

    embedder = get_embedder()
    dim = embedder.get_sentence_embedding_dimension()

    vs = VectorStore(dim=dim)
    vs.add(

        texts=[c.text for c in chunks],
        metas=[{"chunk_id": c.chunk_id, "text": c.text, "section": c.section} for c in chunks],
    )

    path = os.path.join(data_dir, state.selected_paper.arxiv_id)
    vs.save(path)

    state.vectorstore_path = path

    
    state._vs = vs  # type: ignore[attr-defined]
    return state
