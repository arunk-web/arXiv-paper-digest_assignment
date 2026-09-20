from src.state import AgentState, QATurn
from src.llm import call_llm

_SYSTEM = """You are answering questions about ONE specific research paper using
ONLY the retrieved excerpts given to you. Rules:
- If the excerpts do not contain the answer, reply exactly:
  "I couldn't find that in the paper." and nothing else.
- Never use outside/general knowledge to fill gaps.
- Keep answers concise (2-6 sentences) and cite which section each fact came from
  when possible, e.g. "(from: results)".
"""



def ask(state: AgentState, question: str, k: int = 5) -> QATurn:
    vs = getattr(state, "_vs", None)
    if vs is None:
        from src.vectorstore import VectorStore
        vs = VectorStore.load(state.vectorstore_path)


    hits = vs.search(question, k=k)
    if not hits:
        turn = QATurn(question=question, answer="I couldn't find that in the paper.",
                       grounded=False, sources=[])
        state.qa_history.append(turn)
        return turn

    context = "\n\n".join(f"[{m['section']}] {m['text']}" for m, _ in hits)
    prompt = f"Retrieved excerpts:\n{context}\n\nQuestion: {question}\nAnswer:"

    try:
        answer = call_llm(_SYSTEM, prompt).strip()
    except Exception as e:
        answer = f"(LLM error: {e})"


    grounded = "couldn't find that in the paper" not in answer.lower()
    sources = sorted({m["section"] for m, _ in hits})

    turn = QATurn(question=question, answer=answer, grounded=grounded, sources=sources)
    state.qa_history.append(turn)
    return turn
