import os
import re
import fitz  # PyMuPDF
import arxiv
from src.state import AgentState

_SECTION_HEADERS = [
    "abstract", "introduction", "related work", "background", "method",
    "methods", "methodology", "approach", "experiments", "experimental setup",
    "results", "evaluation", "discussion", "limitations", "conclusion",
    "conclusions", "references", "appendix",
]
_HEADER_RE = re.compile(
    r"^\s*(\d+\.?\s*)?(" + "|".join(_SECTION_HEADERS) + r")\s*$",
    re.IGNORECASE,
)


def _download(state: AgentState, data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    client = arxiv.Client()
    paper = next(client.results(arxiv.Search(id_list=[state.selected_paper.arxiv_id])))
    path = os.path.join(data_dir, f"{state.selected_paper.arxiv_id}.pdf")
    paper.download_pdf(dirpath=data_dir, filename=os.path.basename(path))
    return path


def run(state: AgentState, data_dir: str = "./data/pdfs") -> AgentState:
    if state.halt:
        return state

    try:
        state.pdf_path = _download(state, data_dir)
    except Exception as e:
        state.errors.append(f"pdf_download failed: {e}")
        state.halt = True
        state.halt_reason = "Could not download the PDF from arXiv."
        return state

    try:
        doc = fitz.open(state.pdf_path)
    except Exception as e:
        state.errors.append(f"pdf_open failed: {e}")
        state.halt = True
        state.halt_reason = "Downloaded PDF is unreadable/corrupted."
        return state

    # --- huge paper guard ---
    if doc.page_count > 80:
        state.parse_warning = (
            f"Paper has {doc.page_count} pages; only the first 40 and last 10 "
            f"pages are parsed to keep this pipeline responsive."
        )
        pages_to_read = list(range(0, min(40, doc.page_count))) + \
            list(range(max(40, doc.page_count - 10), doc.page_count))
    else:
        pages_to_read = list(range(doc.page_count))

    full_text_parts = []
    char_count = 0
    for i in pages_to_read:
        page = doc.load_page(i)
        text = page.get_text("text")
        full_text_parts.append(text)
        char_count += len(text.strip())
    doc.close()

    full_text = "\n".join(full_text_parts)
    state.full_text = full_text

    # --- scanned-PDF / broken-layout detection ---
    avg_chars_per_page = char_count / max(len(pages_to_read), 1)
    if avg_chars_per_page < 200:
        state.parse_ok = False
        state.parse_warning = (
            "Extracted text is very sparse (likely a scanned PDF or an unusual "
            "layout). Falling back to using only the arXiv abstract for the "
            "briefing; QA grounding quality will be reduced."
        )
        state.parsed_sections = {"abstract": state.selected_paper.abstract}
        return state

    state.parse_ok = True

    # --- naive section splitting by scanning lines against known headers ---
    sections = {}
    current = "preamble"
    sections[current] = []
    for line in full_text.split("\n"):
        if _HEADER_RE.match(line.strip()):
            current = line.strip().lower()
            sections.setdefault(current, [])
        else:
            sections[current].append(line)

    state.parsed_sections = {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}
    if "abstract" not in state.parsed_sections:
        state.parsed_sections["abstract"] = state.selected_paper.abstract

    return state
