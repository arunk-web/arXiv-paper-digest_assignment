"""
Shared, persistent state object that flows through every node of the graph.
Kept as a plain dataclass (not framework-specific) so the graph logic is easy
to read and port to LangGraph / any other orchestrator later.
"""
from dataclasses import dataclass, field

from typing import Optional, List, Dict, Any


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str

    authors: List[str]
    abstract: str

    published: str
    pdf_url: str
    categories: List[str]


@dataclass
class Chunk:
    chunk_id: str
    text: str

    section: str
    page: int


@dataclass
class QATurn:

    question: str
    answer: str

    grounded: bool
    sources: List[str] = field(default_factory=list)


@dataclass
class AgentState:
    
    raw_query: str = ""

    
    mode: str = ""                     
    parsed_arxiv_id: Optional[str] = None

    
    candidates: List[PaperMeta] = field(default_factory=list)

    
    selected_paper: Optional[PaperMeta] = None

  
    pdf_path: Optional[str] = None
    parsed_sections: Dict[str, str] = field(default_factory=dict)  # {section_name: text}
    full_text: str = ""
    parse_ok: bool = False
    parse_warning: Optional[str] = None

   
    chunks: List[Chunk] = field(default_factory=list)
    vectorstore_path: Optional[str] = None

    
    briefing: Optional[Dict[str, Any]] = None

    
    qa_history: List[QATurn] = field(default_factory=list)

    
    errors: List[str] = field(default_factory=list)
    halt: bool = False
    halt_reason: Optional[str] = None
