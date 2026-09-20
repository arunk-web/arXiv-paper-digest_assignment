# Autonomous arXiv Paper Digest & QA Agent

Ek agent jo topic ya arXiv paper ID leta hai, paper ko fetch aur parse karta hai,
ek structured summary (briefing) banata hai, aur us paper ke baare me follow-up
questions ka answer deta hai (RAG based).

## Architecture (State Graph)

Pipeline 7 steps me chalta hai, har step state ko update karke agle ko deta hai:

1. **Query Understanding** — topic hai ya arXiv ID, ye pehchanta hai
2. **arXiv Retrieval** — arXiv API se matching papers dhoondta hai
3. **Selection** — agar multiple papers mile to embedding similarity se best pick karta hai
4. **Fetch & Parse** — PDF download karke PyMuPDF se text nikalta hai, sections me todta hai
5. **Chunk & Embed** — text ko chunks me todke sentence-transformers se embed karta hai, FAISS me store karta hai
6. **Summarize** — LLM (Groq) se structured JSON briefing banwata hai (summary, method, results, limitations)
7. **QA Loop** — user ke question ko embed karke relevant chunks retrieve karta hai, sirf unhi chunks ke basis pe grounded answer deta hai

Sab state `AgentState` object me store hota hai jo har node ko pass hota hai.

## Failure Handling

- Zero arXiv results milne pe pipeline ruk jata hai, clear message deta hai
- PDF corrupt/scanned ho to abstract-only mode me fallback karta hai
- Bohot bade papers (80+ pages) me sirf shuru aur end ke pages parse hote hain
- QA me agar answer paper me nahi milta to model "I couldn't find that in the paper" bolta hai, guess nahi karta

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env
```

`.env` me free Groq API key daalo (https://console.groq.com/keys se milti hai).

## Run

```bash
python main.py "KV-cache compression for LLMs"
```

## Example Output

Paper "EVICPRESS" pe test kiya gaya:

- Briefing me title, authors, summary, method, key results, limitations sab sahi aaye
- QA test 1: "What compression method does EVICPRESS use?" → Answer sahi mila (sources ke saath)
- QA test 2: "Does the paper report results on multi-GPU inference?" → Model ne bola "I couldn't find that in the paper" — matlab hallucinate nahi kiya

## Design Decisions

- LangGraph jaisi framework ki jagah simple Python state machine use kiya — kam dependencies, samajhna easy
- Embeddings local (sentence-transformers) rakhe taaki koi paid API na lage
- QA answers sirf retrieved chunks pe based hote hain, model ko explicitly bola gaya hai ki na mile to "don't know" bole

## Limitations / Future Work

- Section detection regex-based hai, har paper ke headers standard nahi hote to thoda miss ho sakta hai
- Scanned PDFs ke liye OCR nahi hai
- Time milta to: answer-verification step add karta, tables/figures better parse karta