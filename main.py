"""
CLI entrypoint.

Usage:
    python main.py "recent work on KV-cache compression for LLMs"
    python main.py 2401.12345
    python main.py https://arxiv.org/abs/2401.12345
"""
import sys
import json

import os
from dotenv import load_dotenv

load_dotenv()

from src.graph import run_pipeline

from src.nodes import qa as qa_node


def print_briefing(briefing: dict):
    print("\n" + "=" * 70)
    print(f"TITLE:    {briefing['title']}")

    print(f"AUTHORS:  {', '.join(briefing['authors'])}")
    print(f"ARXIV ID: {briefing['arxiv_id']}   ({briefing['published']})")
    print(f"LINK:     {briefing['link']}")

    if briefing.get("parsing_note"):
        print(f"NOTE:     {briefing['parsing_note']}")
    print("-" * 70)
    print("WHY IT MATTERS:\n" + briefing["summary"])

    print("\nPROBLEM STATEMENT:\n" + briefing["problem_statement"])
    print("\nMETHOD:")


    for b in briefing["method"]:
        print(f"  - {b}")
    print("\nKEY RESULTS:")
    for b in briefing["key_results"]:

        print(f"  - {b}")
    print("\nLIMITATIONS:")
    for b in briefing["limitations"]:
        print(f"  - {b}")
    print("\nSUGGESTED FOLLOW-UP QUESTIONS:")

    for b in briefing["follow_up_questions"]:
        print(f"  - {b}")
    print("=" * 70 + "\n")


def save_briefing(briefing: dict, arxiv_id: str, data_dir: str):

    os.makedirs(os.path.join(data_dir, "briefings"), exist_ok=True)

    path = os.path.join(data_dir, "briefings", f"{arxiv_id}.json")
    with open(path, "w") as f:

        json.dump(briefing, f, indent=2)
    print(f"[saved] briefing -> {path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <topic | arxiv id | arxiv url>")
        sys.exit(1)


    query = " ".join(sys.argv[1:])

    data_dir = os.environ.get("DATA_DIR", "./data")

    state = run_pipeline(query)

    if state.halt:
        print(f"\n[STOPPED] {state.halt_reason}")

        if state.errors:
            print("Errors:", state.errors)

        sys.exit(1)

    if state.errors:
        print("[warnings]", state.errors)


    print_briefing(state.briefing)

    save_briefing(state.briefing, state.selected_paper.arxiv_id, data_dir)

    # --- interactive QA loop ---
    print("Ask questions about this paper (blank line to quit):")
    while True:

        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:

            break
        turn = qa_node.ask(state, q)
        print(f"A> {turn.answer}")
        
        if turn.sources:
            print(f"   (sources: {', '.join(turn.sources)})")


if __name__ == "__main__":
    main()
