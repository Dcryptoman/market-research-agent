"""
agent.py - an AI agent that takes a goal, plans steps, and uses tools.

Run:
    python agent.py "Compare Bitcoin and gold over the last year"
or just:
    python agent.py          (it will ask you for a goal)

HOW AN AGENT WORKS (the "agent loop"):
    1. We send Claude the goal + the list of tools it may use.
    2. Claude replies with either:
         - a request to use one or more tools  -> stop_reason == "tool_use"
         - a final answer                      -> stop_reason == "end_turn"
    3. If it asked for tools, WE run them (tools.py) and send the results back.
    4. Repeat from 2. Claude sees every earlier step, so it can react to
       results - retry a bad symbol, fetch more data, change its plan.

Claude decides WHAT to do next; our code decides WHETHER it's allowed and
actually does it. That split is what makes an agent safe to run.
"""

import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from tools import TOOL_DEFINITIONS, run_tool

load_dotenv(Path(__file__).parent / ".env")

MODEL = "claude-opus-5-5"
MAX_STEPS = 15   # safety cap: stop after this many round-trips, so a confused
                 # agent can't loop forever and run up your API bill

# Beta features used below:
#  - "thinking-display-updates": shows Claude's short progress notes between
#    tool calls ("Got BTC data, now fetching gold...") instead of hiding them.
#  - "server-side-fallback": if a request is ever declined by a safety check,
#    the API retries it on a suitable fallback model instead of just stopping.
BETAS = ["thinking-display-updates-2026-08-18", "server-side-fallback-2026-07-01"]

SYSTEM_PROMPT = """You are a market research agent. You turn a user's goal into a short plan, carry it out with your tools, and finish with a clear written summary.

How to work:
- Before your first tool call, state your plan as a short numbered list (3-6 steps).
- Use the tools for every number you report. Never invent or estimate prices or statistics.
- If a tool returns an error (e.g. a wrong symbol), fix the problem and retry, or explain why you can't.
- Fetch independent symbols in parallel (several tool calls in one turn) when you can.
- When the analysis is done, write the summary, save it with save_report, then reply with the summary.

The summary:
- Start with 2-3 sentences giving the main findings, then the key numbers (a small table is good), then what the indicators suggest about trend, momentum and risk.
- Name the data period and the date of the latest close.
- Describe what the data shows. Do not tell the reader to buy or sell anything. End with one line: "Not financial advice - generated automatically from Yahoo Finance data."
"""


def print_step(label: str, text: str):
    """Print one labelled, indented line (or block) of agent activity."""
    if label.startswith("tool") or label in ("ok", "error"):  # keep tool lines short
        text = text[:150] + ("..." if len(text) > 150 else "")
        label = f"  {label}" if label in ("ok", "error") else label
    lines = text.strip().splitlines() or [""]
    print(f"[{label}] {lines[0]}")
    for line in lines[1:]:
        print(" " * (len(label) + 3) + line)


def run_agent(goal: str, show=print_step) -> str:
    """
    Run the agent loop for one goal. Returns Claude's final answer.

    `show(label, text)` is called for every step as it happens, with label
    "agent" (plan/message), "note" (progress note), "tool N" (a tool call in
    round N), or "ok"/"error" (that tool's full result). The terminal prints
    them; the web page (app_streamlit.py) draws them instead.
    """
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": goal}]  # the conversation so far

    for step in range(1, MAX_STEPS + 1):
        # --- Ask Claude what to do next -------------------------------------
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
            thinking={"type": "adaptive", "display": "updates"},
            output_config={"effort": "medium"},
            betas=BETAS,
            fallbacks="default",
        )
        # Keep Claude's whole reply (text, notes AND tool requests) in the
        # history, unchanged - the API needs it to continue the conversation.
        messages.append({"role": "assistant", "content": response.content})

        # --- Show what Claude said / is thinking about ----------------------
        for block in response.content:
            if block.type == "thinking" and block.thinking:
                show("note", block.thinking)
            elif block.type == "text" and block.text.strip() and response.stop_reason == "tool_use":
                show("agent", block.text)

        # --- Finished? -----------------------------------------------------
        if response.stop_reason == "refusal":
            return "(Claude declined this request.)"
        if response.stop_reason != "tool_use":
            final = "\n".join(b.text for b in response.content if b.type == "text")
            if response.stop_reason == "max_tokens":
                final += "\n\n(Answer cut off - it hit the max_tokens limit.)"
            return final

        # --- Run every tool Claude asked for, then send back ALL results ----
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            show(f"tool {step}", f"{block.name}({json.dumps(block.input)})")
            result, is_error = run_tool(block.name, block.input)
            show("error" if is_error else "ok", result)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,   # links each result to its request
                "content": result,
                "is_error": is_error,
            })
        messages.append({"role": "user", "content": tool_results})

    return f"(Stopped after {MAX_STEPS} steps without finishing - try a simpler goal.)"


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No API key found. Copy .env.example to .env in the market_agent folder "
              "and put your key after ANTHROPIC_API_KEY= (see README.md).")
        return

    goal = " ".join(sys.argv[1:]).strip()
    if not goal:
        print("What should the agent do? Example:")
        print("  Compare Bitcoin, gold and the S&P 500 over the last year\n")
        goal = input("Goal: ").strip()
        if not goal:
            return

    print(f"\nGoal: {goal}\n" + "-" * 60)
    try:
        answer = run_agent(goal)
    except anthropic.AuthenticationError:
        print("\nYour API key was rejected. Check ANTHROPIC_API_KEY in .env.")
        return
    except anthropic.RateLimitError:
        print("\nRate limited by the API - wait a moment and try again.")
        return
    except anthropic.APIConnectionError:
        print("\nCouldn't reach the Anthropic API - check your internet connection.")
        return
    except anthropic.APIStatusError as error:
        print(f"\nAPI error ({error.status_code}): {error.message}")
        return

    print("-" * 60 + "\n" + answer)


if __name__ == "__main__":
    main()
