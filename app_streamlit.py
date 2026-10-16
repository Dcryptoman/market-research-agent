"""
app_streamlit.py - the market agent as a web page, showing every step live.

Run:
    streamlit run app_streamlit.py

It uses exactly the same agent loop as the terminal version (agent.py) -
we just pass in a different `show` function that draws each step on the page
instead of printing it.

Reminder: Streamlit re-runs this whole file on every click. Anything that must
survive a re-run (the last run's steps and report) lives in st.session_state.
"""

import json
import os

import anthropic
import streamlit as st

from agent import run_agent
from tools import REPORTS_FOLDER

st.set_page_config(page_title="Market Research Agent", page_icon="📈")
st.title("📈 Market Research Agent")
st.caption("Give it a goal. It plans the steps, pulls live market data with its "
           "tools, and writes a report. Read-only - it cannot trade.")

EXAMPLES = [
    "Compare Bitcoin, gold and the S&P 500 over the last year",
    "Analyse Apple's trend and momentum over the last year",
    "Is EUR/USD or GBP/USD more volatile right now?",
    "How correlated are Bitcoin and Ethereum? Use 2 years of data",
    "Compare the DAX, FTSE and S&P 500 over 6 months",
]


def safe_markdown(text: str) -> str:
    """Streamlit treats $...$ as maths formulas - escape $ so prices show normally."""
    return text.replace("$", "\\$")


def render_step(kind: str, text: str):
    """Draw one agent step. `kind` matches the labels agent.run_agent emits."""
    if kind == "agent":
        st.markdown(safe_markdown(text))
    elif kind == "note":
        st.caption(f"💭 {text}")
    elif kind.startswith("tool"):
        name, _, args = text.partition("(")
        args = args[:-1]  # drop the closing ")"
        if len(args) > 300:  # e.g. save_report carries the whole report
            args = args[:300] + " ..."
        st.markdown(f"🔧 **{name}** &nbsp;·&nbsp; round {kind.split()[1]}")
        st.code(args, language="json")
    elif kind == "ok":
        with st.expander("✅ Tool result"):
            try:
                st.json(json.loads(text))
            except ValueError:
                st.text(text)
    elif kind == "error":
        st.error(f"Tool error (the agent sees this and can adjust): {text}")


def show_report(answer: str):
    st.subheader("Report")
    st.markdown(safe_markdown(answer))
    st.download_button("Download report (.md)", answer,
                       file_name="market_report.md", mime="text/markdown")


# ---------------------------------------------------------------------------
# Sidebar: example goals and past reports.
# ---------------------------------------------------------------------------
def use_example(example: str):
    st.session_state.goal = example  # runs before the text box is drawn


with st.sidebar:
    st.header("Try an example")
    for example in EXAMPLES:
        st.button(example, on_click=use_example, args=(example,), use_container_width=True)

    st.header("Past reports")
    reports = sorted(REPORTS_FOLDER.glob("*.md"), reverse=True) if REPORTS_FOLDER.exists() else []
    # One button per report (newest first). Buttons rather than a dropdown:
    # a dropdown resets itself whenever a new report joins the list.
    for report in reports[:10]:
        st.button(report.stem, key=f"report_{report.name}", use_container_width=True,
                  on_click=lambda name=report.name: st.session_state.update(picked=name))
    if not reports:
        st.caption("None yet - run the agent to create one.")

picked = st.session_state.get("picked")


# ---------------------------------------------------------------------------
# Main area: goal box + run button.
# ---------------------------------------------------------------------------
goal = st.text_area("Goal", key="goal", height=80,
                    placeholder="e.g. Compare Bitcoin and gold over the last year")
run = st.button("Run agent", type="primary", disabled=not goal.strip())

if run:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("No API key found. Put ANTHROPIC_API_KEY in the .env file (see README.md).")
        st.stop()

    st.session_state.events = []
    st.session_state.answer = None
    st.subheader("Steps")
    steps_box = st.container(border=True)

    def show(kind, text):
        """Called by the agent loop for every step - draw it right away."""
        st.session_state.events.append((kind, text))
        with steps_box:
            render_step(kind, text)

    answer = None
    with st.spinner("Agent working... (usually 30-90 seconds)"):
        try:
            answer = run_agent(goal, show=show)
        except anthropic.AuthenticationError:
            st.error("Your API key was rejected. Check ANTHROPIC_API_KEY in .env.")
        except anthropic.RateLimitError:
            st.error("Rate limited by the API - wait a moment and try again.")
        except anthropic.APIConnectionError:
            st.error("Couldn't reach the Anthropic API - check your internet connection.")
        except anthropic.APIStatusError as error:
            st.error(f"API error ({error.status_code}): {error.message}")

    st.session_state.answer = answer
    if answer:
        show_report(answer)

elif st.session_state.get("events"):
    # Any other click (e.g. download) re-runs the page: redraw the last run.
    st.subheader("Steps (last run)")
    with st.container(border=True):
        for kind, text in st.session_state.events:
            render_step(kind, text)
    if st.session_state.get("answer"):
        show_report(st.session_state.answer)

if picked and (REPORTS_FOLDER / picked).exists():
    st.divider()
    st.subheader(f"Saved report: {picked}")
    if st.button("Close saved report"):
        st.session_state.picked = None
        st.rerun()
    st.markdown(safe_markdown((REPORTS_FOLDER / picked).read_text(encoding="utf-8")))
