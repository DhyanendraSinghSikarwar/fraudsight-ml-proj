"""Analyst console. Streamlit, deliberately.

The console exists to make one claim demonstrable: a brief is readable and
actionable in about fifteen seconds. So the layout puts the narrative and the
three highest-weight signals above the fold, and the full evidence bundle behind
an expander. A console that renders the raw JSON would prove the opposite.

Three things here are not decoration:

* **The decision buttons are the feedback loop.** Every click POSTs to
  `/decision` and lands in the retraining data and the eval set. This is the C7
  capability and it is what makes the system a loop rather than a pipeline.
* **Disagreement is one click, not a form.** If capturing "the agent was wrong"
  is more effort than accepting, the data will say the agent is always right.
* **The trace is one click away.** An analyst who cannot see why will stop
  trusting the brief, and a system analysts do not trust does not reduce review
  volume no matter what the eval report says.

Run with `streamlit run src/fraudsight/console/app.py`, or `make console`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import streamlit as st

from fraudsight.config import settings
from fraudsight.data.cases import load_eval_cases
from fraudsight.triage.pipeline import TriagePipeline

st.set_page_config(page_title="fraudsight — analyst console", layout="wide")

DIRECTION_ICON = {"risk_increasing": "▲", "risk_decreasing": "▼"}
RECOMMENDATION_COLOUR = {
    "block": "#b3261e",
    "approve": "#146c2e",
    "request_step_up_auth": "#8a5a00",
    "escalate_to_analyst": "#1f4e8c",
}


@st.cache_resource
def _pipeline() -> TriagePipeline:
    return TriagePipeline.build()


@st.cache_data
def _cases() -> list[dict[str, Any]]:
    return [c.as_dict() for c in load_eval_cases()]


def _record_decision(transaction_id: str, decision: str, agreed: bool, notes: str = "") -> None:
    path = settings.artifact_dir / "analyst_decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "transaction_id": transaction_id,
                    "analyst_id": st.session_state.get("analyst_id", "console"),
                    "decision": decision,
                    "agreed_with_agent": agreed,
                    "notes": notes,
                }
            )
            + "\n"
        )


def main() -> None:
    st.title("fraudsight — analyst console")

    cases = _cases()
    with st.sidebar:
        st.header("Queue")
        st.caption(f"{len(cases)} cases in the triage band")
        strata = sorted({c["stratum"] for c in cases})
        chosen_stratum = st.selectbox("Stratum", ["all", *strata])
        pool = [c for c in cases if chosen_stratum in ("all", c["stratum"])]
        labels = [f"{c['case_id']}  ${c['amount']:,.0f}  score {c['model_score']:.3f}" for c in pool]
        idx = st.selectbox("Case", range(len(pool)), format_func=lambda i: labels[i])
        st.divider()
        st.text_input("Analyst id", key="analyst_id", value="console")
        show_truth = st.checkbox(
            "Reveal confirmed outcome",
            value=False,
            help="Off by default. Seeing the answer before deciding is how a review "
            "process stops measuring anything.",
        )

    case = pool[idx]
    pipe = _pipeline()

    with st.spinner("Running triage…"):
        decision = asyncio.run(pipe.process(case["transaction"]))

    state = decision.state
    brief = state.brief if state else None

    top = st.columns([2, 1, 1, 1])
    top[0].metric("Transaction", case["transaction_id"])
    top[1].metric("Amount", f"${case['amount']:,.2f}")
    top[2].metric("Model score", f"{decision.score:.3f}")
    top[3].metric("Latency", f"{decision.total_ms:,.0f} ms")

    if brief is None:
        st.error(f"No brief produced — {state.outcome if state else 'unknown'}")
        st.stop()

    colour = RECOMMENDATION_COLOUR.get(str(brief.recommendation), "#444")
    st.markdown(
        f"""<div style="border-left:6px solid {colour};padding:0.75rem 1rem;
        background:rgba(127,127,127,0.08);border-radius:4px;">
        <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;
        opacity:0.7;">{brief.recommendation} &nbsp;·&nbsp; confidence
        {brief.confidence:.2f}</div>
        <div style="font-size:1.05rem;margin-top:0.4rem;">{brief.narrative}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.subheader("Decisive signals")
    for e in brief.top_signals(3):
        icon = DIRECTION_ICON.get(str(e.direction), "•")
        st.markdown(
            f"**{icon} {e.signal}** &nbsp;`{e.source_tool}` &nbsp;w={e.weight:.2f}  \n{e.detail}"
        )

    if brief.missing_information:
        st.warning("Could not retrieve: " + "; ".join(brief.missing_information))

    st.subheader("Decision")
    cols = st.columns(4)
    actions = [
        ("Approve", "approve"),
        ("Block", "block"),
        ("Step-up auth", "request_step_up_auth"),
        ("Need more info", "escalate_to_analyst"),
    ]
    for col, (label, value) in zip(cols, actions, strict=True):
        if col.button(label, use_container_width=True):
            if value == "escalate_to_analyst":
                st.info("Kept in the queue.")
            else:
                _record_decision(
                    case["transaction_id"], value, agreed=value == str(brief.recommendation)
                )
                st.success(f"Recorded: {value}. Fed to the eval set and retraining data.")

    with st.expander("Full evidence bundle"):
        st.json(state.bundle.as_dict())

    with st.expander("All evidence items"):
        for e in brief.evidence:
            st.markdown(
                f"{DIRECTION_ICON.get(str(e.direction), '•')} **{e.signal}** "
                f"(`{e.source_tool}`, w={e.weight:.2f}) — {e.detail}"
            )

    with st.expander("Similar resolved cases"):
        for c in state.bundle.similar_cases:
            st.markdown(f"**{c['case_id']}** — _{c['outcome']}_  \n{c['narrative']}")

    with st.expander("Trace"):
        st.write({s.name: f"{s.duration_ms:.1f} ms" for s in state.spans})
        st.json(
            {
                "outcome": str(state.outcome),
                "repairs": state.repairs,
                "cost_usd": state.cost_usd,
                "degraded": state.degraded,
                "route": state.route,
                "policy": state.policy,
                "injection": state.injection,
                "redactions": state.redactions,
                "prompt_ids": state.prompt_ids,
            }
        )

    if show_truth:
        st.divider()
        outcome = "CONFIRMED FRAUD" if case["label"] == 1 else "CONFIRMED LEGITIMATE"
        agreed = str(brief.recommendation) == case["expected_action"]
        (st.success if agreed else st.error)(
            f"{outcome} — agent recommended {brief.recommendation} "
            f"({'agrees' if agreed else 'disagrees'})"
        )


main()
