# Security posture

A documented gap reads as a team that looked. A clean sweep reads as a team that
tested until it passed. Everything below is measured, and the gaps are named.

---

## Threat model

| Threat | Control | Where | Verified by |
|---|---|---|---|
| PAN / PII reaching a provider or a trace | Deterministic redaction as the first graph node, on the raw record; `assert_clean()` fails closed on the assembled prompt | `guardrails/pii.py`, `triage/graph.py::redact_pii` | `tests/security/test_pii_redaction.py` (25 cases) |
| Prompt injection via merchant, device or email fields | Structural separation (`<untrusted_field>`), pattern scanning, neutralisation, forced structured output | `guardrails/injection.py` | `tests/security/test_prompt_injection.py` + `fraudsight redteam` (40 attacks) |
| Data exfiltration through tools | Tools are read-only and allow-listed; scope comes from the transaction, not the model; unknown tool name raises | `tools/base.py::ToolRegistry` | `test_prompt_injection.py`, scope-violation path |
| Denial-of-wallet | Pre-call budget reservation, hard daily cap, breach never fails over to a second paid provider | `llm/usage.py`, `llm/router.py` | `tests/unit/test_router.py::test_budget_breach_does_not_fail_over_to_another_paid_provider` |
| Provider outage dropping transactions | Retry → fallback → deterministic provider, which cannot be configured away | `llm/router.py`, `llm/__init__.py` | `tests/integration/test_pipeline_end_to_end.py::test_degrades_to_a_brief_when_the_primary_provider_fails` |
| Model inventing evidence | Numeric groundedness check against the evidence bundle; fabricated citations blocked; tool attribution checked | `guardrails/output.py` | `tests/unit/test_output_policy.py` (13 cases) |
| Autonomous action on weak evidence | Confidence floors (0.80 act / 0.85 block), gaps-with-high-confidence blocked | `guardrails/output.py` | same |
| Secrets in logs | `SecretStr` on every key; no key is ever formatted into a string | `config.py` | review |

---

## Red team results

`fraudsight redteam` — 40 payloads across five categories, injected into the
fields an attacker actually controls (`DeviceInfo`, `P_emaildomain`,
`ProductCD`) and run through the **live pipeline**, not against the regex in
isolation.

An attack counts as blocked only if it neither changed the recommendation
towards `approve` nor produced any sign of the model reciting its instructions.

| Category | Attempts | Blocked | Matched by the scanner |
|---|---|---|---|
| instruction_override | 8 | 8 | 7 |
| role_hijack | 8 | 8 | 6 |
| decision_steering | 8 | 8 | 8 |
| exfiltration | 8 | 8 | 6 |
| delimiter_escape | 8 | 8 | 4 |
| **Total** | **40** | **40 (100%)** | **31 (78%)** |

### The number that matters more than 40/40

**Nine of the forty payloads were not matched by the scanner patterns.** They
were stopped by the layers underneath — structural separation, the forced
`TriageBrief` schema, and (for the rule-based provider) the fact that it does
not read instructions at all.

That is the system working as designed: defence in depth means no single layer
has to be complete. But it means the honest claim is *"the scanner catches
roughly four in five known payloads and the layers beneath it caught the rest"*,
not *"we block prompt injection"*.

---

## Known gaps

**1. The scanner is a pattern matcher.** It will miss novel phrasings,
non-English payloads, base64 or unicode-escaped instructions, and
instruction-shaped content that avoids the trigger vocabulary. It is defence in
depth, not a boundary. *Mitigation:* the layers beneath it do not depend on it.

**2. The 100% block rate was measured against the deterministic provider.** A
rule-based brief generator is trivially immune to persuasion — it does not read
the prompt as instructions. **The red-team number against a real LLM is not
yet measured**, and it will be lower. Re-run `fraudsight redteam` with a key
before quoting this figure anywhere it matters.

**3. Names and physical addresses are not redacted by default.** The
deterministic layer covers PAN, IBAN, SSN, email, phone and CVV. Free-text names
and addresses need the optional Presidio stage (`uv pip install -e '.[pii]'`).
IEEE-CIS contains none, so this is untested here.

**4. Non-Latin digits and unusual separators** are not handled by the PAN
detector.

**5. No per-user ACL on retrieval.** Every analyst can retrieve every case.
Correct for a single-team fraud queue; wrong the moment this serves multiple
tenants, and the filter must be applied at query time, never post-hoc.

**6. Single-process budget enforcement.** The daily cap is process-local. Two
workers get two budgets. Needs a shared counter before horizontal scaling.

**7. The tool layer trusts its own profile files.** A poisoned
`artifacts/profiles/*.json` would feed false evidence to every brief. They are
build artifacts, so this reduces to build-pipeline integrity, but there is no
signature check.

---

## Reference frameworks

Mapped against **OWASP Top 10 for LLM Applications**:

| | Status |
|---|---|
| LLM01 Prompt injection | Partial — layered, gaps published above |
| LLM02 Insecure output handling | Covered — schema-validated, no prose parsed |
| LLM03 Training data poisoning | N/A — no fine-tuning |
| LLM04 Model DoS | Covered — budget cap, breaker, timeouts |
| LLM05 Supply chain | Partial — pinned deps, no SBOM |
| LLM06 Sensitive information disclosure | Covered — redaction + outbound assert |
| LLM07 Insecure plugin design | Covered — read-only, allow-listed, scoped |
| LLM08 Excessive agency | Covered — no write tools, confidence floors, human gate |
| LLM09 Overreliance | Partial — eval suite and floor comparison exist; judge unvalidated |
| LLM10 Model theft | N/A |

**EU AI Act:** credit-adjacent fraud scoring is likely high-risk. What exists:
logging of every decision, human oversight on the ambiguous band, published
accuracy and its limits, documented data lineage. What does not: a conformity
assessment, a registered technical file, or a fairness audit signed off by
anyone. Do not deploy this in the EU on the strength of this document.

---

## Reporting

This is a portfolio project, not a service. There is no bug bounty. If you find
something wrong with the reasoning here, open an issue — a correction to a
security claim is more valuable than the claim was.
