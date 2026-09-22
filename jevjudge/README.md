# jevjudge

An OpenAI-compatible **structured-output judge** backed by TypeSafe's **Jev** (System One) model.

Point any router that asks an LLM judge for a JSON verdict (NVIDIA NeMo Switchyard's
`llm_classifier` capability, escalation, and custom modes; composite and stage-router classifiers)
at `jevjudge` instead of a chat model. It compiles the verdict's JSON Schema into Jev questions,
asks Jev once (~0.3 s, ~$0.04 per million input tokens), and returns a schema-valid verdict
with calibrated probabilities. Low-confidence verdicts can cascade to a real LLM judge or
abstain so the router fails open.

See the repository README for the full design and Switchyard configs.

```bash
export TYPESAFE_API_KEY=...        # or JEVJUDGE_TRANSPORT=openrouter + OPENROUTER_API_KEY
jevjudge --port 8090
```
