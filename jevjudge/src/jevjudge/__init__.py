"""jevjudge: a Jev (TypeSafe System One) backed judge behind an OpenAI chat-completions facade.

Any router that asks an LLM judge for a JSON verdict with a `response_format` JSON Schema
(NVIDIA NeMo Switchyard's capability, escalation, and custom classifier modes among them)
can point that judge at jevjudge instead. jevjudge compiles the schema into Jev questions,
asks Jev once, and hands back a schema-valid JSON verdict with calibrated probabilities.
"""

__version__ = "0.1.0"
