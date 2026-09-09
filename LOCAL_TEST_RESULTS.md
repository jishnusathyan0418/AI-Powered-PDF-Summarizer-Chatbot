# Local test results - 9 September 2026

The application is running at http://127.0.0.1:8000 using Python 3.11 and the
isolated environment at `%TEMP%\pdf-insight-venv-311`.

## Passed

- `pip check`: no broken requirements.
- 10 integration tests: all passed.
- Actual PDF text extraction, MiniLM embeddings, and Chroma retrieval.
- Document replacement without retaining previous document chunks.
- Missing/invalid uploads, empty PDFs, oversized uploads, and malformed questions.
- Clear JSON responses for missing credentials and simulated model failures.
- Edge browser checks: initial state, invalid-upload retry, valid PDF upload,
  error display, safe response rendering, theme toggle, reset, and another upload.
- No browser JavaScript exceptions.
- JavaScript syntax and Git whitespace checks.

## Not verified

A live Groq-generated answer could not be tested because `GROQ_API_KEY` was absent.
Integration answer tests use a fake chat model. Browser rendering uses one explicitly
simulated response. These do not demonstrate live model correctness.

Add the key locally to this project's ignored `.env`, then rerun the browser test.
It will check live answers against two facts in its sample PDF: 23 days of annual
leave and the project codename ORCHID-742. Do not put a real key in this report.

## Fixes made

- Declared missing dependencies and constrained incompatible package versions.
- Deferred model initialization so the page and indexing can work without a Groq key.
- Fixed Reset and prevented chat before a successful upload.
- Added upload/input validation, temporary file cleanup, and readable API errors.
- Used distinct Chroma collections and clear document reset behavior.
- Rendered responses as text and restored controls after failed requests.
- Corrected the README launch command and added `run-local.ps1`.

This remains a single-user demo with global server state. Conversational follow-up
resolution, page citations, OCR, and comprehensive whole-document summaries remain
outside the implemented behavior. No application can be guaranteed correct for all
PDFs based on these sample tests.
