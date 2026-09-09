# PDF Insight: PDF question-answering chatbot

A local Flask application that extracts PDF text, splits it into overlapping chunks,
retrieves relevant passages with a lightweight in-memory TF-IDF index, and sends the
retrieved context to Groq. The lightweight index avoids downloading large ML models,
which makes the app suitable for free hosting plans with limited memory.

## Run locally (Windows PowerShell)

Python 3.11 was selected for the local test environment. For this Google Drive
workspace, use the launcher to keep dependencies on the local disk:

```powershell
.\run-local.ps1 -Install
# Later starts:
.\run-local.ps1
```

The launcher uses `%TEMP%\pdf-insight-venv-311`.
Open http://127.0.0.1:8000.

For generated answers, create an ignored `.env` file in this project directory with
`GROQ_API_KEY` set to your own Groq API key. Do not commit or share the key. The web
page and PDF indexing can work without a key; answer requests explain when it is
missing. Generating answers requires internet access to Groq.

Upload a text-based PDF, wait for the ready message, and ask a question. Reset clears
the active document and lets you upload another. Uploads are limited to 20 MB.

## Active files and settings

- `server.py`: Flask routes, temporary uploads, validation, and JSON responses.
- `worker.py`: PDF extraction, lightweight TF-IDF retrieval, and the Groq LLM.
- `templates/index.html`, `static/script.js`, `static/style.css`: browser interface.
- Chunk size: 1064 characters; target overlap: 160 characters.
- Retrieval: TF-IDF cosine similarity, up to 6 chunks.
- Generation: `openai/gpt-oss-120b` by default, configurable with `GROQ_MODEL`, temperature 0.1, output limit 256 tokens.

The other worker and exercise files are alternative examples, not imported by
`server.py`.

## Local checks

```powershell
& (Join-Path $env:TEMP 'pdf-insight-venv-311\Scripts\python.exe') -m unittest discover -s tests -v
node --check static/script.js
```

The integration checks use actual generated PDFs and real TF-IDF retrieval.
Answer-generation checks substitute a fake chat model; they
do not demonstrate that a live Groq request succeeds and do not require an API key.

## Current limitations

This is a single-user local demo. Document state is global to the server process,
not isolated by user or browser session. The index is not configured for persistence.
Conversation history is recorded but is not used to resolve follow-up questions.
Answers do not include explicit page citations. A request to summarize uses retrieved
excerpts, so it is not guaranteed to cover the whole PDF. Image-only scans require
OCR, which is not implemented. Context instructions do not guarantee factual answers.

For the optional Edge browser smoke test, start the app in another terminal, then:

```powershell
$taskPython = Join-Path $env:TEMP 'pdf-insight-venv-311\Scripts\python.exe'
& $taskPython -m pip install -r requirements-dev.txt
& $taskPython tests\browser_smoke.py
```

This test uses a generated sample handbook. If a Groq key is configured, it checks
live answers against two known facts. Otherwise, it verifies the missing-key error
and explicitly reports the live-answer test as blocked. A separate simulated
response checks safe text rendering only. Browser artifacts are saved in `.cache/`.
