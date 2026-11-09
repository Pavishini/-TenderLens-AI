# TenderLens AI

TenderLens AI is a Streamlit application for reviewing tender PDFs. It extracts
page-aware text, finds relevant passages for questions, summarizes tender
requirements, and compares those requirements with a company profile supplied
by the user.

> **Important:** TenderLens AI is an aid for document review, not legal,
> financial, or procurement advice. Verify every requirement against the
> original tender. The PDFs in `samples/` are fictional test documents.

## Features

- Upload text-based tender PDFs and extract text from each page with PyMuPDF.
- Keep page numbers attached to extracted excerpts and retrieved text chunks.
- Split text into overlapping chunks, embed them with Sentence Transformers,
  and retrieve relevant chunks using a FAISS similarity index.
- Ask questions about a tender and receive answers grounded in retrieved
  passages, with supporting page references.
- Extract tender title and reference, issuing organization when available,
  deadlines, pre-bid details, financial and experience criteria, EMD, fees,
  certificates, and required documents.
- Compare tender criteria with a user-entered company profile. Missing or
  unverified facts are not assumed.
- Review results in the Overview, Eligibility, Document Checklist, and Ask
  TenderLens tabs.
- Use Gemini through Google's `google-genai` SDK. Temporary model availability
  or timeout errors trigger one attempt with a backup model.

## Technology

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
- FAISS
- Google Gemini API and the `google-genai` SDK
- `python-dotenv`

## Project structure

```text
TenderLens-AI/
├── .streamlit/
│   └── config.toml
├── samples/
│   ├── sample_tender_it_services.pdf
│   └── sample_tender_solar.pdf
├── .env.example
├── .gitignore
├── app.py
├── pdf_processor.py
├── rag_engine.py
├── tender_analyzer.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11 or later
- A Gemini API key for tender analysis, eligibility assessment, and AI answers
- Internet access the first time the sentence embedding model is downloaded

PDF text extraction and local vector search do not require a Gemini API key.
Image-only or scanned PDFs are not supported because OCR is not implemented.

## Run on Windows (PowerShell)

Open PowerShell in the project directory and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env`, replace `your_gemini_api_key_here` with your own Gemini API key,
then start the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually
`http://localhost:8501`). Keep PowerShell open while using the app; press
**Ctrl+C** in that window to stop it.

## Run on macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Add your Gemini API key to `.env`, then run:

```bash
streamlit run app.py
```

## Gemini configuration

The app reads the key from the project `.env` file:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
```

Optional model overrides:

```dotenv
GEMINI_MODEL=gemini-3.6-flash
GEMINI_FALLBACK_MODEL=gemini-3.5-flash-lite
```

Never commit `.env`, Streamlit secrets, or real API keys. Only share the
placeholder values in `.env.example`. The application sends retrieved tender
excerpts (and the company profile during eligibility checks) to the Gemini API.
Do not analyze confidential documents unless you are authorized to share their
content with that service. Gemini requests are configured with `store=False`;
this does not replace checking the provider's current data-handling terms.

## Sample documents

Use the fictional, text-based PDFs in `samples/` to explore the interface.
They are for local testing only and are not real procurement opportunities.
