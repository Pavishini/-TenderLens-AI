"""Streamlit dashboard for grounded tender document analysis."""

import hashlib
import html
import os
import time

import streamlit as st

from pdf_processor import PDFProcessingError, extract_pdf_pages
from rag_engine import RAGEngine
from tender_analyzer import (
    GeminiConfigurationError,
    GeminiRequestError,
    SCALAR_FIELDS,
    analyze_eligibility,
    analyze_tender,
    answer_question,
)


st.set_page_config(
    page_title="TenderLens AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #101b35; }
    [data-testid="stSidebar"] * { color: #eef3ff; }
    [data-testid="stSidebar"] textarea {
        color: #17233c !important;
        -webkit-text-fill-color: #17233c;
        background: #ffffff;
    }
    [data-testid="stSidebar"] textarea::placeholder {
        color: #667085 !important;
        opacity: 1;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #c1cce0;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        color: #17233c;
    }
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px; color: white;
        background: linear-gradient(120deg, #14284e, #315c9b);
        margin-bottom: 1.2rem;
    }
    .hero p { color: #dce8ff; margin-bottom: 0; }
    .source-note { color: #64748b; font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _initialize_state() -> None:
    defaults = {
        "document_digest": None,
        "document_error": None,
        "document_name": None,
        "pages": None,
        "rag_engine": None,
        "tender_analysis": None,
        "eligibility_results": None,
        "eligibility_profile_digest": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _format_source(item: dict) -> str:
    page = item.get("page_number")
    return f"Page {page}" if page else "Page not identified"


def _render_field(label: str, field: dict) -> None:
    value = field.get("value") or "Not found in the tender"
    page = _format_source(field)
    with st.container(border=True):
        st.markdown(f"**{label}**")
        st.write(value)
        if field.get("evidence"):
            st.caption(f'{page} · “{field["evidence"]}”')
        elif field.get("page_number"):
            st.caption(page)


def _render_list(items: list[dict], empty_message: str) -> None:
    if not items:
        st.info(empty_message)
        return
    for item in items:
        with st.container(border=True):
            st.markdown(f"**{item['name']}**")
            evidence = item.get("evidence", "")
            citation = _format_source(item)
            st.caption(f"{citation} · “{evidence}”" if evidence else citation)


def _process_upload(uploaded_file) -> None:
    pdf_bytes = uploaded_file.getvalue()
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    if digest == st.session_state.document_digest:
        return

    st.session_state.document_digest = digest
    st.session_state.document_error = None
    st.session_state.document_name = uploaded_file.name
    st.session_state.pages = None
    st.session_state.rag_engine = None
    st.session_state.tender_analysis = None
    st.session_state.eligibility_results = None
    st.session_state.eligibility_profile_digest = None

    try:
        with st.spinner("Extracting PDF text and building the searchable index..."):
            pages = extract_pdf_pages(pdf_bytes)
            rag_engine = RAGEngine(pages)
        st.session_state.pages = pages
        st.session_state.rag_engine = rag_engine
    except (PDFProcessingError, ValueError, RuntimeError, ImportError) as exc:
        st.session_state.document_error = str(exc)
    except Exception as exc:
        st.session_state.document_error = (
            f"Could not prepare the PDF for search: {exc}"
        )


def _run_tender_analysis() -> None:
    with st.status(
        "Analyzing tender. Using Gemini 3.6 Flash, with one fast backup if needed (up to 90 seconds)...",
        expanded=False,
    ) as status:
        started = time.monotonic()
        try:
            analysis = analyze_tender(st.session_state.rag_engine)
            st.session_state.tender_analysis = analysis
            st.session_state.eligibility_results = None
            st.session_state.eligibility_profile_digest = None
            elapsed = time.monotonic() - started
            status.update(
                label=f"Tender analysis complete in {elapsed:.1f} seconds.",
                state="complete",
                expanded=False,
            )
        except (GeminiConfigurationError, GeminiRequestError) as exc:
            status.update(
                label="Tender analysis failed.",
                state="error",
                expanded=True,
            )
            st.error(str(exc))


def _status_color(status: str) -> str:
    return {
        "MET": "🟢",
        "NOT MET": "🔴",
        "NEEDS VERIFICATION": "🟡",
    }.get(status, "⚪")


_initialize_state()
st.sidebar.markdown("## TenderLens AI")
st.sidebar.caption("Grounded tender review, with source pages.")
uploaded_file = st.sidebar.file_uploader(
    "Upload tender PDF",
    type=["pdf"],
    help="Text-based PDFs are supported. Scanned/image-only documents need OCR, which is not included.",
)
st.sidebar.markdown("---")
company_profile = st.sidebar.text_area(
    "Company profile",
    key="company_profile",
    height=180,
    placeholder=(
        "Enter only verified company facts, such as annual turnover, "
        "relevant experience, certificates, and registrations."
    ),
    help="Only the information entered here is used for eligibility comparison.",
)
st.sidebar.caption(
    "Company information is user-provided and is not independently verified."
)
st.sidebar.markdown("---")
if os.getenv("GEMINI_API_KEY", "").strip():
    st.sidebar.success("Gemini API key detected")
else:
    st.sidebar.info("Add GEMINI_API_KEY to .env to enable AI analysis and answers.")

if uploaded_file is None:
    if st.session_state.document_digest is not None:
        st.session_state.document_digest = None
        st.session_state.document_error = None
        st.session_state.document_name = None
        st.session_state.pages = None
        st.session_state.rag_engine = None
        st.session_state.tender_analysis = None
        st.session_state.eligibility_results = None
        st.session_state.eligibility_profile_digest = None
else:
    _process_upload(uploaded_file)

st.markdown(
    """
    <div class="hero">
      <h1>TenderLens AI</h1>
      <p>Understand tender requirements, assess eligibility, and ask questions
      with answers grounded in your document.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, eligibility_tab, checklist_tab, ask_tab = st.tabs(
    ["Overview", "Eligibility", "Document Checklist", "Ask TenderLens"]
)

with overview_tab:
    if st.session_state.document_error:
        st.error(st.session_state.document_error)
    elif st.session_state.rag_engine is None:
        st.info("Upload a tender PDF from the sidebar to get started.")
        st.caption(
            "Text is extracted page by page. Scanned PDFs are not supported; "
            "this application does not include OCR."
        )
    else:
        pages = st.session_state.pages
        total_chars = sum(len(page.text) for page in pages)
        st.caption(
            f"**{html.escape(st.session_state.document_name or 'Tender PDF')}** "
            f"· {len(pages)} pages · {total_chars:,} readable text characters"
        )
        if st.button("Analyze tender", type="primary"):
            _run_tender_analysis()

        analysis = st.session_state.tender_analysis
        if analysis is None:
            st.info(
                "The document is indexed and ready. Analyze it to extract tender "
                "details; local question retrieval is available in the Ask tab."
            )
        else:
            st.subheader("Tender at a glance")
            top_fields = [
                ("tender_title", "Tender title"),
                ("reference_number", "Reference number"),
                ("submission_deadline", "Submission deadline"),
                ("emd_amount", "EMD amount"),
            ]
            columns = st.columns(4)
            for column, (key, label) in zip(columns, top_fields):
                field = analysis[key]
                column.metric(label, field["value"] or "Not found")
                if field["page_number"]:
                    column.caption(f'Page {field["page_number"]}')

            st.subheader("Tender details")
            detail_fields = [
                ("pre_bid_meeting_date", SCALAR_FIELDS["pre_bid_meeting_date"]),
                ("minimum_annual_turnover", SCALAR_FIELDS["minimum_annual_turnover"]),
                ("minimum_experience", SCALAR_FIELDS["minimum_experience"]),
                ("tender_fee", SCALAR_FIELDS["tender_fee"]),
            ]
            columns = st.columns(2)
            for index, (key, label) in enumerate(detail_fields):
                with columns[index % 2]:
                    _render_field(label, analysis[key])

            st.subheader("Mandatory certificates")
            _render_list(
                analysis["mandatory_certificates"],
                "No mandatory certificates were identified in the retrieved excerpts.",
            )

            st.subheader("Required documents")
            _render_list(
                analysis["required_documents"],
                "No required documents were identified in the retrieved excerpts.",
            )

with eligibility_tab:
    st.subheader("Eligibility assessment")
    st.caption(
        "Results are based only on the tender excerpts and the company profile "
        "you provide. Review the cited requirements before bidding."
    )
    analysis = st.session_state.tender_analysis
    if analysis is None:
        st.info("Upload and analyze a tender in the Overview tab first.")
    else:
        profile_digest = hashlib.sha256(company_profile.encode("utf-8")).hexdigest()
        if st.button("Compare profile with criteria", type="primary"):
            with st.status(
                "Comparing eligibility. Using Gemini 3.6 Flash, with one fast backup if needed (up to 90 seconds)...",
                expanded=False,
            ) as status:
                try:
                    results = analyze_eligibility(analysis, company_profile)
                    st.session_state.eligibility_results = results
                    st.session_state.eligibility_profile_digest = profile_digest
                    status.update(
                        label="Eligibility comparison complete.",
                        state="complete",
                        expanded=False,
                    )
                except (GeminiConfigurationError, GeminiRequestError) as exc:
                    status.update(
                        label="Eligibility comparison failed.",
                        state="error",
                        expanded=True,
                    )
                    st.error(str(exc))

        results = st.session_state.eligibility_results
        if (
            results
            and st.session_state.eligibility_profile_digest == profile_digest
        ):
            for result in results:
                with st.container(border=True):
                    st.markdown(
                        f"### {_status_color(result['status'])} "
                        f"{result['criterion']} — {result['status']}"
                    )
                    st.markdown(f"**Tender requirement:** {result['requirement']}")
                    st.write(result["explanation"])
                    if result["company_evidence"]:
                        st.caption(
                            f'Company profile evidence: “{result["company_evidence"]}”'
                        )
                    if result["requirement"] == "Not found in the tender context.":
                        st.caption("Tender source: not identified; verify in the PDF.")
                    else:
                        criterion_source = next(
                            (
                                analysis[key]
                                for key in (
                                    "minimum_annual_turnover",
                                    "minimum_experience",
                                )
                                if SCALAR_FIELDS[key] == result["criterion"]
                            ),
                            None,
                        )
                        if criterion_source is None:
                            criterion_source = next(
                                (
                                    item
                                    for item in analysis["mandatory_certificates"]
                                    if item["name"] == result["criterion"]
                                ),
                                None,
                            )
                        if criterion_source and criterion_source.get("page_number"):
                            st.caption(
                                f"Tender source: Page {criterion_source['page_number']}"
                            )
        elif company_profile.strip():
            st.info("Run the comparison to see criterion-by-criterion results.")
        else:
            st.info(
                "No company facts have been entered. Run the comparison to mark "
                "criteria as needing verification, or add verified information "
                "in the sidebar."
            )

with checklist_tab:
    st.subheader("Document checklist")
    analysis = st.session_state.tender_analysis
    if analysis is None:
        st.info("Upload and analyze a tender in the Overview tab first.")
    else:
        documents = analysis["required_documents"]
        if not documents:
            st.info("No required documents were identified in the retrieved excerpts.")
        else:
            st.caption("Track your preparation locally in this browser session.")
            digest = st.session_state.document_digest or "tender"
            for item in documents:
                item_id = hashlib.sha1(item["name"].encode("utf-8")).hexdigest()[:10]
                st.checkbox(
                    item["name"],
                    key=f"checklist_{digest}_{item_id}",
                )
                citation = _format_source(item)
                st.caption(
                    f"{citation} · {item['evidence']}"
                    if item.get("evidence")
                    else citation
                )

with ask_tab:
    st.subheader("Ask TenderLens")
    st.caption(
        "Answers use retrieved tender excerpts only. Review the page references "
        "and supporting passages before relying on an answer."
    )
    if st.session_state.rag_engine is None:
        st.info("Upload a readable tender PDF to ask a question.")
    else:
        with st.form("tender_question_form"):
            question = st.text_input(
                "Your question",
                placeholder="What is the submission deadline?",
            )
            submitted = st.form_submit_button("Ask", type="primary")
        if submitted:
            if not question.strip():
                st.warning("Enter a question about the tender.")
            else:
                with st.status(
                    "Searching the tender. Using Gemini 3.6 Flash, with one fast backup if needed (up to 90 seconds)...",
                    expanded=False,
                ) as status:
                    try:
                        retrieved = st.session_state.rag_engine.retrieve(
                            question, top_k=4
                        )
                        answer = answer_question(question, retrieved)
                        status.update(
                            label="Answer ready.",
                            state="complete",
                            expanded=False,
                        )
                        st.markdown("#### Answer")
                        st.write(answer)
                        if retrieved:
                            st.markdown("#### Supporting excerpts")
                            for item in retrieved:
                                with st.expander(f"Page {item.page_number}"):
                                    st.write(item.text)
                    except (
                        GeminiConfigurationError,
                        GeminiRequestError,
                        ValueError,
                    ) as exc:
                        status.update(
                            label="Could not answer the question.",
                            state="error",
                            expanded=True,
                        )
                        st.error(str(exc))

st.markdown("---")
st.caption(
    "TenderLens AI supports document review; it does not replace checking the "
    "original tender or obtaining professional advice."
)
