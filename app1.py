# app.py
import streamlit as st
from summarizer import (
    get_text_from_url,
    extract_text_from_pdfbytes,
    summarize,
    extract_keywords,
    estimate_reading_time,
    extract_action_items,
)
import base64
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

st.set_page_config(page_title="EchoNotes — Smart Summarizer", page_icon="📝", layout="wide")

# -------------------------
# Header (stylish) - updated text per request
# -------------------------
SVG_HEADER = """
<div style="background: linear-gradient(90deg,#071028,#0b1b2b); padding:26px; border-radius:10px;">
  <h1 style="color:white; margin:0; font-family: Inter, sans-serif;">📝 EchoNotes By:- A. Aryan</h1>
  <p style="color:#cbd5e1; margin-top:6px; font-size:14px;">EchoNotes is a lightweight, privacy-friendly extractive summarizer for notes, articles and PDFs — built to help you read faster and extract action items.</p>
</div>
"""
st.markdown(SVG_HEADER, unsafe_allow_html=True)
st.write("")

# -------------------------
# Layout: input | results
# -------------------------
left, right = st.columns((1, 1.4))

with left:
    st.subheader("Input")
    input_mode = st.radio("Input type", ("Paste text", "Article URL", "Upload PDF", "Example text"), index=0)
    user_text = ""
    uploaded_file = None

    sample_text = (
        "OpenAI released a new model update today. The update focuses on improved reasoning capabilities "
        "and cost efficiency. Developers should test their prompts and adjust to token usage. "
        "Teams must update SDKs by next week. Action: schedule regression tests and update documentation."
    )

    if input_mode == "Paste text":
        user_text = st.text_area("Paste your text here", height=240, placeholder="Paste meeting transcript, article text, or notes...")
    elif input_mode == "Article URL":
        url = st.text_input("Paste article URL", placeholder="https://example.com/article")
        if st.button("Fetch article"):
            if url.strip() == "":
                st.warning("Please paste a URL first.")
            else:
                with st.spinner("Fetching article..."):
                    fetched = get_text_from_url(url)
                    if not fetched:
                        st.error("Couldn't extract text from the URL. Try pasting the article text or upload a PDF.")
                    else:
                        user_text = fetched
                        st.success("Article fetched. Scroll to Results.")
    elif input_mode == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                bytes_data = uploaded_file.read()
                extracted = extract_text_from_pdfbytes(bytes_data)
                if not extracted:
                    st.error("Could not extract text from PDF or PDF has images (scan). Try text PDF.")
                else:
                    user_text = extracted
                    st.success("PDF text extracted. Scroll to Results.")
    else:
        user_text = st.text_area("Example text (editable)", value=sample_text, height=220)

    st.markdown("---")
    st.subheader("Summarization settings")
    method = st.selectbox("Algorithm", ("lexrank", "lsa", "textrank"))
    sentences = st.slider("Number of sentences in summary", min_value=1, max_value=8, value=3)
    show_reading_time = st.checkbox("Show estimated reading time", value=True)
    extract_actions = st.checkbox("Extract action items", value=True)
    top_k_keywords = st.slider("Top keywords", min_value=3, max_value=15, value=7)

    # important: create button BEFORE using it
    run = st.button("Generate summary")

with right:
    st.subheader("Result")
    result_area = st.empty()
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        algo_card = st.empty()
    with meta_col2:
        info_card = st.empty()

    summary_box = st.empty()
    keywords_box = st.empty()
    action_box = st.empty()
    chart_box = st.empty()
    download_box = st.empty()
    wordcloud_box = st.empty()

# -------------------------
# Algorithm info (full forms + short explanation)
# -------------------------
with st.expander("Algorithm info — what do these mean?"):
    st.markdown(
        """
        **LexRank** (Lexical Rank) — *Graph-based sentence centrality*.  
        Sentences are nodes; similarity between sentences become edges. A PageRank-style algorithm finds the most central (representative) sentences.

        **LSA** (Latent Semantic Analysis) — *Topic / matrix decomposition method*.  
        Uses SVD to find hidden topics in the text and selects sentences that cover those topics (good for theme-focused summaries).

        **TextRank** — *Graph-based ranking using word overlap*.  
        Similar to LexRank but uses word co-occurrence/overlap as its similarity signal. Fast and often very readable.

        **Use this guide:**  
        - Short news / factual pieces → **LexRank** or **TextRank**.  
        - Long essays / concept-heavy text → **LSA** for coverage of themes.  
        - Try all three on the same text to compare outputs.
        """
    )

# -------------------------
# Run summarizer
# -------------------------
if run:
    if not user_text or len(user_text.strip()) < 20:
        st.warning("Please provide at least ~20 words to summarize (paste text, URL, or upload a PDF).")
    else:
        with st.spinner("Running summarizer..."):
            try:
                summary = summarize(user_text, sentences=sentences, method=method)
            except Exception as e:
                summary = f"Summarization engine error: {e}"

            keywords = extract_keywords(user_text, top_n=top_k_keywords)
            reading = estimate_reading_time(user_text) if show_reading_time else None
            actions = extract_action_items(user_text) if extract_actions else []

        # Meta cards
        algo_card.markdown(f"**Algorithm:** `{method}`  \n**Sentences:** {sentences}")
        info_text = f"**Words:** {len(user_text.split())}"
        if reading:
            info_text += f"  \n**Est. read:** {reading}"
        info_card.markdown(info_text)

        # Summary output
        summary_box.markdown("### 🧾 Summary")
        if summary.startswith("Input too short") or summary.startswith("Could not") or summary.startswith("Unable") or summary.startswith("Summarization engine error"):
            summary_box.info(summary)
        else:
            summary_box.write(summary)

        # Keywords and bar chart
        if keywords:
            keywords_box.markdown("### 🔑 Top Keywords")
            keywords_box.write(", ".join(keywords))

            counts = Counter([w.lower().strip(".,") for w in user_text.split() if w.isalpha()])
            kw_counts = [counts.get(k, 0) for k in keywords]
            fig, ax = plt.subplots(figsize=(6, 2.2))
            ax.barh(range(len(keywords))[::-1], kw_counts[::-1])
            ax.set_yticks(range(len(keywords))[::-1])
            ax.set_yticklabels(keywords[::-1])
            ax.invert_yaxis()
            ax.set_xlabel("Frequency")
            plt.tight_layout()
            chart_box.pyplot(fig)
        else:
            keywords_box.info("No keywords found.")

        # Word cloud
        try:
            wc = WordCloud(width=600, height=300, background_color=None, mode="RGBA")
            wc_img = wc.generate(" ".join(keywords) if keywords else user_text)
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.imshow(wc_img, interpolation="bilinear")
            ax2.axis("off")
            plt.tight_layout()
            wordcloud_box.pyplot(fig2)
        except Exception:
            # non-fatal
            wordcloud_box.info("Word cloud not available (install 'wordcloud' and 'matplotlib').")

        # Action items
        if actions:
            action_box.markdown("### ✅ Action Items (extracted)")
            for i, a in enumerate(actions, 1):
                action_box.write(f"{i}. {a}")
        else:
            action_box.info("No obvious action items found (try longer text or include directives).")

        # Download summary
        def make_downloadable(text, filename="echonotes_summary.txt"):
            b = text.encode()
            b64 = base64.b64encode(b).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">⬇️ Download summary</a>'
            return href

        download_box.markdown(make_downloadable(summary), unsafe_allow_html=True)
        result_area.success("Done — results are shown below.")

# -------------------------
# Footer: how to use + best approach + credit
# -------------------------
st.markdown("---")
st.markdown("## How to use EchoNotes (quick guide)")
st.markdown(
    """
    1. **Paste text** for quick notes, chat transcripts, or meeting text.  
    2. **Use Article URL** for news and blogs (works best on standard news/article pages), Sometimes it may not work due to public or private restrictions. (Use text paste in that case)   
    3. **Upload PDF** for documents — works best with selectable text PDFs (not scanned images).  

    **Best approach for best results**
    - Short factual news (1–2 pages): use **LexRank** or **TextRank**, 2–4 sentences.  
    - Long essays / research / conceptual docs: use **LSA** and increase to 4–6 sentences to cover themes.  
    - Meeting transcripts: use 3–5 sentences and enable Action Items to extract TODOs.  
    - If a fetched article fails, copy-paste the article text directly (some pages are JS heavy and can't be scraped).
    """
)

st.markdown("### About")
st.markdown(
    '''
    EchoNotes — created by **A. Aryan**
    - An AI summarizer designed to help you extract the key points, keywords, and action items from notes, articles, and PDFs quickly. 
    - Use it to speed up reading and to generate concise takeaways for study, work, or research.
    '''
    )
