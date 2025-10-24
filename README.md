# AI Summarizer — EchoNotes

**EchoNotes** — created by **A. Aryan**  
A lightweight, privacy-first extractive summarizer built with Streamlit.  
Main app file: `app1.py`

---

## 🔍 What this app does (exactly)
EchoNotes takes text from three input modes and produces a concise extractive summary plus helpful metadata:

- **Paste text** — paste meeting transcripts, notes, or any text.  
- **Article URL** — fetches article body (uses `newspaper3k` and falls back to `trafilatura` when necessary).  
- **Upload PDF** — extracts selectable text from PDFs (uses `PyPDF2`).  
- **Example text** — quick sample to test the app.

Outputs:
- Extractive summary (selectable number of sentences)  
- Top keywords and a keyword frequency bar chart  
- Word cloud visualization (if `wordcloud` available)  
- Heuristic action-item extraction (TODOs, directives)  
- Estimated reading time

---

## 🧠 Algorithms (what the dropdown options mean)
The app supports three extractive summarization algorithms. Dropdown values and full forms shown in the UI:

- **LexRank** — *Lexical Rank* (Graph-based sentence centrality using sentence similarity and PageRank). Best for factual news / concise summaries.  
- **LSA** — *Latent Semantic Analysis* (matrix-decomposition topic coverage using SVD). Best for theme-heavy / long documents.  
- **TextRank** — (Graph-based ranking using word overlap; similar idea to LexRank). Fast, often very readable.

Recommended quick guide inside the app:  
- Short news → LexRank / TextRank (2–4 sentences)  
- Long essays / papers → LSA (4–6 sentences)  
- Meeting transcripts → 3–5 sentences + enable Action Items

  ## Live Project Link:
  https://aisummarizerecho.streamlit.app/ 
