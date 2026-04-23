# app.py
import pathlib, shutil
import typing
import time
import datetime as dt
import subprocess
import os
import signal
import streamlit as st
import dataclasses

# --- loaders ---
import docx2txt
import bs4
import markdown as md
import pypdf


def _force_quit(_signum, _frame):
   # In packaged desktop mode, close-window events can leave worker threads running.
   # Force process exit so the app quits when macOS requests termination.
   os._exit(0)
for _sig_name in ("SIGTERM", "SIGINT", "SIGHUP"):
   _sig = getattr(signal, _sig_name, None)
   if _sig is not None:
       try:
           signal.signal(_sig, _force_quit)
       except Exception:
           pass
       


# ------------------- Config -------------------
HUGGING_FACE_KEY = os.environ.get(HUGGING_FACE_KEY)
FIRECRAWL_API_KEY = os.environ.get(FIRECRAWL_API_KEY)
SERP_API_KEY = os.environl.get(SERP_API_KEY)

DATA_DIR = pathlib.Path("/tmp/IdeaStar/data")
CHROMA_DIR = "/tmp/IdeaStar/chromaDB"
COLLECTION = "real_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SYSTEM_PROMPT = """You are a research scholar. Your job is to do academic research tasks regarding the query question.
"""


# ------------------- Data model -------------------
@dataclasses.dataclass
class RawDoc:
   doc_id: str
   title: str
   text: str
   path: str
   mtime_iso: str
   filetype: str


# ------------------- Utilities: loaders -------------------
def read_text_txt(path: pathlib.Path) -> str:
   return path.read_text(encoding="utf-8", errors="ignore")

def read_text_pdf(path: pathlib.Path) -> str:
   reader = pypdf.PdfReader(str(path))
   pages = []
   for p in reader.pages:
       pages.append(p.extract_text() or "")
   return "\n".join(pages)

def read_text_docx(path: pathlib.Path) -> str:
   return docx2txt.process(str(path)) or ""

def html_to_text(html: str) -> str:
   soup = bs4.BeautifulSoup(html, "html.parser")
   for bad in soup(["script", "style", "noscript"]):
       bad.extract()
   return soup.get_text(separator=" ", strip=True)

def read_text_html(path: pathlib.Path) -> str:
   return html_to_text(path.read_text(encoding="utf-8", errors="ignore"))

def read_text_md(path: pathlib.Path) -> str:
   html = md.markdown(path.read_text(encoding="utf-8", errors="ignore"))
   return html_to_text(html)

def load_file(path: pathlib.Path) -> typing.Optional[RawDoc]:
   if not path.is_file():
       return None
   ext = path.suffix.lower()
   try:
       if ext == ".pdf":
           text = read_text_pdf(path)
           ftype = "pdf"
       elif ext == ".docx":
           text = read_text_docx(path)
           ftype = "docx"
       elif ext in (".html", ".htm"):
           text = read_text_html(path)
           ftype = "html"
       elif ext in (".md", ".markdown"):
           text = read_text_md(path)
           ftype = "markdown"
       elif ext in (".txt",):
           text = read_text_txt(path)
           ftype = "text"
       else:
           return None  # unsupported
   except Exception as e:
       st.warning(f"[loader] Skipping {path.name} due to error: {e}")
       return None
  
   text = " ".join(text.split())
   if not text.strip():
       return None
   mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
   title = path.stem.replace("_", " ").strip() or path.name
   return RawDoc(
       doc_id=str(path.resolve()),
       title=title,
       text=text,
       path=str(path.resolve()),
       mtime_iso=mtime.isoformat(),
       filetype=ftype,
   )

def walk_data_dir(data_dir: pathlib.Path) -> typing.List[RawDoc]:
   docs: typing.List[RawDoc] = []
   for p in data_dir.rglob("*"):
       rd = load_file(p)
       if rd:
           docs.append(rd)
   return docs


# -------------------- Chunk: overlapped chunk ---------------------
def chunk_text(text: str, max_chars: int, overlap: int) -> typing.List[str]:
   chunks = []
   pre_chunks = text.split("\n\n")
   for t in pre_chunks:
       t = " ".join(t.split())
       start = 0
       n = len(t)
       while start < n:
           end = min(n, start + max_chars)
           chunks.append(t[start:end])
           if end == n: break
           start = max(0, end - overlap)
   return chunks


# ------------------- Embedding + Chroma -------------------
@st.cache_resource
def get_embedder():
   import sentence_transformers
   return sentence_transformers.SentenceTransformer(EMBEDDING_MODEL)

def get_collection():
   import chromadb
   client = chromadb.PersistentClient(
       path=CHROMA_DIR,
       settings=chromadb.config.Settings(anonymized_telemetry=False),
   )
   try:
       return client.get_collection(COLLECTION)
   except Exception:
       return client.create_collection(COLLECTION)

def initialize_collection():
    import chromadb
    client = chromadb.PersistentClient(
       path=CHROMA_DIR,
       settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    try:
       client.delete_collection(COLLECTION)  # removes all stored items
    except Exception:
       pass
    return client.create_collection(COLLECTION)

def index_docs(docs: typing.List[RawDoc], embedder, max_chars: int, overlap: int):
   coll = initialize_collection()
   ids, docs_text, metas = [], [], []

   for rd in docs:
       for i, ch in enumerate(chunk_text(rd.text, max_chars=max_chars, overlap=overlap)):
           ids.append(f"{rd.doc_id}::chunk::{i}")
           docs_text.append(ch)
           metas.append({
               "title": rd.title,
               "source_path": rd.path,
               "filetype": rd.filetype,
               "last_modified": rd.mtime_iso,
               "chunk_index": i,
           })

   if not ids:
       st.info("[index] No documents to upsert.")
       return 0

   st.write(f"[index] Embedding {len(ids)} chunks…")
   vecs = embedder.encode(docs_text, normalize_embeddings=True).tolist()
   coll.upsert(ids=ids, documents=docs_text, metadatas=metas, embeddings=vecs)
   return len(ids)

def needs_reindex(docs: typing.List[RawDoc]) -> bool:
   """If any file mtime is newer than last index time marker, or marker missing."""
   marker = pathlib.Path(CHROMA_DIR) / ".last_indexed"
   if not marker.exists():
       return True
   last = dt.datetime.fromtimestamp(marker.stat().st_mtime)
   for rd in docs:
       mtime = dt.datetime.fromisoformat(rd.mtime_iso)
       if mtime > last:
           return True
   return False

def touch_index_marker():
   marker = pathlib.Path(CHROMA_DIR) / ".last_indexed"
   marker.parent.mkdir(parents=True, exist_ok=True)
   marker.write_text(str(time.time()))

def copy_answer_to_clipboard(answer: str) -> typing.Tuple[bool, str]:
   # PyInstaller apps may launch without UTF-8 locale; force it for clipboard tools.
   clip_env = dict(os.environ)
   clip_env["LANG"] = "en_US.UTF-8"
   clip_env["LC_ALL"] = "en_US.UTF-8"
   clip_env["LC_CTYPE"] = "UTF-8"

   try:
       proc = subprocess.run(
           ["pbcopy"],
           input=answer.encode("utf-8"),
           stdout=subprocess.DEVNULL,
           stderr=subprocess.PIPE,
           env=clip_env,
           check=False,
       )
       if proc.returncode == 0:
           return True, "Copied answer to clipboard."
       err = proc.stderr.decode("utf-8", errors="ignore").strip() or "unknown pbcopy error"
       # Fallback to AppleScript clipboard assignment when pbcopy fails in packaged apps.
       osa = subprocess.run(
           ["osascript", "-e", "set the clipboard to (read (POSIX file \"/dev/stdin\") as «class utf8»)"],
           input=answer.encode("utf-8"),
           stdout=subprocess.DEVNULL,
           stderr=subprocess.PIPE,
           env=clip_env,
           check=False,
       )
       if osa.returncode == 0:
           return True, "Copied answer to clipboard."
       osa_err = osa.stderr.decode("utf-8", errors="ignore").strip() or err
       return False, f"Clipboard copy failed: {osa_err}"
   except Exception as e:
       return False, f"Clipboard copy failed: {e}"


# ------------------- Retrieval + LLM -------------------
def retrieve(question: str, embedder, k: int = 5, where: typing.Optional[typing.Dict]=None) -> typing.List[typing.Dict]:
   coll = get_collection()
   qvec = embedder.encode([question], normalize_embeddings=True).tolist()
   res = coll.query(query_embeddings=qvec, n_results=k, where=where)
   out = []
   if not res or not res.get("ids") or not res["ids"][0]:
       return out
   for i in range(len(res["ids"][0])):
       out.append({
           "id": res["ids"][0][i],
           "text": res["documents"][0][i],
           "metadata": res["metadatas"][0][i],
           "score": float(res["distances"][0][i]) if "distances" in res else None
       })
   return out


def build_prompt(question: str, hits: typing.List[typing.Dict]) -> str:
   ctx = []
   for h in hits:
       m = h["metadata"]
       ctx.append(
           f"### {m.get('title')} ({m.get('source_path')})\n"
           f"[filetype={m.get('filetype')}, last_modified={m.get('last_modified')}, chunk={m.get('chunk_index')}]\n"
           f"{h['text']}"
       )
   context = "\n\n".join(ctx)
   return f"""{SYSTEM_PROMPT}

               # Context
               {context}
              
               # Question
               {question}
              
               # Answering rules
                    1. If the question is in Chinese, your answer must be in Chinese. If the question is not related to a research task, response by guiding user to ask a research task. Otherwise, do the following steps.
                    2. First, do the literature review. Look for gaps, under-explored research areas.
                    3. Second, identify key themes. Focus on topics that align with the goals and pay attention to recurring themes, particular aspects, methodologies across different studies.
                    4. Thirdly, formulate research questions. Develop specific topics based on the gaps or themes identified. Ensure these topics offer fresh perspectives or new insights. They should be clear, focused, and researchable.
                    5. Make sure the scope is neither too broad nor too narrow. The research topics should connect with existing theories or models which provide foundations for the research.
                """

def generate(prompt: str, temperature: float = 0.2) -> str:
   try:
       import huggingface_hub
       client = huggingface_hub.InferenceClient(
           provider="nscale",
           api_key=HUGGING_FACE_KEY,
           )

       resp = client.chat.completions.create(
           model="Qwen/Qwen3-4B-Instruct-2507",  #"Qwen/Qwen3-4B-Thinking-2507"
           messages=[
               {
                   "role": "user",
                   "content": prompt
               }
           ],
           stream = False,
           temperature= temperature
       )
       return resp.choices[0].message.content

       # resp = ollama.chat(
       #     model="llama3.2:3b",
       #     messages=[{"role": "user", "content": prompt}],
       #     stream=False,
       #     options={"temperature": temperature}
       # )
       # return resp["message"]["content"]

   except Exception as e:
       return f"[ERROR calling hugging face llm] {e}"
  
if "reindex_needed" not in st.session_state:
    st.session_state.reindex_needed = False
if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None
if "answer" not in st.session_state:
    st.session_state.answer = ""


# ------------------- UI -------------------
st.set_page_config(page_title="IdeaStar", layout="wide")
st.title("Your Topic Sage")

# Ensure data dir exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("本地资料库索引设置")
    top_k = st.slider("Top-K (retrieval)", min_value=1, max_value=10, value=5, step=1)
    chunk_chars = st.slider("Chunk size (chars)", min_value=300, max_value=3000, value=1200, step=100)
    chunk_overlap = st.slider("Chunk overlap (chars)", min_value=0, max_value=600, value=150, step=10)
    filt = st.selectbox("Filter by file type (optional)", options=["", "pdf", "docx", "html", "markdown", "text"])
    temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.markdown("---")
    reindex_btn = st.button("(Re)Index")

# Upload files
st.subheader("连接本地资料库corpus")

uploaded = st.file_uploader(
    "Add PDFs, DOCX, HTML, MD, or TXT",
    type=["pdf", "docx", "html", "htm", "md", "markdown", "txt"],
    accept_multiple_files='directory',
)

current_signature = tuple(sorted((f.name, f.size) for f in uploaded)) if uploaded else ()

# Streamlit reruns on every interaction; only rewrite DATA_DIR when uploads change.
if current_signature != st.session_state.uploaded_signature:
    if uploaded:
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Keep DATA_DIR aligned with the currently selected upload list.
        for f in uploaded:
            dest = DATA_DIR / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                out.write(f.read())
        st.success(f"Connected {len(uploaded)} file(s)")
    elif st.session_state.uploaded_signature not in (None, ()):
        # User removed all selected files; clear DATA_DIR as well.
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        st.info("All uploaded files removed.")

    st.session_state.uploaded_signature = current_signature

# List current files
st.subheader("当前资料corpus")
files = sorted([p for p in DATA_DIR.rglob("*") if p.is_file()])
if files:
    st.write(f"{len(files)} file(s)")
    for p in files[:50]:
        stat = p.stat()
        st.caption(f"• {p.name} — {p.suffix[1:].lower()} — {dt.datetime.fromtimestamp(stat.st_mtime)}")
else:
    st.info("No files yet. Upload above or place files")

# Indexing
embedder = get_embedder()
docs = walk_data_dir(DATA_DIR)

# Auto-index if needed or on button
st.session_state.reindex_needed = needs_reindex(docs)

if reindex_btn:
    st.session_state.reindex_needed = False
    if not docs:
        st.warning("No supported documents found to index.")
    else:
        with st.spinner("Indexing… this can take a minute on first run (downloading embedding model)."):
            embedder = get_embedder()
            n_chunks = index_docs(docs, embedder, max_chars=chunk_chars, overlap=chunk_overlap)
            touch_index_marker()
        st.success(f"Indexed {n_chunks} chunk(s)")

if st.session_state.reindex_needed:
    st.info("Index appears stale or missing. Click (Re)Index in the sidebar to build.")

st.markdown("---")
st.header("Ask")
q = st.text_input("Question", placeholder="e.g., Give me a topic about ai development?")
col1, col2 = st.columns(2)
with col1:
    ask_btn = st.button("Ask with corpus")
with col2:
    ask_direct_btn = st.button("Ask without corpus")


if ask_btn and q.strip(): 
    if st.session_state.reindex_needed:
        st.info("Index appears stale or missing. Click (Re)Index in the sidebar to build.")
    else:
        embedder = get_embedder()
        filt_dict = {"filetype": filt} if filt else None
        st.caption(f"Filter: {filt_dict if filt_dict else 'None'} • Top-K: {top_k} • Temp: {temperature}")
        with st.spinner("Retrieving relevant chunks…"):
            hits = retrieve(q, embedder, k=top_k, where=filt_dict)
        if not hits:
            st.warning("No relevant context found. Try re-indexing or broadening your query.")
            st.stop()

        st.subheader("Retrieved chunks")
        for i, h in enumerate(hits, start=1):
            m = h["metadata"]
            with st.expander(f"{i}. {m['title']}  |  {m['filetype']}  |  distance-score={h['score']}", expanded=(i == 1)):
                st.caption(f"last_modified={m['last_modified']}  •  chunk={m['chunk_index']}")
                st.write(h["text"])

        prompt = build_prompt(q, hits)
        with st.spinner("Generating grounded answer …"):
            answer = generate(prompt, temperature=temperature)
        st.session_state.answer = answer


if ask_direct_btn and q.strip():
    filt_dict = {"filetype": filt} if filt else None
    st.caption(f"Filter: {filt_dict if filt_dict else 'None'} • Top-K: {top_k} • Temp: {temperature}")
    keythemes = generate(f'Query: {q}. If the query is not a research task, ONLY response by ##NONE##. If the query is a research task, identify the key research themes in the query, ONLY return them in English joined by ;.', temperature=temperature)
    print('key themes: ', keythemes)
    results = []
    if keythemes.strip("#") != 'NONE':
        with st.spinner("Searching literatures …"):
            import firecrawl
            import serpapi
            app = firecrawl.Firecrawl(api_key=FIRECRAWL_API_KEY)
            client = serpapi.Client(api_key=SERP_API_KEY)
            search_results = client.search({
            "engine": "google_scholar",
            "q": f"recent 5 years research papers on {keythemes}",
            "num": 20,
            "hl":"en",
            })
            for result in search_results.get("organic_results", []):
                article_url = result.get("link") # This is the URL of the article
                try:
                    doc = app.scrape(article_url, timeout=2000)
                except:
                    continue
                abstract = None
                for sec in doc.markdown.split('##'):
                    if sec.strip().lower().startswith('abstract'):
                        abstract = sec
                results.insert(0,{"title": result.get('title'), "abstract": abstract})

    ctx=[]
    for paper in results:
       ctx.append(
           f"### {paper.get("title")}\n"
           f"{paper.get("abstract")}"
       )
    context = "\n\n".join(ctx)

    prompt = f"""{SYSTEM_PROMPT}

        # Context
        {context}
        
        # Question
        {q}
        
        # Answering rules
            1. If the question is in Chinese, your answer must be in Chinese. If the question is not related to a research task, response by guiding user to ask a research task. Otherwise, do the following steps.
            2. First, do the literature review. Look for gaps, under-explored research areas.
            3. Second, identify key themes. Focus on topics that align with the goals and pay attention to recurring themes, particular aspects, methodologies across different studies.
            4. Thirdly, formulate research questions. Develop specific topics based on the gaps or themes identified. Ensure these topics offer fresh perspectives or new insights. They should be clear, focused, and researchable.
            5. Make sure the scope is neither too broad nor too narrow. The research topics should connect with existing theories or models which provide foundations for the research.
        """
    with st.spinner("Generating grounded answer with QWen 3 (4B) …"):
        answer = generate(prompt, temperature=temperature)
    st.session_state.answer = answer

if st.session_state.answer:
    st.header("Answer")
    st.write(st.session_state.answer)
    if st.button("Copy Answer", key="copy_answer"):
        ok, msg = copy_answer_to_clipboard(st.session_state.answer)
        if ok:
            st.success(msg)
        else:
            st.error(msg)