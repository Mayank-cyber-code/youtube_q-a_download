import os
import tempfile
import shutil
import re
import json
import logging
import time
import sys
from pathlib import Path
from io import BytesIO
from collections import Counter

import yt_dlp
import webvtt
import wikipedia

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

from deep_translator import GoogleTranslator
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("Missing OPENAI_API_KEY environment variable")

SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY", "840841f6e68a37b9a298ceed6161481")  # Use your key

COOKIES_PATH = "cookies.txt"
YTDLP_COOKIES_CONTENT = os.getenv("YTDLP_COOKIES_CONTENT")
if YTDLP_COOKIES_CONTENT and not os.path.exists(COOKIES_PATH):
    with open(COOKIES_PATH, "w", encoding="utf-8") as f:
        f.write(YTDLP_COOKIES_CONTENT)
    logger.info(f"Wrote YouTube cookies to {COOKIES_PATH}")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

VECTORSTORE_ROOT = Path("vectorstores")
VECTORSTORE_ROOT.mkdir(exist_ok=True)

# --- Redis helpers ---
def redis_set_str(key: str, val: str):
    redis_client.set(key, val)

def redis_get_str(key: str):
    val = redis_client.get(key)
    return val

def redis_set_json(key: str, val):
    redis_client.set(key, json.dumps(val))

def redis_get_json(key: str):
    val = redis_client.get(key)
    if val:
        return json.loads(val)
    return None

def append_to_list(key: str, val):
    redis_client.rpush(key, json.dumps(val))

def get_list(key: str):
    items = redis_client.lrange(key, 0, -1)
    if not items:
        return []
    return [json.loads(item) for item in items]

# --- FastAPI Init ---
app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Utility functions ---
def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be|shorts)\/?([a-zA-Z0-9_-]{11})"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("Could not extract YouTube video ID")

def download_subtitles(video_url: str, tmpdir: str):
    video_id = extract_video_id(video_url)

    # ScraperAPI proxy URL
    scraper_proxy = f"http://api.scraperapi.com?api_key={SCRAPERAPI_KEY}&url=" if SCRAPERAPI_KEY else None

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "outtmpl": f"{tmpdir}/%(id)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }

    if scraper_proxy:
        ydl_opts["proxy"] = scraper_proxy

    if os.path.exists(COOKIES_PATH):
        ydl_opts["cookiefile"] = COOKIES_PATH
        logger.info(f"Using YouTube cookies from {COOKIES_PATH}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
        # Check subtitle file
        path_vtt = Path(tmpdir) / f"{video_id}.en.vtt"
        if path_vtt.exists():
            logger.info(f"Subtitle downloaded: {path_vtt}")
            return str(path_vtt), info
        # fallback: any vtt in tmpdir
        for file in Path(tmpdir).glob("*.vtt"):
            logger.info(f"Using fallback subtitle: {file}")
            return str(file), info
    except Exception as e:
        msg = str(e).lower()
        if ("confirm" in msg and "bot" in msg) or "authentication" in msg or "sign in" in msg:
            logger.warning("Video requires authentication (cookies)")
            raise HTTPException(403, "Video requires authentication (cookies)")
        logger.warning(f"yt-dlp subtitle download failed: {e}")
    return None, None

def parse_subtitle(vtt_path: str) -> str:
    if not vtt_path or not Path(vtt_path).exists():
        return ""
    try:
        captions = [c.text.strip() for c in webvtt.read(vtt_path) if c.text.strip()]
        return " ".join(captions)
    except Exception as e:
        logger.warning(f"Parsing subtitles failed: {e}")
        return ""

def translate_to_english(text: str) -> str:
    if not text.strip():
        return text
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        logger.info("Translated transcript to English.")
        return translated
    except Exception as e:
        logger.warning(f"Translation error: {e}")
        return text

def fetch_wikipedia_summary(query: str) -> str:
    try:
        results = wikipedia.search(query)
        if not results:
            return ""
        summary = wikipedia.summary(results[0], sentences=5)
        return summary
    except Exception:
        return ""

def perform_eda(text: str):
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)
    unique_words = len(set(words))
    sentences = text.split('.')
    avg_sentence_len = word_count / max(len(sentences), 1)
    stopwords = set("the and is to a of in that it for on you with as this was but are have be at or from".split())
    filtered_words = [w for w in words if w not in stopwords]
    common_words = Counter(filtered_words).most_common(20)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    features = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.data
    indices = tfidf_matrix.indices
    tfidf_scores = list(zip(indices, scores))
    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    keywords = [features[i] for i, _ in tfidf_scores[:10]]
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "avg_sentence_len": round(avg_sentence_len, 2),
        "common_words": common_words,
        "tfidf_keywords": keywords,
        "sentiment": {"polarity": round(sentiment.polarity, 3), "subjectivity": round(sentiment.subjectivity, 3)}
    }

def create_wordcloud_image(text: str) -> str:
    cloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    buf = BytesIO()
    cloud.to_image().save(buf, format="PNG")
    return buf.getvalue().hex()

def summarize_text(text: str, sentences=5) -> str:
    if not text.strip() or len(text.split()) < 20:
        return text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences)
    return " ".join(str(sentence) for sentence in summary)

def save_vectorstore(vstore: FAISS, path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    vstore.save_local(str(path))

def load_vectorstore(path: Path, embeddings) -> FAISS | None:
    if not path.exists():
        return None
    try:
        return FAISS.load_local(str(path), embeddings)
    except Exception as e:
        logger.warning(f"Loading vectorstore failed: {e}")
        return None

# --- Session Memory Helpers ---
from langchain.schema import messages_from_dict, messages_to_dict

def get_memory(session_id: str) -> ConversationBufferMemory:
    data = redis_get_json(f"memory:{session_id}")
    messages = messages_from_dict(data) if data else []
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True, messages=messages)

def save_memory(session_id: str, memory: ConversationBufferMemory):
    messages = memory.load_memory_variables({}).get("history", [])
    redis_set_json(f"memory:{session_id}", messages_to_dict(messages))

# --- API endpoints ---
@app.post("/api/ask")
async def api_ask(request: Request):
    data = await request.json()
    video_url = data.get("video_url", "").strip()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "default").strip()

    if not video_url or not question:
        raise HTTPException(400, "Missing video_url or question")

    video_id = extract_video_id(video_url)
    transcript_key = f"transcript:{video_id}"
    metadata_key = f"metadata:{video_id}"
    vectorstore_dir = VECTORSTORE_ROOT / video_id
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    transcript = redis_get_str(transcript_key)
    metadata = redis_get_json(metadata_key)
    vectorstore = load_vectorstore(vectorstore_dir, embeddings) if vectorstore_dir.exists() else None

    if not transcript or vectorstore is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vtt_path, info = download_subtitles(video_url, tmpdir)
            raw_transcript = parse_subtitle(vtt_path) if vtt_path else ""
            transcript = translate_to_english(raw_transcript) if raw_transcript else fetch_wikipedia_summary(info.get("title") if info else "")
            transcript = transcript or ""
            metadata = {
                "title": info.get("title") if info else "",
                "description": info.get("description") if info else "",
            }
            redis_set_str(transcript_key, transcript)
            redis_set_json(metadata_key, metadata)

            docs = [Document(page_content=transcript, metadata={"source": video_url})]
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, overlap=100)
            chunks = splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            save_vectorstore(vectorstore, vectorstore_dir)

    memory = get_memory(session_id)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    qa_chain = ConversationalRetrievalChain(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

    start_time = time.perf_counter()
    result = qa_chain({"question": question})
    latency = time.perf_counter() - start_time

    save_memory(session_id, memory)

    answer = result.get("answer", str(result))

    append_to_list("answers_log", {
        "video_id": video_id,
        "video_url": video_url,
        "question": question,
        "answer": answer,
        "session_id": session_id,
        "latency_sec": round(latency, 3),
    })

    return JSONResponse({"answer": answer, "latency_sec": round(latency, 3)})

@app.post("/api/eda")
async def api_eda(request: Request):
    data = await request.json()
    video_url = data.get("video_url", "").strip()

    if not video_url:
        raise HTTPException(400, "Missing video_url")

    video_id = extract_video_id(video_url)
    transcript_key = f"transcript:{video_id}"
    transcript = redis_get_str(transcript_key)

    if not transcript:
        with tempfile.TemporaryDirectory() as tmpdir:
            vtt_path, _ = download_subtitles(video_url, tmpdir)
            raw_transcript = parse_subtitle(vtt_path) if vtt_path else ""
            transcript = translate_to_english(raw_transcript) if raw_transcript else ""
            if not transcript:
                raise HTTPException(404, "Transcript not found")
            redis_set_str(transcript_key, transcript)

    eda_results = perform_eda(transcript)
    wordcloud_b64 = create_wordcloud_image(transcript)
    try:
        summary = summarize_text(transcript, 5)
    except Exception as e:
        logger.warning(f"Summarization failure: {e}")
        summary = "Summary not available."

    return JSONResponse({
        "eda": eda_results,
        "wordcloud_base64": wordcloud_b64,
        "summary": summary,
        "transcript_length": len(transcript),
        "video_id": video_id,
    })

@app.get("/api/benchmark_log")
async def benchmark_log():
    logs = get_list("answers_log")
    return JSONResponse({"answers_log": logs})

@app.get("/")
async def serve_frontend():
    path = Path("static") / "frontend.html"
    if path.exists():
        return FileResponse(str(path))
    return JSONResponse({"error": "Frontend not found"}, status_code=404)

@app.get("/python-version")
async def python_version():
    return {"python_version": sys.version}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
