import os
import tempfile
import shutil
import re
import json
import base64
import logging
import time
import sys
from pathlib import Path
from io import BytesIO
from collections import Counter

import yt_dlp
import webvtt
import wikipedia
import wikipedia.exceptions

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

from deep_translator import GoogleTranslator
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ------------------ Configuration ------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY environment variable not set")

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

# ------------------ Redis Helpers ------------------

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
    return [json.loads(i) for i in items] if items else []

# ------------------ FastAPI App ------------------

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------ Utility functions ------------------

def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})"
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    raise ValueError("Could not parse YouTube video ID")

def download_subtitles(video_url: str, tmpdir: str):
    video_id = extract_video_id(video_url)
    expected_vtt = VECTORSTORE_ROOT / f"{video_id}.en.vtt"
    ydl_opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "outtmpl": f"{tmpdir}/%(id)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }
    if os.path.exists(COOKIES_PATH):
        ydl_opts["cookiefile"] = COOKIES_PATH
        logger.info(f"Using YouTube cookies at {COOKIES_PATH}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
        # Return expected vtt if exists
        path_vtt = Path(tmpdir) / f"{video_id}.en.vtt"
        if path_vtt.exists():
            logger.info(f"Found subtitle: {path_vtt}")
            return str(path_vtt), info
        # fallback to first vtt
        for file in Path(tmpdir).glob("*.vtt"):
            logger.info(f"Using fallback subtitle: {file}")
            return str(file), info
    except Exception as e:
        err_msg = str(e).lower()
        if ("confirm" in err_msg and "bot" in err_msg) or "authentication" in err_msg or "sign in" in err_msg:
            logger.warning("Requires authentication cookies")
            raise HTTPException(403, "YouTube video requires authentication (cookies).")
        logger.warning(f"yt_dlp subtitles download failed: {e}")
    return None, None

def parse_subtitle(vtt_path: str) -> str:
    if not vtt_path or not Path(vtt_path).is_file():
        return ""
    try:
        captions = [c.text.strip() for c in webvtt.read(vtt_path) if c.text.strip()]
        return " ".join(captions)
    except Exception as e:
        logger.warning(f"Failed to parse vtt file: {e}")
        return ""

def translate_text_to_en(text: str) -> str:
    if not text.strip():
        return text
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        logger.info("Translated transcript to English.")
        return translated
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

def get_wikipedia_summary(query: str) -> str:
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return ""
        page_title = search_results[0]
        return wikipedia.summary(page_title, sentences=5)
    except Exception:
        return ""

def transcript_eda(text: str):
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)
    unique_words = len(set(words))
    avg_sentence_len = len(text.split(".")) and word_count / max(1, len(text.split(".")))
    stopwords = {"the", "and", "is", "to", "a", "of", "in", "that", "it", "for",
                 "on", "you", "with", "as", "this", "was", "but", "are", "have",
                 "be", "at", "or", "from"}
    filtered_words = [w for w in words if w not in stopwords]
    common_words = Counter(filtered_words).most_common(20)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.data
    indices = tfidf_matrix.indices
    tfidf_scores = list(zip(indices, scores))
    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    keywords = [feature_names[i] for i, _ in tfidf_scores[:10]]
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
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def summarize_text_sumy(text: str, sentences_count: int = 5) -> str:
    if not text or len(text.split()) < 20:
        return text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

def save_vectorstore_to_disk(vectorstore: FAISS, dirpath: Path):
    if dirpath.exists():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(dirpath))

def load_vectorstore_from_disk(dirpath: Path, embeddings) -> FAISS | None:
    if not dirpath.exists():
        return None
    try:
        return FAISS.load_local(str(dirpath), embeddings)
    except Exception as e:
        logger.warning(f"Could not load vectorstore from disk: {e}")
        return None

# Memory helpers

from langchain.schema import messages_from_dict, messages_to_dict

def get_memory(session_id: str) -> ConversationBufferMemory:
    key = f"memory:{session_id}"
    data = redis_get_json(key)
    messages = messages_from_dict(data) if data else []
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, messages=messages)
    return memory

def save_memory(session_id: str, memory: ConversationBufferMemory):
    key = f"memory:{session_id}"
    data = messages_to_dict(memory.load_memory_messages())
    redis_set_json(key, data)

# API endpoints

@app.post("/api/ask")
async def api_ask(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    video_url = (data.get("video_url") or "").strip()
    question = (data.get("question") or "").strip()
    session_id = (data.get("session_id") or "default").strip()

    if not video_url or not question:
        raise HTTPException(400, "Missing video_url or question")

    try:
        video_id = extract_video_id(video_url)
    except Exception as e:
        raise HTTPException(400, str(e))

    transcript_key = f"transcript:{video_id}"
    metadata_key = f"metadata:{video_id}"
    vecstore_dir = VECTORSTORE_ROOT / video_id

    transcript = redis_get_str(transcript_key)
    metadata = redis_get_json(metadata_key)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vectorstore = None
    if transcript and vecstore_dir.exists():
        vectorstore = load_vectorstore_from_disk(vecstore_dir, embeddings)

    if not transcript or vectorstore is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vtt_path, info = download_subtitles(video_url, tmpdir)
            transcript_text = ""
            if vtt_path:
                transcript_text = parse_subtitle(vtt_path)
                transcript_text = translate_text_to_en(transcript_text)
            else:
                transcript_text = get_wikipedia_summary(info.get("title") if info else "")

            transcript = transcript_text or ""
            metadata = {
                "title": info.get("title") if info else "",
                "description": info.get("description") if info else "",
            }

            # Save transcripts and metadata to Redis
            redis_set_str(transcript_key, transcript)
            redis_set_json(metadata_key, metadata)

            # Prepare documents and create vectorstore
            docs = [Document(page_content=transcript, metadata={"source": video_url})]
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            vectorstore = FAISS.from_documents(chunks, embeddings)
            save_vectorstore_to_disk(vectorstore, vecstore_dir)

    memory = get_memory(session_id)
    retriever = vectorstore.as_retriever()
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=retriever,
        memory=memory,
    )

    start_time = time.perf_counter()
    result = qa_chain({"question": question})
    latency = time.perf_counter() - start_time

    # Save updated memory
    save_memory(session_id, memory)

    answer = result.get("answer") or result

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
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    video_url = (data.get("video_url") or "").strip()

    if not video_url:
        raise HTTPException(400, "Missing video_url")

    try:
        video_id = extract_video_id(video_url)
    except Exception as e:
        raise HTTPException(400, str(e))

    transcript_key = f"transcript:{video_id}"
    transcript = redis_get_str(transcript_key)

    if not transcript:
        with tempfile.TemporaryDirectory() as tmpdir:
            vtt_path, _ = download_subtitles(video_url, tmpdir)
            transcript_text = ""
            if vtt_path:
                transcript_raw = parse_subtitle(vtt_path)
                transcript_text = translate_text_to_en(transcript_raw)
            transcript = transcript_text or ""

            if not transcript:
                raise HTTPException(404, "Transcript not found")

            redis_set_str(transcript_key, transcript)

    eda_results = transcript_eda(transcript)
    wordcloud_b64 = create_wordcloud_image(transcript)
    try:
        summary = summarize_text_sumy(transcript, sentences_count=5)
    except Exception as e:
        logger.warning(f"Summarizer failed: {e}")
        summary = "Summary unavailable"

    return JSONResponse({
        "eda": eda_results,
        "wordcloud_base64": wordcloud_b64,
        "summary": summary,
        "transcript_length_chars": len(transcript),
        "video_id": video_id,
    })

@app.get("/api/benchmark_log")
async def benchmark_log():
    logs = get_list("answers_log")
    return JSONResponse({"answers_log": logs})

@app.get("/")
async def serve_frontend():
    index_path = Path("static") / "frontend.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"error": "Frontend not found."}, status_code=404)

@app.get("/python-version")
async def python_version():
    return {"python_version": sys.version}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
