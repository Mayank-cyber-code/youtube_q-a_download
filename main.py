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
from langchain.text_splitter import RecursiveCharacterSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

from langchain.memory import ConversationMemory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import Summarizer as TextRankSummarizer  # alias for clarity

from deep_translator import GoogleTranslator
import redis

import textblob
try:
    textblob.download_corpora.download_all()
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")


# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("Missing OPENAI_API_KEY environment variable")

SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY", "840841cde6842...")  # Replace with your actual key

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
def redis_set(key: str, val):
    redis_client.set(key, val)

def redis_set_json(key: str, val):
    redis_client.set(key, json.dumps(val))

def redis_get(key: str):
    return redis_client.get(key)

def redis_get_json(key: str):
    val = redis_client.get(key)
    return json.loads(val) if val else None

def append_list(key: str, val):
    redis_client.rpush(key, json.dumps(val))

def get_list(key: str):
    res = redis_client.lrange(key, 0, -1)
    if not res:
        return []
    return [json.loads(x) for x in res]


# --- FastAPI app & middleware ---
app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Utility functions ---

def extract_video_id(url: str) -> str:
    for pattern in [r"(?:v=|\/videos\/|embed\/|youtu\.be\/|shorts)\/?([a-zA-Z0-9_-]{11})"]:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Failed to extract YouTube video ID.")


def download_subtitles(video_url: str, tmpdir: str):
    video_id = extract_video_id(video_url)

    proxy_base_url = None
    if SCRAPERAPI_KEY:
        proxy_base_url = f"http://api.scraperapi.com?api_key={SCRAPERAPI_KEY}&url="

    options = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "outtmpl": f"{tmpdir}/%(id)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }
    if proxy_base_url:
        options["proxy"] = proxy_base_url
        logger.info(f"Using ScraperAPI proxy base URL: {proxy_base_url}")

    if os.path.exists(COOKIES_PATH):
        options["cookiefile"] = COOKIES_PATH
        logger.info(f"Using cookies from {COOKIES_PATH}")

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(video_url)
        path_vtt = Path(tmpdir) / f"{video_id}.en.vtt"
        if path_vtt.exists():
            logger.info(f"Subtitle found at {path_vtt}")
            return str(path_vtt), info

        # Fallback to first available .vtt
        for f in Path(tmpdir).glob("*.vtt"):
            logger.info(f"Using fallback subtitle {f}")
            return str(f), info
    except Exception as e:
        txt = str(e).lower()
        if "confirm" in txt and "bot" in txt or "authentication" in txt or "sign in" in txt:
            logger.warning("Video requires authentication (cookies)")
            raise HTTPException(status_code=403, detail="Video requires authentication (cookies).")
        logger.warning(f"yt_dlp subtitle download error: {e}")
    return None, None


def parse_subtitle(vtt_path: str) -> str:
    if not vtt_path or not Path(vtt_path).exists():
        return ""
    try:
        captions = [cap.text.strip() for cap in webvtt.read(vtt_path) if cap.text.strip()]
        return " ".join(captions)
    except Exception as e:
        logger.warning(f"Subtitle parsing error: {e}")
        return ""


def translate_to_english(text: str) -> str:
    if not text.strip():
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text


def fetch_wikipedia_summary(query: str) -> str:
    try:
        results = wikipedia.search(query)
        if results:
            return wikipedia.summary(results[0], sentences=5)
    except Exception:
        pass
    return ""


def perform_eda(text: str):
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)
    unique_words = len(set(words))
    sentences = list(filter(None, (sentence.strip() for sentence in text.split('.'))))
    avg_sentence_len = word_count / max(1, len(sentences))
    stopwords = {"the", "and", "is", "to", "a", "of", "in", "that", "it", "for", "on", "you", "with", "as", "this", "was", "but", "are", "have", "be", "at", "or", "from"}
    filtered_words = [w for w in words if w not in stopwords]
    common_words = Counter(filtered_words).most_common(20)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.data
    indices = tfidf_matrix.indices
    tfidf_scores = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
    keywords = [feature_names[i] for i, _ in tfidf_scores[:10]]
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "avg_sentence_len": round(avg_sentence_len, 2),
        "common_words": common_words,
        "tfidf_keywords": keywords,
        "sentiment": {"polarity": round(sentiment.polarity, 3), "subjectivity": round(sentiment.subjectivity, 3)},
    }


def create_wordcloud(text: str) -> str:
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def summarize_text(text: str, sentences: int = 5) -> str:
    if not text.strip() or len(text.split()) < 20:
        return text
    parser = PlaintextParser(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences)
    return " ".join(str(sentence) for sentence in summary)


def save_vectorstore(vstore: FAISS, path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True)
    vstore.save_local(str(path))


def load_vectorstore(path: Path, embeddings) -> FAISS | None:
    if not path.exists():
        return None
    try:
        return FAISS.load_local(str(path), embeddings)
    except Exception as e:
        logger.warning(f"Vectorstore loading failed: {e}")
        return None


# Session memory helpers
from langchain.schema import messages_from_dict, messages_to_dict


def get_memory(session_id: str):
    raw = redis_get_json(f"memory:{session_id}")
    messages = messages_from_dict(raw) if raw else []
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True, messages=messages)


def save_memory(session_id: str, memory):
    messages = memory.load_memory().messages if hasattr(memory, "load_memory") else []
    redis_set_json(f"memory:{session_id}", messages_to_dict(messages))


# --- API Endpoints ---

@app.post("/api/ask")
async def api_ask(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    video_url = data.get("video_url", "").strip()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "default").strip()

    if not video_url or not question:
        raise HTTPException(status_code=400, detail="You must provide video_url and question.")

    video_id = extract_video_id(video_url)
    transcript_key = f"transcript:{video_id}"
    metadata_key = f"metadata:{video_id}"
    vectorstore_path = VECTORSTORE_ROOT / video_id
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    transcript = redis_get(transcript_key)
    metadata = redis_get_json(metadata_key)
    vectorstore = load_vectorstore(vectorstore_path, embeddings) if vectorstore_path.exists() else None

    if not transcript or vectorstore is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vtt_path, info = download_subtitles(video_url, tmpdir)
            raw_transcript = parse_subtitle(vtt_path) if vtt_path else None
            if raw_transcript:
                transcript = translate_to_english(raw_transcript)
            else:
                transcript = fetch_wikipedia_summary(info.get("title")) if info else None
            if not transcript:
                raise HTTPException(status_code=404, detail="Unable to retrieve transcript or summary.")
            metadata = {
                "title": info.get("title") if info else "",
                "description": info.get("description") if info else "",
            }
            redis_set(transcript_key, transcript)
            redis_set_json(metadata_key, metadata)

            # Validate non-empty transcript text
            if not transcript.strip():
                raise HTTPException(status_code=400, detail="Transcript is empty after processing.")

            documents = [Document(page_content=transcript, metadata={"source": video_url})]
            splitter = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(documents)

            if not chunks:
                raise HTTPException(status_code=400, detail="Failed to split transcript into chunks.")

            try:
                vectorstore = FAISS.from_documents(chunks, embeddings)
            except Exception as e:
                logger.error(f"FAISS vectorstore creation error: {e}")
                raise HTTPException(status_code=500, detail=f"Error creating vectorstore: {e}")

            save_vectorstore(vectorstore, vectorstore_path)

    memory = get_memory(session_id)
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    qa_chain = ConversationalRetrievalChain(llm=chat_model, retriever=vectorstore.as_retriever(), memory=memory)

    start_time = time.perf_counter()
    result = qa_chain({"question": question})
    latency = time.perf_counter() - start_time

    save_memory(session_id, memory)

    answer = result.get("answer", str(result))

    append_list("answers_log", {
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
        raise HTTPException(status_code=400, detail="Invalid JSON")

    video_url = data.get("video_url", "").strip()

    if not video_url:
        raise HTTPException(status_code=400, detail="Missing video_url")

    video_id = extract_video_id(video_url)
    transcript_key = f"transcript:{video_id}"

    transcript = redis_get(transcript_key)
    if not transcript:
        with tempfile.TemporaryDirectory() as tmpdir:
            vtt_path, _ = download_subtitles(video_url, tmpdir)
            raw_transcript = parse_subtitle(vtt_path) if vtt_path else None
            if raw_transcript:
                transcript = translate_to_english(raw_transcript)
            else:
                transcript = None
            if not transcript:
                raise HTTPException(status_code=404, detail="Transcript not found.")
            redis_set(transcript_key, transcript)

    eda_result = perform_eda(transcript)
    wordcloud_b64 = create_wordcloud(transcript)
    try:
        summary = summarize_text(transcript)
    except Exception:
        summary = "Summary unavailable."

    return JSONResponse({
        "eda": eda_result,
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
async def root():
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
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
