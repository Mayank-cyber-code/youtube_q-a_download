import os
import tempfile
import re
import json
import base64
import logging
import time
import sys
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
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

from deep_translator import GoogleTranslator
import redis
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
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
redis_client = redis.from_url(REDIS_URL, decode_responses=False)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Redis helper functions ---

def redis_set(key: str, value):
    redis_client.set(key, pickle.dumps(value))

def redis_get(key: str):
    val = redis_client.get(key)
    return pickle.loads(val) if val else None

def redis_set_json(key: str, value):
    redis_client.set(key, json.dumps(value).encode('utf-8'))

def redis_get_json(key: str):
    val = redis_client.get(key)
    return json.loads(val.decode('utf-8')) if val else None

def append_answer_log(entry):
    redis_client.rpush("answers_log", json.dumps(entry))

def get_all_answer_logs():
    logs = redis_client.lrange("answers_log", 0, -1)
    return [json.loads(log.decode('utf-8')) if isinstance(log, bytes) else json.loads(log) for log in logs]

# --- Helper functions ---

def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|/videos/|embed/|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Couldn't extract YouTube video ID from URL.")

def download_vtt_and_info(video_url: str, tmpdir: str):
    video_id = extract_video_id(video_url)
    expected_vtt = os.path.join(tmpdir, f"{video_id}.en.vtt")
    ydl_opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    if os.path.exists(COOKIES_PATH):
        ydl_opts["cookiefile"] = COOKIES_PATH
        logger.info(f"Using YouTube cookies: {COOKIES_PATH}")
    info = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
        if os.path.isfile(expected_vtt):
            logger.info(f"Found subtitle file: {expected_vtt}")
            return expected_vtt, info
        for fname in os.listdir(tmpdir):
            if fname.endswith(".vtt"):
                vtt_path = os.path.join(tmpdir, fname)
                logger.info(f"Found subtitle file (fallback): {vtt_path}")
                return vtt_path, info
    except Exception as e:
        err_msg = str(e).lower()
        if "confirm you're not a bot" in err_msg or "sign in" in err_msg or "authentication" in err_msg:
            logger.warning("YouTube video requires authentication (cookies).")
            raise HTTPException(
                status_code=403,
                detail=(
                    "This YouTube video requires authentication (cookies). "
                    "Please try a different public video, or ask the app owner to provide valid cookies."
                ),
            )
        logger.warning(f"yt-dlp subtitle download failed: {e}")
    return None, info

def parse_transcript_from_vtt(vtt_path: str) -> str:
    if not vtt_path or not os.path.isfile(vtt_path):
        return ""
    try:
        captions = []
        for caption in webvtt.read(vtt_path):
            if caption.text.strip():
                captions.append(caption.text.strip())
        return " ".join(captions).strip()
    except Exception as e:
        logger.warning(f"Failed to parse VTT file: {e}")
        return ""

def translate_text_to_en(text: str) -> str:
    if not text.strip():
        return text
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        logger.info("Translated transcript to English using deep_translator.")
        return translated
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

def get_transcript_and_translate(vtt_path: str) -> str:
    transcript = parse_transcript_from_vtt(vtt_path)
    return translate_text_to_en(transcript)

def get_wikipedia_summary(query: str) -> str:
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            logger.info(f"No Wikipedia results for query: {query}")
            return ""
        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=5)
        logger.info(f"Wikipedia summary found for '{page_title}'")
        return summary
    except (wikipedia.DisambiguationError, wikipedia.PageError, Exception) as e:
        logger.warning(f"Wikipedia search failed for query '{query}': {e}")
        return ""

def transcript_eda_analysis(text: str):
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)
    unique_words = len(set(words))
    avg_sentence_len = len(text.split('.')) and word_count / len(text.split('.'))
    stopwords = set([
        "the", "and", "is", "to", "a", "of", "in", "that", "it", "for", "on", "you",
        "with", "as", "this", "was", "but", "are", "have", "be", "at", "or", "from"
    ])
    filtered_words = [w for w in words if w not in stopwords]
    common_words = Counter(filtered_words).most_common(20)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    indices = tfidf_matrix[0].indices
    data = tfidf_matrix.data
    features = vectorizer.get_feature_names_out()
    tfidf_scores = sorted(zip(indices, data), key=lambda x: x[1], reverse=True)[:10]
    tfidf_keywords = [features[idx] for idx, score in tfidf_scores]
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)
    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "avg_sentence_len": round(avg_sentence_len, 2) if avg_sentence_len else 0,
        "common_words": common_words,
        "tfidf_keywords": tfidf_keywords,
        "sentiment": {"polarity": polarity, "subjectivity": subjectivity}
    }

def create_wordcloud_image(text: str):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    buf = BytesIO()
    wordcloud.to_image().save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def summarize_text_sumy(text: str, sentences_count: int = 5):
    if not text or len(text.split()) < 20:
        return text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join(str(sentence) for sentence in summary)

from langchain.schema import messages_from_dict, messages_to_dict

def get_memory(session_id: str) -> ConversationBufferMemory:
    key = f"memory:{session_id}"
    data = redis_get_json(key)
    messages = messages_from_dict(data) if data else []
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        messages=messages
    )
    return memory

def save_memory(session_id: str, memory: ConversationBufferMemory):
    key = f"memory:{session_id}"
    data = messages_to_dict(memory.load_memory_messages())
    redis_set_json(key, data)

@app.post("/api/ask")
async def api_ask(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    video_url = (data.get("video_url") or "").strip()
    question = (data.get("question") or "").strip()
    session_id = (data.get("session_id") or "default").strip()

    if not video_url or not question:
        raise HTTPException(status_code=400, detail="Missing video_url or question.")

    video_id = None
    try:
        video_id = extract_video_id(video_url)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    try:
        transcript_key = f"transcript:{video_id}"
        embeddings_key = f"embeddings:{video_id}"
        metadata_key = f"metadata:{video_id}"

        transcript = redis_get(transcript_key)
        vector_store = redis_get(embeddings_key)
        metadata = redis_get_json(metadata_key)

        if transcript and vector_store:
            logger.info(f"Redis cache hit for video {video_id}")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                vtt_file, yt_info = download_vtt_and_info(video_url, tmpdir)
                transcript = get_transcript_and_translate(vtt_file) if vtt_file else ""

                title = (yt_info.get("title") if yt_info else "") or ""
                description = (yt_info.get("description") if yt_info else "") or ""
                metadata = {"title": title, "description": description}

                documents = []
                if transcript:
                    documents.append(Document(page_content=transcript, metadata={"source": video_url}))
                else:
                    wiki_summary = get_wikipedia_summary(title)
                    if wiki_summary:
                        documents.append(Document(page_content=wiki_summary, metadata={"source": "wikipedia"}))
                    else:
                        base_context = f"Video title: {title}\nDescription: {description}"
                        documents.append(Document(page_content=base_context, metadata={"source": video_url}))

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                split_docs = text_splitter.split_documents(documents)

                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vector_store = FAISS.from_documents(split_docs, embeddings)

                redis_set(transcript_key, transcript)
                redis_set(embeddings_key, vector_store)
                redis_set_json(metadata_key, metadata)

        memory = get_memory(session_id)
        retriever = vector_store.as_retriever()
        chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

        qa_chain = ConversationalRetrievalChain.from_llm(
            chat_model,
            retriever,
            memory=memory,
        )

        start_time = time.perf_counter()
        result = qa_chain({"question": question})
        latency = time.perf_counter() - start_time

        answer = result["answer"] if (isinstance(result, dict) and "answer" in result) else result

        save_memory(session_id, memory)

        log_entry = {
            "video_id": video_id,
            "video_url": video_url,
            "question": question,
            "answer": answer,
            "latency_sec": round(latency, 3),
            "session_id": session_id
        }
        append_answer_log(log_entry)

        return JSONResponse({"answer": answer, "latency_sec": round(latency, 3)})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/api/eda")
async def api_eda(request: Request):
    try:
        data = await request.json()
        video_url = (data.get("video_url") or "").strip()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    if not video_url:
        raise HTTPException(status_code=400, detail="Missing video_url.")

    try:
        video_id = extract_video_id(video_url)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    try:
        transcript_key = f"transcript:{video_id}"
        transcript = redis_get(transcript_key)
        if not transcript:
            with tempfile.TemporaryDirectory() as tmpdir:
                vtt_file, _ = download_vtt_and_info(video_url, tmpdir)
                transcript = get_transcript_and_translate(vtt_file) if vtt_file else ""
            if not transcript:
                raise HTTPException(status_code=404, detail="Transcript not found.")
            redis_set(transcript_key, transcript)

        eda_results = transcript_eda_analysis(transcript)
        wordcloud_img = create_wordcloud_image(transcript)

        try:
            summary = summarize_text_sumy(transcript, sentences_count=5)
        except Exception as e:
            summary = "Summarization unavailable"
            logger.warning(f"Summarization failed: {e}")

        response_data = {
            "eda": eda_results,
            "wordcloud_base64": wordcloud_img,
            "summary": summary,
            "transcript_length_chars": len(transcript),
            "video_id": video_id,
        }

        return JSONResponse(response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in EDA: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/api/benchmark_log")
def api_get_benchmark_log():
    logs = get_all_answer_logs()
    return JSONResponse({"answers_log": logs})

@app.get("/")
def serve_frontend():
    frontend_path = os.path.join("static", "frontend.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return JSONResponse({"error": "Frontend not found. Upload frontend.html to /static."}, status_code=404)

@app.get("/python-version")
def python_version():
    return {"python_version": sys.version}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
