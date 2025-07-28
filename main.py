import os
import tempfile
import re
import json
import base64
import logging
import asyncio
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
from langchain.chains import RetrievalQA

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI


from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load environment variables and config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY environment variable not set")

COOKIES_PATH = "cookies.txt"
YTDLP_COOKIES_CONTENT = os.getenv("YTDLP_COOKIES_CONTENT")
if YTDLP_COOKIES_CONTENT and not os.path.exists(COOKIES_PATH):
    with open(COOKIES_PATH, "w", encoding="utf-8") as f:
        f.write(YTDLP_COOKIES_CONTENT)
    logger.info(f"Wrote YouTube cookies to {COOKIES_PATH}")

# --- Simple in-memory cache (could swap with Redis, file-based cache, or DB) ---
cache = {
    "embeddings": {},  # video_id -> FAISS vectorstore
    "transcripts": {},  # video_id -> transcript text
    "metadata": {},  # video_id -> video metadata dict
    "answers_log": [],  # list of dicts for benchmarking answers
}

# Initialize summarization pipeline (optional)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    frontend_path = os.path.join("static", "frontend.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return JSONResponse(
        {"error": "Frontend not found. Upload frontend.html to /static."}, status_code=404
    )


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

        # fallback to any .vtt in tmpdir
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
    # Basic cleaning and tokenizing for word count
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)
    unique_words = len(set(words))
    avg_sentence_len = len(text.split('.')) and word_count / len(text.split('.'))

    # Most common words excluding stopwords (basic)
    stopwords = set([
        "the", "and", "is", "to", "a", "of", "in", "that", "it", "for", "on", "you",
        "with", "as", "this", "was", "but", "are", "have", "be", "at", "or", "from"
    ])
    filtered_words = [w for w in words if w not in stopwords]
    common_words = Counter(filtered_words).most_common(20)

    # TF-IDF keywords (top 10)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    indices = tfidf_matrix[0].indices
    data = tfidf_matrix.data
    features = vectorizer.get_feature_names_out()
    tfidf_scores = sorted(zip(indices, data), key=lambda x: x[1], reverse=True)[:10]
    tfidf_keywords = [features[idx] for idx, score in tfidf_scores]

    # Sentiment analysis
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
    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    return base64_img


@app.post("/api/ask")
async def api_ask(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    video_url = (data.get("video_url") or "").strip()
    question = (data.get("question") or "").strip()

    if not video_url or not question:
        raise HTTPException(status_code=400, detail="Missing video_url or question.")

    video_id = None
    try:
        video_id = extract_video_id(video_url)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    try:
        # Check cache first for embeddings and transcript
        if video_id in cache["embeddings"] and video_id in cache["transcripts"]:
            logger.info(f"Cache hit for video_id={video_id}")
            transcript = cache["transcripts"][video_id]
            vector_store = cache["embeddings"][video_id]
            metadata = cache["metadata"].get(video_id, {})
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                vtt_file, yt_info = download_vtt_and_info(video_url, tmpdir)
                transcript = parse_transcript_from_vtt(vtt_file) if vtt_file else ""

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

                # Split documents for embeddings
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                split_docs = text_splitter.split_documents(documents)

                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vector_store = FAISS.from_documents(split_docs, embeddings)

                # Cache for reuse
                cache["transcripts"][video_id] = transcript
                cache["embeddings"][video_id] = vector_store
                cache["metadata"][video_id] = metadata

        # Setup LangChain QA chain
        chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=retriever)

        # Retrieve answer
        answer = qa_chain.run(question)

        # Log Q&A for benchmarking (basic structure)
        cache["answers_log"].append({
            "video_id": video_id,
            "video_url": video_url,
            "question": question,
            "answer": answer
            # Future: Include timestamps, user ratings, model info, etc.
        })

        return JSONResponse({"answer": answer})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/api/eda")
async def api_eda(request: Request):
    """
    EDA endpoint that returns transcript analytics & visualizations for a given video URL.
    """
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
        # Use cached transcript or fetch transcript fresh
        if video_id in cache["transcripts"]:
            transcript = cache["transcripts"][video_id]
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                vtt_file, _ = download_vtt_and_info(video_url, tmpdir)
                transcript = parse_transcript_from_vtt(vtt_file) if vtt_file else ""

            if not transcript:
                raise HTTPException(status_code=404, detail="Transcript not found.")

            cache["transcripts"][video_id] = transcript

        eda_results = transcript_eda_analysis(transcript)
        wordcloud_img = create_wordcloud_image(transcript)

        # Optionally add summarization (experimental)
        try:
            summary = summarizer(transcript, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
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
    """
    Endpoint to fetch logged Q&A pairs for benchmarking analysis.
    """
    return JSONResponse({"answers_log": cache["answers_log"]})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
