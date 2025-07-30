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

SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY", "840841f3a6e68c37a29881d961d5c481")

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
    allow_origins=["*"],  # For production, restrict origins as needed
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

    # Prepare ScraperAPI proxy URL
    scraper_proxy = None
    if SCRAPERAPI_KEY:
        scraper_proxy = f"http://api.scraperapi.com?api_key={SCRAPERAPI_KEY}&url="

    ydl_opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "outtmpl": f"{tmpdir}/%(id)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }

    # Pass ScraperAPI proxy to yt-dlp if set
    if scraper_proxy:
        ydl_opts["proxy"] = scraper_proxy
        logger.info(f"Using ScraperAPI proxy via {scraper_proxy}")

    if os.path.exists(COOKIES_PATH):
        ydl_opts["cookiefile"] = COOKIES_PATH
        logger.info(f"Using YouTube cookies at {COOKIES_PATH}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)

        # Check if subtitle downloaded
        path_vtt = Path(tmpdir) / f"{video_id}.en.vtt"
        if path_vtt.exists():
            logger.info(f"Found subtitle: {path_vtt}")
            return str(path_vtt), info

        # fallback to first *.vtt if available
        for file in Path(tmpdir).glob("*.vtt"):
            logger.info(f"Using fallback subtitle: {file}")
            return str(file), info

    except Exception as e:
        err_msg = str(e).lower()
        if ("confirm" in err_msg and "bot" in err_msg) or "authentication" in err_msg or "sign in" in err_msg:
            logger.warning("YouTube video requires authentication (cookies)")
            raise HTTPException(403, "YouTube video requires authentication (cookies).")
        logger.warning(f"yt_dlp subtitles download failed: {e}")

    return None, None

# Rest of your code (parse_subtitle, translate_text_to_en, get_wikipedia_summary, etc.) remains the same

# (Keep all other functions unchanged from previous main.py)

# Your API endpoints (`/api/ask`, `/api/eda`, etc.) also stay unchanged

# You only need to update your Dockerfile or environment to set SCRAPERAPI_KEY if desired.

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
