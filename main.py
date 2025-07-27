import os
import tempfile
import re
import yt_dlp
import webvtt
import wikipedia
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load OpenAI API key ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY environment variable not set")

# Initialize OpenAI client (new SDK 1.x+)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Handle YouTube cookies ---
COOKIES_PATH = "cookies.txt"  # relative path

YTDLP_COOKIES_CONTENT = os.getenv("YTDLP_COOKIES_CONTENT")
if YTDLP_COOKIES_CONTENT:
    # Write cookies content to file if not exists
    if not os.path.exists(COOKIES_PATH):
        with open(COOKIES_PATH, "w", encoding="utf-8") as f:
            f.write(YTDLP_COOKIES_CONTENT)
        logger.info(f"Wrote YouTube cookies to {COOKIES_PATH}")

app = FastAPI()

# CORS middleware (allow all; customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
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
    
    # Add cookiefile option if cookies file exists
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
    """
    Search and get the Wikipedia summary for the given query.
    Returns empty string if not found or error.
    """
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            logger.info(f"No Wikipedia results for query: {query}")
            return ""
        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=5)
        logger.info(f"Wikipedia summary found for '{page_title}'")
        return summary
    except Exception as e:
        logger.warning(f"Wikipedia search failed for query '{query}': {e}")
        return ""


def ask_openai(question: str, context: str = "", model: str = "gpt-3.5-turbo") -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant capable of answering questions about YouTube videos "
                "using the given transcript, Wikipedia summary, title, or description, in English."
            ),
        }
    ]
    if context:
        messages.append({"role": "user", "content": f"Context:\n{context}"})
    messages.append({"role": "user", "content": f"Q: {question}\nA:"})

    try:
        completion = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, max_tokens=350
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return f"Error from OpenAI: {e}"


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

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            vtt_file, yt_info = download_vtt_and_info(video_url, tmpdir)
            transcript = parse_transcript_from_vtt(vtt_file) if vtt_file else ""

            title = (yt_info.get("title") if yt_info else "") or ""
            description = (yt_info.get("description") if yt_info else "") or ""

            if transcript:
                context = transcript
            elif title or description:
                wiki_summary = get_wikipedia_summary(title)
                if wiki_summary:
                    context = wiki_summary
                else:
                    context = f"Video title: {title}\nDescription: {description}"
            else:
                context = ""

            answer = ask_openai(question, context)
            return JSONResponse({"answer": answer})

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
