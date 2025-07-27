import os
import tempfile
import re
import yt_dlp
import webvtt
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging

import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API key — required
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY environment variable not set")

# Optional YouTube cookies content from env variable
YTDLP_COOKIES_CONTENT = os.getenv("YTDLP_COOKIES_CONTENT")
COOKIES_PATH = "cookies.txt"

# Write cookies to file if content is present and file doesn't exist yet
if YTDLP_COOKIES_CONTENT and not os.path.exists(COOKIES_PATH):
    with open(COOKIES_PATH, "w", encoding="utf-8") as f:
        f.write(YTDLP_COOKIES_CONTENT)
    logger.info(f"Wrote YouTube cookies to {COOKIES_PATH}")

# Initialize OpenAI client (new SDK 1.x+)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# CORS middleware, allowing all origins - customize in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from ./static
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    frontend_path = os.path.join("static", "frontend.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return JSONResponse({"error": "Frontend not found. Upload frontend.html to /static."}, status_code=404)


def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID (11 chars) from various URL formats."""
    patterns = [
        r"(?:v=|/videos/|embed/|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Couldn't extract YouTube video ID from URL.")


def download_vtt_and_info(video_url: str, tmpdir: str):
    """
    Download English auto-generated subtitles (.vtt) and video info using yt-dlp.
    Returns: tuple (vtt_filepath or None, info dict or None)
    Raises HTTPException if YouTube requires authentication.
    """
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

    # Add cookiefile option if cookie file exists
    if os.path.exists(COOKIES_PATH):
        ydl_opts["cookiefile"] = COOKIES_PATH
        logger.info(f"Using YouTube cookies: {COOKIES_PATH}")

    info = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)

        # Check for expected vtt file
        if os.path.isfile(expected_vtt):
            logger.info(f"Found subtitle file: {expected_vtt}")
            return expected_vtt, info

        # Fallback: find any .vtt file in tmpdir
        for fname in os.listdir(tmpdir):
            if fname.endswith(".vtt"):
                path = os.path.join(tmpdir, fname)
                logger.info(f"Found subtitle file (fallback): {path}")
                return path, info

    except Exception as e:
        err_msg = str(e)
        if ("confirm you’re not a bot" in err_msg or "Sign in" in err_msg or "authentication" in err_msg.lower()):
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
    """Extract plain text transcript from a .vtt subtitle file."""
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


def ask_openai(question: str, context: str = "", model: str = "gpt-3.5-turbo") -> str:
    """Call OpenAI chat completions using the new OpenAI Python SDK."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant capable of answering questions about YouTube videos "
                "using the given transcript, title, or description, in English."
            ),
        }
    ]
    if context:
        messages.append({"role": "user", "content": f"Transcript/Context:\n{context}"})
    messages.append({"role": "user", "content": f"Q: {question}\nA:"})

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=350,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return f"Error from OpenAI: {e}"


@app.post("/api/ask")
async def api_ask(request: Request):
    """
    Expects JSON body: { "video_url": "...", "question": "..." }
    Returns JSON: { "answer": "..." }
    """
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
