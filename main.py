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

# OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY environment variable not set")

# Instantiate OpenAI client with API key using new SDK (v1+)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# Enable CORS middleware (allow all origins)
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
    else:
        return JSONResponse({"error": "Frontend not found. Please upload frontend.html to /static."}, status_code=404)


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL (11 chars)."""
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
    Download English auto-generated subtitle (.vtt) and video info using yt-dlp.
    Returns: (vtt_filepath or None, yt_info dict or None)
    """
    video_id = extract_video_id(video_url)
    vtt_path_expected = os.path.join(tmpdir, f"{video_id}.en.vtt")
    ydl_opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    info = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
        if os.path.isfile(vtt_path_expected):
            logger.info(f"Subtitle file found: {vtt_path_expected}")
            return vtt_path_expected, info
        # fallback: find any .vtt file in tmpdir
        for entry in os.listdir(tmpdir):
            if entry.endswith(".vtt"):
                filepath = os.path.join(tmpdir, entry)
                logger.info(f"Subtitle file found (fallback): {filepath}")
                return filepath, info
    except Exception as e:
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
        logger.warning(f"Could not parse VTT: {e}")
        return ""


def ask_openai(question: str, context: str = "", model: str = "gpt-3.5-turbo") -> str:
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
    POST endpoint expecting JSON: { "video_url": "...", "question": "..." }
    Returns: JSON { "answer": "..." }
    """
    try:
        req = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    video_url = req.get("video_url", "").strip()
    question = req.get("question", "").strip()

    if not video_url or not question:
        raise HTTPException(status_code=400, detail="Missing video_url or question")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            vtt_file, yt_info = download_vtt_and_info(video_url, tmpdir)
            transcript_text = parse_transcript_from_vtt(vtt_file) if vtt_file else ""
            title = (yt_info.get("title") if yt_info else "") or ""
            description = (yt_info.get("description") if yt_info else "") or ""

            if transcript_text:
                context = transcript_text
            elif title or description:
                context = f"Video title: {title}\nDescription: {description}"
            else:
                context = ""

            answer = ask_openai(question, context)
            return JSONResponse({"answer": answer})

    except ValueError as ve:
        # e.g. invalid URL video ID extraction
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
