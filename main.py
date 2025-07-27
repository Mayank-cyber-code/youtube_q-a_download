import os
import tempfile
import yt_dlp
import webvtt
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY environment variable not set")
openai.api_key = OPENAI_API_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_video_id(url: str) -> str:
    import re
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Couldn't find YouTube Video ID in URL")

def download_vtt_and_info(video_url: str, tmpdir: str):
    """
    Download .vtt subtitles and get info (title + description) with yt-dlp.
    Returns: (subtitle_file_or_None, yt_info_dict)
    """
    video_id = extract_video_id(video_url)
    vtt_path = os.path.join(tmpdir, f"{video_id}.en.vtt")
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
        if os.path.exists(vtt_path):
            return vtt_path, info
        # Try fallback with just "_en" (sometimes yt-dlp output template varies)
        for fname in os.listdir(tmpdir):
            if fname.endswith(".vtt"):
                return os.path.join(tmpdir, fname), info
    except Exception as e:
        pass
    return None, info

def parse_transcript_from_vtt(vtt_path: str) -> str:
    if not vtt_path or not os.path.exists(vtt_path):
        return ""
    try:
        captions = []
        for caption in webvtt.read(vtt_path):
            if caption.text.strip():
                captions.append(caption.text)
        transcript_text = " ".join(captions)
        return transcript_text.strip()
    except Exception:
        return ""

def ask_openai(question: str, context: str = "", model: str = "gpt-3.5-turbo") -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant capable of answering questions about YouTube videos using the given transcript, title, or description, in English.",
        }
    ]
    if context:
        messages.append({"role": "user", "content": f"Transcript:\n{context}"})
    messages.append({"role": "user", "content": f"Q: {question}\nA:"})

    try:
        # Using streaming is optional; here we use one-shot completion for simplicity
        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=350,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from OpenAI: {e}"

@app.post("/api/ask")
async def api_ask(request: Request):
    """
    Expects: JSON { video_url: str, question: str }
    Returns: JSON { answer: str }
    """
    req = await request.json()
    video_url = req.get("video_url")
    question = req.get("question")

    if not video_url or not question:
        raise HTTPException(status_code=400, detail="Missing video_url or question")

    with tempfile.TemporaryDirectory() as tmpdir:
        vtt_file, yt_info = download_vtt_and_info(video_url, tmpdir)
        transcript_text = parse_transcript_from_vtt(vtt_file) if vtt_file else ""

        # Set up fallbacks: title & description via yt_info
        title = (yt_info.get("title") if yt_info else None) or ""
        description = (yt_info.get("description") if yt_info else None) or ""

        # Choose best available context
        if transcript_text:
            context = transcript_text
        elif title or description:
            context = f"Video title: {title}\nDescription: {description}"
        else:
            context = ""

        answer = ask_openai(question, context)
        return JSONResponse({"answer": answer})

@app.get("/")
def root():
    return {"status": "YouTube Q&A API is running."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
