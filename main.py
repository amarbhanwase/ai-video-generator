import os
import re
import uuid
import requests
import urllib.parse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
import asyncio
import edge_tts
from PIL import Image
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
from fastapi import File, UploadFile, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
import json

# Load environment variables from .env if present
load_dotenv()

# Ensure directories exist
os.makedirs("app/outputs", exist_ok=True)
os.makedirs("app/static", exist_ok=True)
os.makedirs("app/templates", exist_ok=True)

# Setup App and Templates
app = FastAPI(title="AI Video Generator")

app.mount("/outputs", StaticFiles(directory="app/outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

class RenderRequest(BaseModel):
    scenes: List[Dict[str, Any]]
    aspect_ratio: str

def parse_script(script: str) -> List[str]:
    """
    Breaks a text script into short micro-scenes.
    Targeting 3-5 words per scene for high-paced visual changes (1.5-2.5s per image).
    """
    # First split into sentences/phrases
    raw_sentences = re.split(r'(?<=[.?!,])\s+', script.strip())

    scenes = []
    for sentence in raw_sentences:
        if not sentence.strip():
            continue

        words = sentence.split()

        # If the sentence is short enough, keep it as one scene
        if len(words) <= 6:
            scenes.append(sentence.strip())
        else:
            # Chunk the longer sentences into groups of ~4 words
            chunk_size = 4
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                if chunk.strip():
                    scenes.append(chunk)

    return scenes

async def generate_audio(text: str, index: int, job_id: str) -> str:
    """
    Generates audio for a given text using edge-tts (hi-IN-MadhurNeural).
    Returns the file path to the generated audio.
    """
    output_path = f"app/outputs/{job_id}_{index}.mp3"
    try:
        # We use hi-IN-MadhurNeural for a deep, realistic male narration
        communicate = edge_tts.Communicate(text, "hi-IN-MadhurNeural")
        await communicate.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error generating audio for scene {index}: {str(e)}")
        # Create an empty/dummy file as fallback
        with open(output_path, "wb") as f:
            pass
        return output_path

import random
import time

def process_and_save_image(image_bytes: bytes, output_path: str, target_width: int, target_height: int):
    """
    Ensures downloaded image is properly resized and cropped to the requested
    dimensions (aspect ratio) before feeding into MoviePy.
    """
    import io
    from PIL import Image, ImageOps

    # Load image from bytes
    img = Image.open(io.BytesIO(image_bytes))

    # Resize and crop to fill the target dimensions exactly
    img = ImageOps.fit(img, (target_width, target_height), Image.Resampling.LANCZOS)

    # Convert to RGB to ensure PNG/JPEG compatibility without alpha channel issues in MoviePy
    img = img.convert("RGB")

    # Save the processed image
    img.save(output_path, format="JPEG")

def parse_script_with_ai(script: str) -> List[Dict[str, str]]:
    """
    Uses Gemini 1.5 Flash to split the script into 2-4 second scenes
    and generates detailed photographic image prompts for each.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")

    # Basic fallback if no API key
    if not api_key or api_key == "your_gemini_api_key_here":
        raw_scenes = parse_script(script) # the old basic chunking
        return [{"text": s, "prompt": f"Cinematic scene for: {s}"} for s in raw_scenes]

    try:
        client = genai.Client(api_key=api_key)

        system_instruction = """
        You are a director and AI prompt engineer. Break the provided script into chronological scenes.
        Each scene should be short (about 2-4 seconds of speaking time).
        For each scene, write the exact script text, and generate a highly detailed, photographic Google Flow / Midjourney style image prompt.
        Return the result EXACTLY as a JSON array of objects with keys 'text' and 'prompt'.
        """

        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[system_instruction, f"Script:\n{script}"],
            config=dict(
                response_mime_type="application/json",
            )
        )

        data = json.loads(response.text)
        # Ensure it's a list
        if isinstance(data, list) and all(isinstance(item, dict) and 'text' in item and 'prompt' in item for item in data):
            return data

        raise ValueError("Invalid JSON format from AI")

    except Exception as e:
        print(f"AI Scene generation failed, using basic parse: {e}")
        raw_scenes = parse_script(script)
        return [{"text": s, "prompt": f"Cinematic scene for: {s}"} for s in raw_scenes]

def create_dynamic_frame(t, duration, width, height, base_w, base_h, effect_type, img_clip):
    """
    Helper function to dynamically crop and scale an image frame for Ken Burns effects.
    This avoids Python closure late-binding bugs in loops.
    """
    progress = t / duration
    import numpy as np

    if effect_type == 0:
        # Zoom-out: Start cropped (zoomed in), expand to full view
        current_w = width * (1.0 + 0.1 * (1.0 - progress))
        current_h = height * (1.0 + 0.1 * (1.0 - progress))
        x_center = base_w / 2
        y_center = base_h / 2
    elif effect_type == 1:
        # Zoom-in: Start full view, shrink crop window (zoom in) over time
        current_w = width * (1.0 + 0.1 * progress)
        current_h = height * (1.0 + 0.1 * progress)
        x_center = base_w / 2
        y_center = base_h / 2
    else:
        # Pan-left: Crop window moves from right to left
        current_w = width
        current_h = height
        start_x = base_w - (width / 2)
        end_x = width / 2
        x_center = start_x + (end_x - start_x) * progress
        y_center = base_h / 2

    x1 = int(x_center - current_w / 2)
    y1 = int(y_center - current_h / 2)

    frame = img_clip.get_frame(t)
    cropped = frame[y1:y1+int(current_h), x1:x1+int(current_w)]

    pil_img = Image.fromarray(cropped)
    final_img = pil_img.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(final_img)

def create_video(job_id: str, image_paths: List[str], audio_paths: List[str], aspect_ratio: str) -> str:
    """
    Stitches images and audio into a final video using MoviePy.
    Applies dynamic Ken Burns effect to images.
    """
    output_path = f"app/outputs/{job_id}_final.mp4"

    try:
        clips = []
        width, height = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)

        for i in range(len(image_paths)):
            img_path = image_paths[i]
            aud_path = audio_paths[i]

            try:
                audio_clip = AudioFileClip(aud_path)
                duration = audio_clip.duration
            except Exception:
                duration = 2.0
                audio_clip = None

            img_clip = ImageClip(img_path).with_duration(duration)
            base_w, base_h = int(width * 1.1), int(height * 1.1)
            img_clip = img_clip.resized(new_size=(base_w, base_h))

            effect_type = i % 3

            # Use default arguments to bind loop variables to the lambda closure immediately
            make_frame_dynamic = lambda t, dur=duration, ef=effect_type, clip=img_clip: create_dynamic_frame(
                t, dur, width, height, base_w, base_h, ef, clip
            )

            from moviepy.video.VideoClip import VideoClip
            dynamic_clip = VideoClip(make_frame_dynamic, duration=duration)

            if audio_clip:
                dynamic_clip = dynamic_clip.with_audio(audio_clip)

            clips.append(dynamic_clip)

        if not clips:
            raise Exception("No clips to stitch")

        final_video = concatenate_videoclips(clips, method="chain")
        final_video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=4,
            logger=None
        )

        return output_path
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        return "/outputs/mock.mp4"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

class AnalyzeRequest(BaseModel):
    script: str
    aspect_ratio: str

@app.post("/analyze_script")
async def analyze_script_endpoint(req: AnalyzeRequest):
    if not req.script:
        raise HTTPException(status_code=400, detail="Script is empty")

    scenes = parse_script_with_ai(req.script)
    return {"scenes": scenes}

import urllib.parse
import requests

@app.post("/generate_scene_image")
async def generate_scene_image_endpoint(
    prompt: str = Form(...),
    aspect_ratio: str = Form(...)
):
    job_id = str(uuid.uuid4())[:8]
    output_path = f"app/outputs/{job_id}_pollinations.jpg"
    width, height = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)

    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&model=turbo"

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        process_and_save_image(response.content, output_path, width, height)
        return {"image_url": f"/{output_path.replace('app/', '')}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_scene_image")
async def upload_scene_image_endpoint(
    aspect_ratio: str = Form(...),
    file: UploadFile = File(...)
):
    job_id = str(uuid.uuid4())[:8]
    output_path = f"app/outputs/{job_id}_upload.jpg"
    width, height = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)

    try:
        content = await file.read()
        process_and_save_image(content, output_path, width, height)
        return {"image_url": f"/{output_path.replace('app/', '')}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RenderRequestV2(BaseModel):
    scenes: List[Dict[str, Any]]
    aspect_ratio: str
    audio_type: str # 'male', 'female', 'custom'

@app.post("/render_video")
async def render_video_endpoint(
    aspect_ratio: str = Form(...),
    audio_type: str = Form(...),
    scenes: str = Form(...), # JSON string
    custom_audio: UploadFile = File(None)
):
    import json
    scenes_data = json.loads(scenes)

    if not scenes_data:
        raise HTTPException(status_code=400, detail="Scenes data is empty")

    job_id = str(uuid.uuid4())[:8]

    audio_paths = []
    image_paths = []

    # Format image paths correctly from URL back to local path
    for s in scenes_data:
        if not s.get("image_url"):
            raise HTTPException(status_code=400, detail="Not all scenes have images")
        # map web url "/outputs/..." to local path "app/outputs/..."
        local_path = "app" + s["image_url"] if s["image_url"].startswith("/outputs") else s["image_url"]
        image_paths.append(local_path)

    if audio_type in ['male', 'female']:
        voice = "hi-IN-MadhurNeural" if audio_type == 'male' else "hi-IN-SwaraNeural"
        for i, scene_data in enumerate(scenes_data):
            text = scene_data.get("text", "")
            out_path = f"app/outputs/{job_id}_{i}.mp3"

            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(out_path)
            audio_paths.append(out_path)

        final_video_path = await run_in_threadpool(create_video, job_id, image_paths, audio_paths, aspect_ratio)

    elif audio_type == 'custom':
        if not custom_audio:
            raise HTTPException(status_code=400, detail="Custom audio file required")

        # Save custom audio
        master_audio_path = f"app/outputs/{job_id}_master.mp3"
        with open(master_audio_path, "wb") as f:
            f.write(await custom_audio.read())

        # For custom audio, we create silent clips proportional to word counts
        # Then we attach the master audio at the end.

        # Calculate relative duration based on word count
        word_counts = [max(len(s.get("text", "").split()), 1) for s in scenes_data]
        total_words = sum(word_counts)

        from moviepy import AudioFileClip
        master_clip = AudioFileClip(master_audio_path)
        total_duration = master_clip.duration
        master_clip.close()

        durations = [(wc / total_words) * total_duration for wc in word_counts]

        clips = []
        width, height = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)
        base_w, base_h = int(width * 1.1), int(height * 1.1)

        for i in range(len(image_paths)):
            img_clip = ImageClip(image_paths[i]).with_duration(durations[i])
            img_clip = img_clip.resized(new_size=(base_w, base_h))

            effect_type = i % 3
            make_frame_dynamic = lambda t, dur=durations[i], ef=effect_type, clip=img_clip: create_dynamic_frame(
                t, dur, width, height, base_w, base_h, ef, clip
            )

            from moviepy.video.VideoClip import VideoClip
            dynamic_clip = VideoClip(make_frame_dynamic, duration=durations[i])
            clips.append(dynamic_clip)

        final_video = concatenate_videoclips(clips, method="chain")
        master_audio_clip = AudioFileClip(master_audio_path)
        final_video = final_video.with_audio(master_audio_clip)

        final_video_path = f"app/outputs/{job_id}_final.mp4"

        def write_vid():
            final_video.write_videofile(
                final_video_path,
                fps=24,
                codec="libx264",
                audio_codec="aac",
                preset="ultrafast",
                threads=4,
                logger=None
            )

        await run_in_threadpool(write_vid)

    video_url = f"/{final_video_path.replace('app/', '')}" if final_video_path.startswith("app/") else final_video_path

    return {
        "status": "success",
        "video_url": video_url
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
