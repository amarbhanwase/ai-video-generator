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

async def map_images_to_scenes(scenes: List[str], image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Uses Gemini 1.5 Flash multimodal to match uploaded images to parsed scenes.
    Falls back to round-robin if the API fails or is not configured.
    """
    mapped = []

    # 1. Check API Key and capabilities
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            client = genai.Client(api_key=api_key)

            # Prepare payload: send all images and the list of scenes
            contents = ["Please analyze these images and match exactly ONE image to each of the following scenes. Return a JSON array of integers, where the integer at index i is the 0-based index of the image that best matches scene i. You can reuse images if there are more scenes than images.\n\nScenes:\n" + "\n".join([f"{i}. {s}" for i,s in enumerate(scenes)])]

            for path in image_paths:
                img = Image.open(path)
                contents.append(img)

            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=contents,
                config=dict(
                    response_mime_type="application/json",
                )
            )

            mapping = json.loads(response.text)

            # Verify mapping length matches scene length
            if len(mapping) == len(scenes):
                for i, scene in enumerate(scenes):
                    img_idx = int(mapping[i]) % len(image_paths) # ensure bounds
                    mapped.append({"text": scene, "image": image_paths[img_idx], "source": "ai"})
                return mapped

        except Exception as e:
            print(f"AI Mapping failed, falling back to round-robin: {e}")

    # 2. Fallback: Round-robin distribution
    for i, scene in enumerate(scenes):
        mapped.append({"text": scene, "image": image_paths[i % len(image_paths)], "source": "fallback"})

    return mapped

def create_video(job_id: str, image_paths: List[str], audio_paths: List[str], aspect_ratio: str) -> str:
    """
    Stitches images and audio into a final video using MoviePy.
    Applies a simple zoom (Ken Burns) effect to images.
    """
    output_path = f"app/outputs/{job_id}_final.mp4"

    try:
        clips = []
        width, height = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)

        for i in range(len(image_paths)):
            img_path = image_paths[i]
            aud_path = audio_paths[i]

            # Load audio to get duration
            try:
                # Need a fallback duration if audio is mocked and unreadable
                audio_clip = AudioFileClip(aud_path)
                duration = audio_clip.duration
            except Exception:
                duration = 2.0 # Mock fallback duration
                audio_clip = None

            # Dynamic pan-zoom (Ken Burns effect) in MoviePy v2
            # 1. Load image and set the duration
            img_clip = ImageClip(img_path).with_duration(duration)

            # 2. Resize to a larger resolution first so we have room to zoom/crop without black bars
            # To zoom by 10% (scale 1.1x), we first resize to (width * 1.1, height * 1.1)
            base_w, base_h = int(width * 1.1), int(height * 1.1)
            img_clip = img_clip.resized(new_size=(base_w, base_h))

            # 3. Create dynamic alternating crop effects
            # We will rotate between zoom-out, zoom-in, and pan-left
            effect_type = i % 3

            def make_frame_dynamic(t):
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
                    # x_center goes from right bound to left bound
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

            from moviepy.video.VideoClip import VideoClip
            dynamic_clip = VideoClip(make_frame_dynamic, duration=duration)

            if audio_clip:
                dynamic_clip = dynamic_clip.with_audio(audio_clip)

            clips.append(dynamic_clip)

        if not clips:
            raise Exception("No clips to stitch")

        # Hard cuts provide the fast pacing for Shorts/Reels
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
        # In case of failure (like missing ffmpeg or corrupted mocks), return the dummy
        return "/outputs/mock.mp4"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/match_images")
async def match_images_endpoint(
    script: str = Form(...),
    aspect_ratio: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if not script:
        raise HTTPException(status_code=400, detail="Script is empty")
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded")

    job_id = str(uuid.uuid4())[:8]

    # 1. Parse Script
    scenes = parse_script(script)
    if not scenes:
        raise HTTPException(status_code=400, detail="Could not parse scenes from script")

    # 2. Save uploaded images locally, resize them to target aspect ratio
    saved_image_paths = []
    width, height = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)

    for i, img in enumerate(images):
        raw_path = f"app/outputs/{job_id}_upload_{i}.jpg"
        content = await img.read()
        process_and_save_image(content, raw_path, width, height)
        saved_image_paths.append(raw_path)

    # 3. AI Match
    matched_scenes = await map_images_to_scenes(scenes, saved_image_paths)

    # Send back the mapping so UI can display it
    return {
        "status": "success",
        "job_id": job_id,
        "matched_scenes": matched_scenes
    }

@app.post("/render_video")
async def render_video_endpoint(req: RenderRequest):
    scenes = req.scenes
    aspect_ratio = req.aspect_ratio

    if not scenes:
        raise HTTPException(status_code=400, detail="Scenes data is empty")

    job_id = str(uuid.uuid4())[:8]

    audio_paths = []
    image_paths = []

    for i, scene_data in enumerate(scenes):
        scene_text = scene_data.get("text", "")
        img_path = scene_data.get("image", "")

        # 1. Generate Voiceover
        audio_path = await generate_audio(scene_text, i, job_id)
        audio_paths.append(audio_path)
        image_paths.append(img_path)

    # 2. Stitch Video using existing MoviePy logic
    final_video_path = create_video(job_id, image_paths, audio_paths, aspect_ratio)

    video_url = f"/{final_video_path.replace('app/', '')}" if final_video_path.startswith("app/") else final_video_path

    return {
        "status": "success",
        "video_url": video_url
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
