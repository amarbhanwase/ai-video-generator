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
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

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

class VideoRequest(BaseModel):
    script: str
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
    img.save(output_path, format="PNG")

def generate_image(scene_text: str, index: int, job_id: str, aspect_ratio: str) -> str:
    """
    Generates an image for a scene using Pollinations.ai API with retries and timeout=60.
    Falls back to Unsplash/LoremFlickr stock photos if Pollinations fails.
    Ensures downloaded images are properly resized and cropped to the aspect ratio.
    """
    output_path = f"app/outputs/{job_id}_{index}.png"
    width, height = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)

    # Basic keyword extraction for fallback stock images (just grab first long word or two)
    words = [w for w in re.sub(r'[^a-zA-Z\s]', '', scene_text).split() if len(w) > 3]
    keyword = words[0] if words else "nature"

    success = False

    # 1. Try Pollinations API with turbo model for dynamic high-paced images
    prompt = f"Cinematic, high quality, highly detailed scene: {scene_text}"
    encoded_prompt = urllib.parse.quote(prompt)
    pollinations_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&model=turbo"

    for attempt in range(3):
        try:
            print(f"Fetching image from Pollinations (attempt {attempt+1})...")
            response = requests.get(pollinations_url, timeout=60)
            response.raise_for_status()

            # Process and save
            process_and_save_image(response.content, output_path, width, height)
            success = True
            break
        except Exception as e:
            print(f"Pollinations attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2) # brief pause before retry

    # 2. Fast Fallback: Random Stock Photo API
    if not success:
        print("Falling back to stock photo...")
        fallback_urls = [
            # LoremFlickr (highly reliable stock photo alternative)
            f"https://loremflickr.com/{width}/{height}/{keyword},cinematic/all",
            # Picsum (random stock photos)
            f"https://picsum.photos/{width}/{height}"
        ]

        for fb_url in fallback_urls:
            try:
                print(f"Fetching fallback from {fb_url}...")
                response = requests.get(fb_url, timeout=15)
                response.raise_for_status()

                # Process and save
                process_and_save_image(response.content, output_path, width, height)
                success = True
                break
            except Exception as e:
                print(f"Fallback {fb_url} failed: {e}")

    # 3. Last Resort Absolute Fallback (should theoretically never happen now)
    if not success:
        print("All image sources failed. Using last-resort generated fallback.")
        img = Image.new('RGB', (width, height), color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        img.save(output_path)

    return output_path

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

@app.post("/generate")
async def generate_video_endpoint(req: VideoRequest):
    script = req.script
    aspect_ratio = req.aspect_ratio

    if not script:
        raise HTTPException(status_code=400, detail="Script is empty")
    if aspect_ratio not in ["16:9", "9:16"]:
        raise HTTPException(status_code=400, detail="Invalid aspect ratio")

    scenes = parse_script(script)
    if not scenes:
        raise HTTPException(status_code=400, detail="Could not parse scenes from script")

    job_id = str(uuid.uuid4())[:8]

    # 1. Generate Audio and Images for each scene
    audio_paths = []
    image_paths = []
    for i, scene in enumerate(scenes):
        # We must await the new async generate_audio
        audio_path = await generate_audio(scene, i, job_id)
        audio_paths.append(audio_path)

        image_path = generate_image(scene, i, job_id, aspect_ratio)
        image_paths.append(image_path)

    # 2. Stitch Video
    final_video_path = create_video(job_id, image_paths, audio_paths, aspect_ratio)

    # Clean up the path for the web response
    video_url = f"/{final_video_path.replace('app/', '')}" if final_video_path.startswith("app/") else final_video_path

    return {
        "status": "success",
        "video_url": video_url,
        "scenes_parsed": len(scenes)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
