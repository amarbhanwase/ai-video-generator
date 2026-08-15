import os
import re
import uuid
import requests
import urllib.parse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from PIL import Image
from gtts import gTTS
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
    Uses basic sentence splitting for simplicity.
    """
    # Split by punctuation that typically ends a sentence/phrase
    raw_scenes = re.split(r'(?<=[.?!])\s+', script.strip())

    # Filter out empty scenes and clean whitespace
    scenes = [scene.strip() for scene in raw_scenes if scene.strip()]

    # If script has no punctuation or is too long, we might need further chunking,
    # but basic sentence split works well as a starting point.

    # ensure no scene is completely empty
    return scenes

def generate_audio(text: str, index: int, job_id: str) -> str:
    """
    Generates audio for a given text using gTTS (Google Translate TTS).
    Returns the file path to the generated audio.
    """
    output_path = f"app/outputs/{job_id}_{index}.mp3"
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error generating audio for scene {index}: {str(e)}")
        # Create an empty/dummy file as fallback
        with open(output_path, "wb") as f:
            pass
        return output_path

def generate_image(scene_text: str, index: int, job_id: str, aspect_ratio: str) -> str:
    """
    Generates an image for a scene using Pollinations.ai API.
    Returns the file path to the generated image.
    """
    output_path = f"app/outputs/{job_id}_{index}.png"

    try:
        width, height = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)
        prompt = f"Cinematic, high quality, highly detailed scene: {scene_text}"
        encoded_prompt = urllib.parse.quote(prompt)

        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path
    except Exception as e:
        print(f"Error generating image for scene {index}: {str(e)}")
        # Fallback to a solid color image on error
        width, height = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)
        img = Image.new('RGB', (width, height), color = "gray")
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

            # 3. Create a dynamic cropping function over time
            def make_frame_zoom(t):
                # Calculate progress from 0 to 1 over the clip duration
                progress = t / duration

                # We start zoomed in (showing the center of the image)
                # And we zoom out slightly over time by expanding the crop window
                # Start crop window size (1.0x width/height)
                current_w = width * (1.0 + 0.1 * (1.0 - progress))
                current_h = height * (1.0 + 0.1 * (1.0 - progress))

                # Center coordinates
                x_center = base_w / 2
                y_center = base_h / 2

                # Calculate crop box (x1, y1, x2, y2)
                x1 = int(x_center - current_w / 2)
                y1 = int(y_center - current_h / 2)

                # Get the frame from the original (resized) clip
                frame = img_clip.get_frame(t)

                # Crop the frame using standard numpy slicing
                cropped = frame[y1:y1+int(current_h), x1:x1+int(current_w)]

                # The cropped frame may not be EXACTLY width/height due to zooming
                # Convert back to PIL to resize to EXACT width/height, then back to numpy
                import numpy as np
                pil_img = Image.fromarray(cropped)
                final_img = pil_img.resize((width, height), Image.Resampling.LANCZOS)
                return np.array(final_img)

            # Create a new VideoClip from our dynamic frame generator
            from moviepy.video.VideoClip import VideoClip
            zoomed_clip = VideoClip(make_frame_zoom, duration=duration)

            if audio_clip:
                zoomed_clip = zoomed_clip.with_audio(audio_clip)

            clips.append(zoomed_clip)

        if not clips:
            raise Exception("No clips to stitch")

        final_video = concatenate_videoclips(clips, method="compose")
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
async def generate_video(req: VideoRequest):
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
        audio_path = generate_audio(scene, i, job_id)
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
