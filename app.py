import os
import io
import gc
import time
import uuid
import math
import torch
import random
import shutil
import numpy as np
import random
from moviepy.editor import AudioFileClip, afx
from typing import List, Optional, cast
from moviepy.audio.fx import all as afx

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from PIL import Image, ImageOps, ImageDraw, ImageFont

import cv2

from moviepy.editor import ImageSequenceClip, AudioFileClip

from gtts import gTTS

from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers.models.controlnet import ControlNetModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# ---------------------------
# App setup
# ---------------------------
app = FastAPI(title="AI Artisan Ad Generator", version="1.0.0")

OUTPUT_ROOT = "outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---------------------------
# Torch / GPU setup
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if device == "cuda" else torch.float32

# ---------------------------
# Model initialization
# ---------------------------
# Model IDs
SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_ID = "./models/sd-controlnet-canny"

def _try_enable_xformers(pipe):
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        # xFormers not available; continue without it
        pass

def _init_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_ID, torch_dtype=DTYPE
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD15_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None
    )
    # Use a memory-efficient scheduler & features for 6GB VRAM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing("auto")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    _try_enable_xformers(pipe)
    if device == "cuda":
        pipe.to("cuda")
    return pipe

# Initialize once at startup
PIPE = cast(StableDiffusionControlNetPipeline, _init_pipeline())

# ---------------------------
# Utilities
# ---------------------------

def read_imagefile_to_pil(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

def make_output_dir() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(OUTPUT_ROOT, f"ad_{ts}_{uuid.uuid4().hex[:8]}")
    os.makedirs(folder, exist_ok=True)
    return folder

def ensure_max_size(img: Image.Image, max_side: int = 768) -> Image.Image:
    w, h = img.size
    scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
    if scale > 1.0:
        new_w = int(w / scale)
        new_h = int(h / scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)

def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)

def canny_from_pil(img: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    # Convert to OpenCV BGR
    np_img = np.array(img)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(np_img, low, high)
    # Canny returns single channel; make 3 channel for ControlNet
    edges_3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    pil_edges = Image.fromarray(edges_3)
    return pil_edges

def affine_jitter(
    img: Image.Image,
    translate_px: int = 6,
    rotate_deg: float = 3.0,
    scale_jitter: float = 0.02
) -> Image.Image:
    w, h = img.size
    tx = random.randint(-translate_px, translate_px)
    ty = random.randint(-translate_px, translate_px)
    angle = random.uniform(-rotate_deg, rotate_deg)  # float
    scale = 1.0 + random.uniform(-scale_jitter, scale_jitter)
    new_w, new_h = int(w * scale), int(h * scale)
    img2 = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    img2 = img2.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    paste_x = (w - new_w) // 2 + tx
    paste_y = (h - new_h) // 2 + ty
    canvas.paste(img2, (paste_x, paste_y))
    return canvas


def overlay_caption(img: Image.Image, caption: str) -> Image.Image:
    # Draw semi-transparent panel + text bottom-center using PIL (no ImageMagick dependency)
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    margin = int(0.05 * W)
    pad_y = int(0.02 * H)
    # Choose a font
    font_size = max(24, int(W * 0.045))
    try:
        # Common font bundled in many envs; fallback to default
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    text_w, text_h = draw.textbbox((0, 0), caption, font=font)[2:]
    panel_w = min(W - 2*margin, text_w + margin)
    panel_h = text_h + pad_y*2
    panel_x1 = (W - panel_w) // 2
    panel_y1 = H - panel_h - pad_y
    panel_x2 = panel_x1 + panel_w
    panel_y2 = panel_y1 + panel_h

    # Semi-transparent rounded rectangle
    panel_color = (0, 0, 0, 140)
    draw.rounded_rectangle(
        [panel_x1, panel_y1, panel_x2, panel_y2],
        radius=int(0.02 * W),
        fill=panel_color
    )

    # Text
    text_x = (W - text_w) // 2
    text_y = panel_y1 + (panel_h - text_h) // 2
    draw.text((text_x, text_y), caption, font=font, fill=(255, 255, 255, 255))
    return img

def synth_slight_prompt_variation(prompt: str, strength: float = 0.1) -> str:
    # Append tiny stylistic tags to emulate subtle AnimateDiff-like changes
    tags = [
        "cinematic lighting", "soft shadows", "product hero shot",
        "studio backdrop", "high contrast", "vivid color", "macro detail",
        "bokeh", "clean background", "premium aesthetic"
    ]
    k = max(1, int(len(tags) * strength))
    extra = ", ".join(random.sample(tags, k))
    return f"{prompt}, {extra}"

def sd_generate(
    pipe: StableDiffusionControlNetPipeline,
    prompt: str,
    control_image: Image.Image,
    seed: Optional[int] = None,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 15,
    height: int = 512,
    width: int = 512,
):
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.autocast(device if device == "cuda" else "cpu", dtype=DTYPE) if device == "cuda" else torch.no_grad():
        out = pipe(
            prompt=prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
            controlnet_conditioning_scale=1.0
        )
    # Handle both tuple and PipelineOutput
    images = cast(List[Image.Image], getattr(out, "images", out[0]))
    return images[0]

def generate_frames_with_motion(
    product_img: Image.Image,
    base_prompt: str,
    caption: str,
    n_frames: int = 192,
    out_size: int = 512
) -> List[Image.Image]:
    # Prepare reference/control images sequence (jitter + canny thresholds oscillate)
    base = ensure_max_size(product_img, max_side=out_size)
    base = ImageOps.contain(base, (out_size, out_size), Image.Resampling.LANCZOS)
    # Center on square canvas if not square
    if base.size != (out_size, out_size):
        canvas = Image.new("RGB", (out_size, out_size), (255, 255, 255))
        x = (out_size - base.size[0]) // 2
        y = (out_size - base.size[1]) // 2
        canvas.paste(base, (x, y))
        base = canvas

    frames: List[Image.Image] = []
    for i in range(n_frames):
        # tiny motion via affine jitter
        jitted = affine_jitter(base, translate_px=4, rotate_deg=2.0, scale_jitter=0.015)

        # vary canny thresholds smoothly over time
        t = i / max(1, n_frames - 1)
        low = int(80 + 60 * math.sin(2 * math.pi * t))
        high = low + 100
        control = canny_from_pil(jitted, low=max(50, low), high=min(250, high))

        # subtle prompt variation and guidance wobble
        this_prompt = synth_slight_prompt_variation(base_prompt, strength=0.08)
        g_scale = 6.5 + 1.0 * math.sin(2 * math.pi * t)

        seed = random.randint(0, 10_000_000)
        img = sd_generate(
            PIPE,
            prompt=this_prompt,
            control_image=control,
            seed=seed,
            guidance_scale=g_scale,
            num_inference_steps=14,   # keep it light for 6GB VRAM
            height=out_size,
            width=out_size
        )

        # overlay caption on each frame for consistent readability
        img = overlay_caption(img, caption)
        frames.append(img)

        # Free memory between steps
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return frames

def save_frames(frames: List[Image.Image], folder: str) -> List[str]:
    paths = []
    for idx, im in enumerate(frames):
        p = os.path.join(folder, f"frame_{idx:04d}.png")
        im.save(p, format="PNG")
        paths.append(p)
    return paths

def build_video_with_audio(
    frame_paths: List[str],
    audio_path: Optional[str],
    out_path: str,
    fps: int = 4
):
    # Load frames as arrays for moviepy
    frames_np = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]
    clip = ImageSequenceClip(frames_np, fps=fps)

    if audio_path and os.path.exists(audio_path):
        audio_clip = AudioFileClip(audio_path)
        # Ensure audio matches video duration
        audio_clip = audio_clip.set_duration(clip.duration)
        clip = clip.set_audio(audio_clip)

    # Export mp4
    clip.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        preset="medium",
        threads=2
    )
    clip.close()

def synth_voiceover(text: str, out_mp3: str, lang: str = "en"):
    tts = gTTS(text=text, lang=lang)
    tts.save(out_mp3)

# ---------------------------
# Routes
# ---------------------------

@app.get("/")
def root():
    return JSONResponse({"status": "ok", "name": "AI Artisan Ad Generator"})


@app.post("/generate_ad")
async def generate_ad(
    prompt: str = Form(..., description="Text prompt to style the product"),
    caption: str = Form(..., description="Caption to overlay text on video"),
    file: UploadFile = File(..., description="Product image")
):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Please upload a PNG, JPG, or WEBP image.")

    out_dir = make_output_dir()
    raw_image_path = os.path.join(out_dir, "input_image.png")

    pil_image = read_imagefile_to_pil(file)
    pil_image.save(raw_image_path)

    fps = 12
    n_frames = 192  # ~8s at 24fps

    try:
        # 1) Generate frames
        frames = generate_frames_with_motion(
            product_img=pil_image,
            base_prompt=prompt,
            caption=caption,
            n_frames=n_frames,
            out_size=512
        )

        # 2) Save frames
        frame_paths = save_frames(frames, out_dir)

        # 3) Build silent video first
        video_out = os.path.join(out_dir, "final_ad.mp4")
        frames_np = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]
        clip = ImageSequenceClip(frames_np, fps=fps)

        # 4) Add background music
        music_dir = os.path.join("assets", "music")
        music_files = [f for f in os.listdir(music_dir) if f.endswith((".mp3", ".wav"))]

        if not music_files:
            raise HTTPException(status_code=500, detail="No music files found in assets/music/")

        music_path = os.path.join(music_dir, random.choice(music_files))
        music = AudioFileClip(music_path)

        if music.duration > clip.duration:
            music = music.subclip(0, clip.duration)
        else:
            music = afx.audio_loop(music, duration=clip.duration) # type: ignore

        final = clip.set_audio(music)

        # 5) Export final ad
        final.write_videofile(
            video_out,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            preset="medium",
            threads=2
        )
        final.close()

        return FileResponse(
            path=video_out,
            media_type="video/mp4",
            filename=os.path.basename(video_out)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

