# AI Artisan Ad Generator

The AI Artisan Ad Generator is a prototype application that transforms product photos into short video advertisements.  
By combining AI-based frame generation with motion, captions, and background music, the tool produces ready-to-use ads suitable for digital platforms.

---

## Features

- Upload a product image and provide a prompt and caption  
- Generate 8–12 second ads with motion and stylization  
- Automatic caption overlay on each frame  
- Background music integration with trimming or looping  
- Outputs a final `.mp4` video ready for sharing  
- FastAPI backend for easy API access  

---

## Technology Stack

- FastAPI – REST API framework  
- PyTorch, Diffusers, ControlNet – AI model for frame generation  
- MoviePy – Frame stitching, video creation, audio integration  
- FFmpeg – Video/audio encoding backend  
- Python 3.11  

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-artisan-ad-generator.git
   cd ai-artisan-ad-generator

2. python -m venv .venv
   source .venv/bin/activate   # On Linux/Mac
   .venv\Scripts\activate      # On Windows

3. pip install -r requirements.txt

4. Ensure FFmpeg is installed and added to your PATH.
   Download FFmpeg from: https://ffmpeg.org/download.html

5. uvicorn app:app --reload --port 8000

6. curl -X POST "http://127.0.0.1:8000/generate_ad" \
  -F "prompt=Create a stylish ad for a premium watch" \
  -F "caption=Timeless Elegance" \
  -F "file=@watch.png" \
  --output ad.mp4
