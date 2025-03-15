from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline, AutoProcessor, BarkModel
import torch
import soundfile as sf
import numpy as np
import torchaudio
from pydub import AudioSegment
import os

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates for rendering HTML pages
templates = Jinja2Templates(directory="templates")

# Load open-source LLM for text generation (lyrics)
lyrics_model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

# Load Bark model and processor for text-to-speech (vocals)
tts_processor = AutoProcessor.from_pretrained("suno/bark")
tts_model = BarkModel.from_pretrained("suno/bark")

# Load MusicGen model and processor for instrumental music generation
music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")  # Use "musicgen-medium" for better quality
music_model = torch.hub.load("facebook/musicgen-small")

app = FastAPI()

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate_song/")
async def generate_song(
    request: Request,
    description: str = Form(...),  # User provides song theme/idea
    lyrics_genre: str = Form(...),   # Example: "sad love song", "happy pop song", etc.
    music_genre: str = Form(...)
):
    try:
        # Step 1: Generate Lyrics from LLM
        lyrics_prompt = f"Generate poetic song lyrics in {lyrics_genre} style about {lyrics_theme}."
        lyrics_output = lyrics_model(lyrics_prompt, max_length=150, temperature=0.7, do_sample=True)
        lyrics = lyrics_output[0]["generated_text"]
        print("Generated Lyrics:\n", lyrics)

        # Generate speech from lyrics using Bark AI
        speech_inputs = tts_processor(text=lyrics, return_tensors="pt").to(tts_model.device)
        speech_audio = tts_model.generate(**tts_processor(lyrics, return_tensors="pt"))
        
        # Convert to waveform for merging
        audio_vocals, sample_rate_vocals = torchaudio.load(io.BytesIO(sf.write(None, speech_audio[0].numpy(), samplerate=24000, format="wav")))
        audio_vocals = torchaudio.functional.resample(audio_vocals, orig_freq=24000, new_freq=32000)

        # Generate instrumental music
        print(f"Generating {music_genre} background music...")
        music_prompt = f"{music_genre} instrumental music"
        music_inputs = musicgen_processor(text=[lyrics_output[0]["generated_text"]], return_tensors="pt").to(musicgen_model.device)
        music_audio = music_model.generate(inputs["input_ids"], max_new_tokens=500)
        audio_music, sample_rate_music = torchaudio.load(io.BytesIO(sf.write(None, music_audio[0].numpy(), samplerate=32000, format="wav")))

        # Ensure both audio clips have the same length and sample rate
        if sample_rate_vocals != sample_rate_music:
            raise ValueError("Sample rates of vocals and background music do not match!")

        # Adjust the length of the vocal track to match the instrumental
        min_length = min(audio_vocals.shape[1], audio_music.shape[1])
        audio_vocals = audio_vocals[:, :min_length]
        audio_music = audio_music[:, :min_length]

        # Combine vocals with background music
        combined_audio = audio_vocals * 0.6 + audio_music * 0.4  # Blend audio together

        # Save the final song as a WAV file
        song_path = "static/generated_song.wav"
        sf.write(song_path, combined_audio.T, sample_rate_music)

        return JSONResponse(content={"song_url": f"/{song_path}", "lyrics": lyrics_output[0]['generated_text']})
    except Exception as e:
        print(f"Error generating song: {e}")
        return JSONResponse(status_code=500, content={"error": "Error generating song. Please try again."})


# Mount static folder for serving generated files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
