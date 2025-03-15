from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import io
import torch
import soundfile as sf
import torchaudio
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoProcessor,
    BarkModel,
    MusicgenForConditionalGeneration,
    T5Config,
    EncodecConfig,
    MusicgenDecoderConfig
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ----- GPT-2 for Lyrics Generation -----
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # Set the pad token to the EOS token
gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2")

gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2")

def generate_text(prompt, max_new_tokens=150, temperature=0.7):
    # Tokenize the prompt with padding to ensure an attention mask is created
    inputs = gpt2_tokenizer(prompt, return_tensors="pt", padding=True)
    prompt_length = inputs.input_ids.shape[1]

    # Generate tokens with updated parameters including attention_mask and pad_token_id
    output_tokens = gpt2_model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=gpt2_tokenizer.eos_token_id,  # explicitly set the pad token id
        max_new_tokens=max_new_tokens,
        min_length=prompt_length + 20,  # ensures at least 20 new tokens are generated
        temperature=temperature,
        do_sample=True,
        no_repeat_ngram_size=2,         # avoid repeating 2-word sequences
        repetition_penalty=1.3,         # discourage repetition
        top_p=0.95,                     # nucleus sampling for more diverse output
        top_k=50                        # limit to top 50 candidates for each token
    )

    # Remove the prompt tokens from the generated output
    generated_tokens = output_tokens[0][prompt_length:]
    new_text = gpt2_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return new_text

    # Remove the prompt tokens from the generated output
    generated_tokens = output_tokens[0][prompt_length:]
    new_text = gpt2_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return new_text


# ----- Bark TTS for Vocal Synthesis -----
tts_processor = AutoProcessor.from_pretrained("suno/bark")
tts_model = BarkModel.from_pretrained("suno/bark")

# ----- MusicGen for Instrumental Generation -----
music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
music_model = music_model.to(device)

# ----- Routes -----
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


import asyncio

@app.post("/generate_song/")
async def generate_song(
    request: Request,
    description: str = Form(...),
    lyrics_genre: str = Form(...),
    music_genre: str = Form(...),
    duration: int = Form(...),
):
    try:
        # --- Step 1: Generate Lyrics ---
        lyrics_prompt = f"Generate song lyrics in {lyrics_genre} style about {description}.REMEMBER "
        lyrics = generate_text(lyrics_prompt, max_new_tokens=150, temperature=0.7)
        print("Generated Lyrics:\n", lyrics)
        
        # Wait a couple of seconds before proceeding so you can confirm the printed lyrics
        await asyncio.sleep(2)
        
        # --- Step 2: Generate Vocals using Bark TTS ---
        tts_inputs = tts_processor(text=lyrics, return_tensors="pt").to(tts_model.device)
        speech_audio = tts_model.generate(**tts_inputs)
        buffer_vocals = io.BytesIO()
        sf.write(buffer_vocals, speech_audio[0].cpu().numpy(), samplerate=24000, format="WAV")
        buffer_vocals.seek(0)
        audio_vocals, sr_vocals = torchaudio.load(buffer_vocals)
        target_sr = 32000
        if sr_vocals != target_sr:
            audio_vocals = torchaudio.functional.resample(audio_vocals, sr_vocals, target_sr)
            sr_vocals = target_sr

        # --- Step 3: Generate Instrumental Music using MusicGen ---
        music_prompt_text = f"{music_genre} instrumental music"
        music_inputs = music_processor(text=[music_prompt_text], return_tensors="pt").to(device)
        max_tokens = duration * 50  # Adjust max tokens based on duration
        audio_instrument = music_model.generate(**music_inputs, max_new_tokens=max_tokens)

        # Convert to numpy and save to buffer
        audio_instrument = audio_instrument.cpu().numpy().squeeze()
        buffer_music = io.BytesIO()
        sf.write(buffer_music, audio_instrument, samplerate=target_sr, format="WAV")
        buffer_music.seek(0)
        audio_music, sr_music = torchaudio.load(buffer_music)
        
        # Check sample rate of instrumental music
        if sr_music != target_sr:
            raise ValueError("Instrumental sample rate does not match target sample rate.")

        # --- Step 4: Blend Vocals with Instrumental ---
        min_length = min(audio_vocals.shape[1], audio_music.shape[1])
        audio_vocals = audio_vocals[:, :min_length]
        audio_music = audio_music[:, :min_length]
        combined_audio = audio_vocals * 0.6 + audio_music * 0.4

        # --- Step 5: Save the Final Song ---
        song_path = "static/generated_song.wav"
        sf.write(song_path, combined_audio.T, target_sr)

        return JSONResponse(content={"song_url": f"/{song_path}", "lyrics": lyrics})
    
    except Exception as e:
        print(f"Error generating song: {e}")
        return JSONResponse(status_code=500, content={"error": "Error generating song. Please try again."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
