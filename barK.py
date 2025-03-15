import soundfile as sf
from transformers import AutoProcessor, BarkModel

# Load the processor and model
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

# Set the voice preset
voice_preset = "v2/en_speaker_6"

# Generate inputs
inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

# Generate audio
audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

# Save the audio as a WAV file
sf.write("output_audio.wav", audio_array, 22050)  # 22050 Hz is typical for this model

print("Audio saved as 'output_audio.wav'")
