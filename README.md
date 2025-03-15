# AI Song Generator

This project is an AI-powered song generator that integrates a FastAPI backend with a Streamlit frontend. The backend uses Uvicorn to serve the FastAPI application, which leverages GPT-2 for lyrics generation, Bark TTS for vocal synthesis, and MusicGen for instrumental creation. The frontend, built with Streamlit, provides an interactive interface for users to input their song preferences and play the generated song.

## Project Structure

- **app.py**  
  The FastAPI backend application. It includes:
  - An endpoint (`/generate_song/`) that receives user inputs (song description, lyrics genre, instrumental genre, and duration).
  - Functions to generate lyrics, synthesize vocals, generate instrumentals, and blend them into a final song.
  - Uses Uvicorn as the ASGI server when run directly.

- **streamlit_app.py**  
  The Streamlit frontend application. It:
  - Provides a user-friendly interface for inputting song parameters.
  - Sends a POST request to the FastAPI backend to generate a song.
  - Displays the generated lyrics and plays the synthesized audio.

- **run_app.bat**  
  A Windows batch script that:
  - Checks for and creates a Python virtual environment.
  - Installs all necessary dependencies.
  - Launches the FastAPI (uvicorn) server and the Streamlit application in separate command windows.

## Setup and Installation

### Prerequisites

- **Python 3.7+**: Ensure Python is installed on your system.
- **Windows OS**: The provided batch script is designed for Windows environments.
- **Git (optional)**: For cloning the repository.

### Steps to Set Up

1. **Clone or Download the Repository**  
   If using Git:
   ````bash
   git clone https://github.com/Aadik1ng/AI-Songs.git
   cd AI-Songs
   ````bash


## Check for an existing virtual environment (env). If not present, it will create one.

**Activate the virtual environment.**
**Upgrade pip.**
**Install all required Python packages: FastAPI, uvicorn, Jinja2, torch, torchaudio, soundfile, transformers, streamlit, and requests.**
**Launch the FastAPI backend (via uvicorn) in a new command window.**
**Launch the Streamlit frontend in another new command window.**
## Accessing the Application

- **Open your web browser** and navigate to [http://localhost:8000](http://localhost:8000) to access the Streamlit interface.
- **Enter your song details**: Use the interface to input your song theme, select genres, and choose a duration.
- **Generate your song**: Click the **Generate Song** button to start the song creation process. The generated lyrics will be displayed, and the song will be auto-played.

## How It Works

### FastAPI Backend (app.py)
- **Lyrics Generation**: Uses GPT-2 to generate song lyrics based on the user-provided description and selected genre.
- **Vocal Synthesis**: Employs Bark TTS to convert the generated lyrics into vocal audio.
- **Instrumental Generation**: Uses MusicGen to create instrumental music according to the chosen genre and duration.
- **Audio Blending**: Merges the vocal and instrumental tracks into a final song and serves it via a static file endpoint.

### Streamlit Frontend (streamlit_app.py)
- Provides an intuitive UI for users to input song details.
- Sends a request to the FastAPI endpoint to generate the song.
- Displays the resulting lyrics and plays the generated audio directly in the browser.

## Troubleshooting

### Dependency Issues
Ensure that you are using a compatible Python version (3.7 or later). If issues persist, try deleting the `env` folder and re-running the batch script.

### Port Conflicts
By default, the FastAPI server runs on port 8000 and the Streamlit app on port 8501. If these ports are already in use, modify the ports in the `run_app.bat` script and update the corresponding URL in `streamlit_app.py`.

### Virtual Environment Problems
If the virtual environment activation fails, confirm that your system’s execution policies allow running scripts, and that Python is added to your system PATH.

## License

This project is open source. Feel free to modify and use it according to your needs.
