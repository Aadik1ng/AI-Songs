import streamlit as st
import requests

# FastAPI base URL and endpoint
API_BASE_URL = "http://localhost:8000"
GENERATE_SONG_ENDPOINT = f"{API_BASE_URL}/generate_song/"

st.title("AI Song Generator 🎵")

# --- Input Fields ---
description = st.text_input("Describe your song theme (e.g., heartbreak, adventure, happiness)")
lyrics_genre = st.selectbox("Choose Lyrics Genre", ["Sad Love Song", "Happy Pop Song", "Rap", "Rock", "Country"])
music_genre = st.selectbox("Choose Instrumental Genre", ["Ambient", "Rock", "Jazz", "Classical", "Electronic"])
duration = st.slider("Music Duration (seconds)", min_value=10, max_value=60, value=30, step=5)

if st.button("Generate Song"):
    if description:
        with st.spinner("Generating your song... 🎶"):
            payload = {
                "description": description,
                "lyrics_genre": lyrics_genre,
                "music_genre": music_genre,
                "duration": duration
            }
            response = requests.post(GENERATE_SONG_ENDPOINT, data=payload)
            if response.status_code == 200:
                data = response.json()
                st.success("Song Generated Successfully! 🎧")
                st.write("**Lyrics:**")
                st.text_area("Generated Lyrics", value=data["lyrics"], height=200)
                
                # --- Auto-play the generated song ---
                # Build full URL to access the generated audio file
                song_url = API_BASE_URL + data["song_url"]
                st.audio(song_url, format="audio/wav")
            else:
                st.error("Error generating song. Please try again.")
    else:
        st.warning("Please enter a song theme.")
