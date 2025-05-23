import requests
import pyaudio
import wave
import whisper
import os
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings
from gtts import gTTS
import sounddevice as sd
from pydub import AudioSegment
import re
import keyboard
from dotenv import load_dotenv

# Suppress FP16 warning from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load environment variables from .env file
load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not API_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in .env file")

# Hugging Face API configuration for Mixtral
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# System prompt for Nira
system_prompt = """
Your name is Nira.
You are the AI assistant for Analytas, an AI advisory firm specializing in the safe and intelligent deployment of autonomous AI agents for organizations. Your role is to assist website visitors in exploring whether AI agents are suitable for their use case, using only information from the Agenta documentation (https://docs.agenta.ai/). Always begin by asking what brought them to Analytas today—whether they’re curious about how AI agents work, wondering if their company is ready, or looking to schedule a discovery call with the team (requesting their full name, email, phone, date, time, and timezone).
Respond only to questions related to AI agents and Analytas' services as described in the Agenta documentation. Keep answers clear, concise, and focused, avoiding speculation or topics outside the documentation. Your tone must be professional, trustworthy, and conversational, reflecting thoughtful expertise.
For topics outside the Agenta documentation or Analytas' services, respond exactly with: “Great question. I don't have a confident answer on that just yet — but we're always learning. If it’s important, I’d recommend scheduling a short discovery call with our team. They’d be glad to help.”
If the user seems ready or unsure, gently offer to schedule a discovery call to explore how Analytas can support their goals. Your goal is to build trust and provide accurate information about AI agents.
"""

# File to store scraped documentation
DOCS_FILE = "agenta_docs.txt"

# Scrape Agenta documentation
def scrape_agenta_docs(url="https://docs.agenta.ai"):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content = []
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']):
            text = tag.get_text(strip=True)
            if text:
                content.append(text)
        with open(DOCS_FILE, 'w', encoding='utf-8') as f:
            for item in content:
                f.write(item + "\n")
        return content
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

# Load cached documentation
def load_cached_docs():
    if os.path.exists(DOCS_FILE):
        print("Loading cached Agenta documentation...")
        with open(DOCS_FILE, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return None

# Create FAISS index
def create_faiss_index(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, model, documents

# Retrieve relevant documents
def retrieve_relevant_docs(query, index, model, documents, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# Correct common transcription errors
def correct_transcription(text):
    corrections = {
        "A.A agents": "AI agents",
        "A A agents": "AI agents",
        "Nera": "Nira",
        "Neera": "Nira",
        "Motor AI Agents": "What are AI agents",
        "soped": "speed",
        "voicfe": "voice",
        "humkan generated voicfe": "human-generated voice"
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

# Generate response using Hugging Face API
def generate_response(user_input, context):
    prompt = f"{system_prompt}\nContext from Agenta documentation: {context}\nUser: {user_input}\nNira: "
    stop_sequences = ["\nUser:", "\n\n", "User:", "\n"]
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": stop_sequences
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            full_output = result[0]["generated_text"]
            response_text = full_output[len(prompt):].strip() if full_output.startswith(prompt) else full_output
            for stop_token in stop_sequences:
                if stop_token in response_text:
                    response_text = response_text.split(stop_token)[0].strip()
            if not response_text or response_text.startswith("[Respond only"):
                return "Great question. I don't have a confident answer on that just yet — but we're always learning. If it's important, I’d recommend scheduling a short discovery call with our team. They’d be glad to help."
            return response_text
        else:
            return "Unexpected response format."
    elif response.status_code == 403:
        return "Access denied. You might need to request access to this model on Hugging Face."
    elif response.status_code == 401:
        return "Unauthorized. Check your Hugging Face API token."
    else:
        return f"Error {response.status_code}: {response.text}"

# Audio recording function with keyboard control
AUDIO_PATH = "audio/input.wav"

def record_audio_with_keyboard(path=AUDIO_PATH):
    print("Press 's' to start recording, 'e' to end recording...")
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = None
    frames = []

    try:
        # Wait for 's' key to start
        keyboard.wait('s')
        print("Recording started...")
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
        
        # Record until 'e' key is pressed
        while not keyboard.is_pressed('e'):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        print("Recording stopped.")
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save audio
        os.makedirs("audio", exist_ok=True)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        print("Audio saved to:", path)
    except Exception as e:
        print(f"Error during recording: {e}")
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

# Transcription function
def transcribe_whisper(path=AUDIO_PATH):
    print("Transcribing...")
    try:
        model = whisper.load_model("small", device="cpu")
    except Exception as e:
        print(f"Failed to load Whisper small model: {e}. Falling back to base model.")
        model = whisper.load_model("base", device="cpu")
    result = model.transcribe(path, language='en')
    print("Transcribed text:", result["text"])
    return result["text"]

# TTS function with faster speed and human-like enhancements
def text_to_speech(text, output_path="audio/output.wav"):
    print("Generating audio response...")
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        combined_audio = AudioSegment.empty()
        
        for sentence in sentences:
            if sentence:
                tts = gTTS(text=sentence, lang='en', tld='co.uk', slow=False)
                temp_mp3 = "audio/temp_output.mp3"
                os.makedirs("audio", exist_ok=True)
                tts.save(temp_mp3)
                
                audio = AudioSegment.from_mp3(temp_mp3)
                audio = audio.speedup(playback_speed=1.2)
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * 1.059 ** 2)
                })
                combined_audio += audio + AudioSegment.silent(duration=200)
                os.remove(temp_mp3)
        
        combined_audio.export(output_path, format="wav")
        print(f"Audio response saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating TTS: {e}")
        return None

# Play audio function
def play_audio(file_path):
    print("Playing audio response...")
    try:
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        sd.play(audio_array, sample_rate)
        sd.wait()
        print("Audio playback complete.")
    except Exception as e:
        print(f"Error playing audio: {e}")

# Main function
if __name__ == "__main__":
    print("Nira Script Version: 2025-05-23 (4 stop sequences, keyboard-only control, .env support)")
    # Load or scrape Agenta documentation
    documents = load_cached_docs()
    if not documents:
        documents = scrape_agenta_docs()
        if not documents:
            print("Failed to scrape Agenta documentation. Exiting.")
            exit(1)
    index, embedding_model, documents = create_faiss_index(documents)

    print("Nira is ready to assist! Press 'y' to Start/continue conversation or 'q' to quit.")
    while True:
        # Wait for keyboard input
        event = keyboard.read_event(suppress=True)
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == 'q':
                print("Goodbye!")
                break
            elif event.name == 'y':
                record_audio_with_keyboard()
                transcribed_text = transcribe_whisper()
                if transcribed_text:
                    transcribed_text = correct_transcription(transcribed_text)
                    context = " ".join(retrieve_relevant_docs(transcribed_text, index, embedding_model, documents))
                    response = generate_response(transcribed_text, context)
                    print("Nira:", response)
                    audio_file = text_to_speech(response)
                    if audio_file:
                        play_audio(audio_file)
                else:
                    print("No text transcribed from audio.")
                print("\nPress 's' to record another question or 'q' to quit.")