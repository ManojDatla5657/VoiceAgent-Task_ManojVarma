# Nira: AI Voice Assistant for Analytas

**Nira** is a voice-powered AI assistant designed for **Analytas**, an AI advisory firm specializing in autonomous AI agents. It helps users explore AI agent use cases by pulling information from the Agenta documentation.

Nira records questions via microphone, processes them using the Hugging Face API (Mixtral-8x7B), transcribes with Whisper, and responds with gTTS audio. Keyboard controls (`y` to start, `s` to record, `e` to stop, `q` to quit) make it user-friendly. Built in Python, it uses a `.env` file for secure API key storage.

---

## ✨ Features

* **Voice Interaction**: Record questions (`s` to start, `e` to stop) and hear human-like responses.
* **Semantic Search**: Uses FAISS and SentenceTransformer to find relevant Agenta documentation.
* **Speech Processing**: Converts speech to text with Whisper and text to speech with gTTS (1.2x speed, pitch-adjusted).
* **Error Handling**: Fixes common transcription errors and suggests discovery calls for out-of-scope questions.

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/nira-assistant.git
cd nira-assistant
```

### 2. Install Python Dependencies

```bash
pip install requests pyaudio whisper beautifulsoup4 sentence-transformers faiss-cpu gtts sounddevice pydub keyboard python-dotenv
```

### 3. Set Up `.env` File

Create a `.env` file in the project root:

```
HUGGINGFACE_API_TOKEN=your_huggingface_api_token_here
```


## 🧠 Usage

### Run Nira

```bash
python nira_assistant.py
```


### Interact with Nira

* Press `y` to start or continue
* Press `s` to begin recording, `e` to stop
* Ask a question (e.g., "What are AI agents?")
* Nira responds with audio based on Agenta documentation
* Press `q` to exit

---

## 🗂 Project Structure

```
nira-assistant/
├── nira_assistant.py       # Main Nira script
├── .env                    # API key file (not committed)
├── agenta_docs.txt         # Cached Agenta documentation
├── audio/                  # Audio files (input.wav, output.wav, temp_output.mp3)
├── .gitignore              
└── README.md             
```

---

## 📆 Dependencies

* `requests`: HTTP requests for API and scraping
* `pyaudio`: Audio recording
* `whisper`: Speech-to-text transcription
* `beautifulsoup4`: Scrapes Agenta docs
* `sentence-transformers`: Embeds docs for search
* `faiss-cpu`: Semantic search
* `gtts`: Text-to-speech
* `sounddevice`: Audio playback
* `pydub`: Audio enhancements
* `keyboard`: Keyboard input detection
* `python-dotenv`: Loads `.env` file

---
