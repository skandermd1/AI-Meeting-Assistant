# AI Meeting Assistant

An AI-powered web application that transcribes meeting audio, corrects product and financial terminology, and generates structured meeting minutes with actionable task lists.

## Features

- **Audio Transcription** – Upload meeting audio and transcribe it using OpenAI Whisper
- **Terminology Correction** – Automatically fixes financial/product terms and acronyms (e.g., 401k → 401(k) retirement savings plan, VaR → Value at Risk)
- **Meeting Minutes** – Generates structured meeting minutes with key points and decisions
- **Task List** – Produces actionable tasks with assignees and deadlines
- **Export** – Download the generated meeting minutes and tasks as a text file

## Project Structure

| File | Description |
|------|-------------|
| `speech_analyzer.py` | Main AI Meeting Assistant app (Whisper + IBM Watsonx LLMs) |
| `speech2text_app.py` | Simple audio transcription app |
| `simple_speech2text.py` | Downloads sample audio and transcribes it |
| `simple_llm.py` | Basic IBM Watsonx LLM demo |
| `hello.py` | Simple Gradio demo |
| `sample-meeting.wav` | Sample meeting audio file |

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/) (Hugging Face)
- [Gradio](https://gradio.app/)
- [IBM Watsonx AI](https://www.ibm.com/watsonx) (`ibm-watsonx-ai`, `langchain-ibm`)
- [LangChain](https://langchain.com/)

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MeetingAssistant.git
   cd MeetingAssistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch transformers gradio ibm-watsonx-ai langchain langchain-ibm requests
   ```

4. **IBM Watsonx credentials** – The main app uses IBM Watsonx Cloud. Set up an account and configure your API key/credentials in `speech_analyzer.py` as needed.

## Usage

### AI Meeting Assistant (main app)

```bash
python speech_analyzer.py
```

- Runs on `http://0.0.0.0:5002`
- Upload a meeting audio file, click **Submit**, and receive meeting minutes and a task list with a downloadable file

### Simple Audio Transcription

```bash
python speech2text_app.py
```

- Runs on `http://0.0.0.0:5001`
- Upload audio for plain transcription only

## License

MIT
