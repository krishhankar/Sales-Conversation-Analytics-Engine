# Sales Conversation Analytics Engine

This project processes sales conversations from audio recordings or text transcripts. It transcribes audio, identifies the language, scores the conversation, extracts CRM details, and stores the results for search and analytics.

## What it does

- Transcribes audio and detects its language.
- Scores calls for topics such as pricing, objection handling, discovery questions, and next steps.
- Extracts CRM details, including objections, follow-up dates, key topics, and predicted outcomes.
- Searches stored transcripts with natural-language queries.
- Summarizes call scores, outcomes, and languages in an analytics dashboard.
- Includes scripts to seed demo calls and evaluate outcome predictions.

## Tools used

- FastAPI and Uvicorn for the API.
- LangGraph for the processing pipeline.
- Groq models for call scoring and CRM extraction. The configured models are `openai/gpt-oss-20b` and `qwen/qwen3.8-27b` as a fallback.
- faster-whisper for audio transcription. The configured model is `base`, approximately 140 MB.
- langdetect for language detection.
- ChromaDB and `all-MiniLM-L6-v2` embeddings for transcript search.
- PostgreSQL 16 for call records and Redis 7 for caching.
- Hugging Face Datasets for the sample conversation dataset.
- gTTS for generating example audio.
- Docker Compose for running the API and data services.

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
|                              FastAPI Container                               |
|                                                                              |
|   ┌─────────────────┐   ┌───────────────────────────────────────────────┐    |
|   |      Routes     |-->|               LangGraph Pipeline              |    |
|   |                 |   |                                               |    |
|   | POST /calls     |   | ingest --> transcribe and detect_lang         |    |
|   | GET  /calls     |   |     |                    |                    |    |
|   | POST /search    |   |     v                    v                    |    |
|   | GET /analytics  |   | score_call --> extract_crm --> embed          |    |
|   └─────────────────┘   └───────────────────────────────────────────────┘    |
|                                                                              |
|          |                    |                   |                |         |
|          v                    v                   v                v         |
|   ┌─────────────┐   ┌───────────────────┐  ┌─────────────┐  ┌─────────────┐  |
|   |    Redis    |   |   faster-whisper  |  |     Groq    |  |   ChromaDB  |  |
|   └─────────────┘   └───────────────────┘  └─────────────┘  └─────────────┘  |
└──────────────────────────────────────────────────────────────────────────────┘
           |                                                         |
           v                                                         v
  ┌────────────────┐                                       ┌─────────────────┐
  |   PostgreSQL   |                                       |   HuggingFace   |
  └────────────────┘                                       |    Embeddings   |
                                                           └─────────────────┘
```

## Getting started

You will need Docker with Docker Compose and a Groq API key.

Copy the environment template:

```bash
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`, then start the services:

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`. Check that it is running:

```bash
curl http://localhost:8000/health
```

Open [http://localhost:8000/](http://localhost:8000/) to view the dashboard. It shows call totals, average scores, outcome and language distributions, and includes transcript search.

## API endpoints

### Process a text transcript

```bash
curl -X POST http://localhost:8000/calls \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Sales Rep: Hello, thanks for taking my call. Customer: We are looking for an analytics platform, but pricing is a concern. Sales Rep: I understand. We offer a 14-day trial. Would you like to schedule a demo next Tuesday? Customer: Yes, that works."
  }'
```

### Process an audio recording

Generate the example recordings if needed:

```bash
python scripts/generate_tts_samples.py
```

Upload an audio file:

```bash
curl -X POST http://localhost:8000/calls/audio \
  -F "audio=@data/sample_audio/en_sales_sample.mp3"
```

Audio uploads are limited to 25 MiB. Empty uploads are rejected.

### Retrieve a call

```bash
curl http://localhost:8000/calls/CALL_ID
```

### Search conversations

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "calls with pricing objections and confirmed next steps", "n_results": 5}'
```

### View analytics

```bash
curl http://localhost:8000/analytics/summary
```

## Audio samples

Generate sample audio in English, Hindi, and Tamil:

```bash
python scripts/generate_tts_samples.py
```

To download audio clips from a source that provides direct download URLs, edit the `AUDIO_URLS` variable in [scripts/download_sample_audio.py](scripts/download_sample_audio.py) with the clip filenames and URLs, then run:

```bash
python scripts/download_sample_audio.py
```

The script saves the files under `data/sample_audio/`.

## Seed demo calls

The seeder reads the requested number of rows from the Hugging Face dataset and processes them through the pipeline:

```bash
python scripts/seed_dataset.py --count 3 --delay 3
```

PostgreSQL and Redis must be available to the application while seeding.

## Evaluate outcome predictions

Run the evaluation against a streamed sample of labeled conversations:

```bash
python scripts/evaluate.py --sample-size 50 --delay 3
```

The report includes accuracy, precision, recall, specificity, balanced accuracy, F1, macro-F1, MCC, per-class results, and confusion-matrix counts. Results are saved to `evaluation_results.json`.

## Run tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Project structure

```text
Sales-Conversation-Analytics-Engine/
├── app/
│   ├── dashboard.html
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── graph.py
│   ├── vectorstore.py
│   ├── cache.py
│   ├── db/
│   ├── nodes/
│   └── routes/
├── scripts/
│   ├── generate_tts_samples.py
│   ├── download_sample_audio.py
│   ├── seed_dataset.py
│   └── evaluate.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
