# Sales Conversation Analytics Engine

A FastAPI service that **transcribes**, **scores**, and **indexes** sales conversations using a LangGraph pipeline — modeled on enterprise conversation-intelligence platforms (similar to what Darwix AI builds for BFSI/enterprise sales teams).

> **Every dependency in this project is free-tier or locally hosted.** No paid API keys, no credit card required.

---

## Features

- **Multilingual Transcription** — faster-whisper (local, free) with automatic language detection across 99+ languages
- **Rubric-Based Scoring** — LLM evaluates each call against a sales best-practices rubric (pricing, objections, next steps, discovery questions, etc.) using structured Pydantic output
- **CRM Auto-Fill Extraction** — Automatically extracts customer name, objections, predicted outcome, and key topics into a CRM-ready JSON record
- **RAG Semantic Search** — ChromaDB-powered vector search across all processed conversations ("find calls where the customer objected to price but still converted")
- **Aggregate Analytics** — Dashboard-ready statistics: average scores, outcome distribution, language breakdown
- **Evaluation Pipeline** — Script that benchmarks scoring accuracy against labeled dataset outcomes (accuracy, F1, precision, recall)

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI Container                      │
│                                                               │
│  ┌─────────┐   ┌──────────────────────────────────────────┐  │
│  │  Routes  │──▶│          LangGraph Pipeline              │  │
│  │         │   │                                          │  │
│  │ POST /calls │   │  ingest ──▶ transcribe ──▶ detect_lang  │  │
│  │ GET  /calls │   │     │                    │              │  │
│  │ POST /search│   │     ▼                    ▼              │  │
│  │ GET  /analytics│   │  score_call ──▶ extract_crm ──▶ embed  │  │
│  └─────────┘   └──────────────────────────────────────────┘  │
│       │                │              │            │          │
│       ▼                ▼              ▼            ▼          │
│   ┌───────┐    ┌──────────────┐  ┌────────┐  ┌──────────┐   │
│   │ Redis │    │ faster-whisper│  │  Groq  │  │ ChromaDB │   │
│   │(cache)│    │   (local)    │  │(free)  │  │ (local)  │   │
│   └───────┘    └──────────────┘  └────────┘  └──────────┘   │
└──────────────────────────────────────────────────────────────┘
        │                                           │
        ▼                                           ▼
  ┌──────────┐                              ┌─────────────┐
  │PostgreSQL│                              │ HuggingFace │
  │ (Docker) │                              │ Embeddings  │
  └──────────┘                              │  (local)    │
                                            └─────────────┘
```

## Stack — All Free & Local

| Component | Tool | Free? | Notes |
|---|---|---|---|
| API Framework | FastAPI | ✅ OSS | Async, high-performance |
| Pipeline | LangGraph | ✅ OSS | Stateful graph orchestration |
| LLM | Groq API (`openai/gpt-oss-20b`, with `qwen/qwen3.8-27b` fallback) | ✅ Free tier | no credit card, model availability can vary by account |
| Transcription | faster-whisper (`base` model) | ✅ Local | CPU-compatible, ~140MB |
| Vector Store | ChromaDB (embedded) | ✅ Local | PersistentClient, no server needed |
| Embeddings | `all-MiniLM-L6-v2` | ✅ Local | Via `langchain-huggingface` |
| Database | PostgreSQL 16 | ✅ Docker | Local container |
| Cache | Redis 7 | ✅ Docker | Local container |
| Dataset | DeepMostInnovations/saas-sales-conversations | ✅ HuggingFace | 100K labeled conversations |
| Audio Demo | gTTS (synthetic) | ✅ Free | No API key needed |

### Whisper Model Options

The default model is `base` (~140MB, fast, good for demos). For better multilingual accuracy, change `WHISPER_MODEL_SIZE` in your `.env` file:

| Model | Size | Speed | Best For |
|---|---|---|---|
| `tiny` | ~75MB | Fastest | Quick testing |
| `base` | ~140MB | Fast | **Default — good balance** |
| `small` | ~460MB | Moderate | Better multilingual accuracy |
| `medium` | ~1.5GB | Slower | High accuracy |
| `large-v3-turbo` | ~3GB | Slowest | Best accuracy, needs GPU |

## Prerequisites

- **Docker** and **Docker Compose** (to run PostgreSQL + Redis + the API)
- **Groq API key** (free, no credit card): [console.groq.com](https://console.groq.com)

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/your-username/Sales-Conversation-Analytics-Engine.git
cd Sales-Conversation-Analytics-Engine

# Copy the environment template and add your Groq API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_your_key_here
```

### 2. Launch everything

```bash
docker compose up --build
```

This starts:
- FastAPI API on `http://localhost:8000`
- PostgreSQL on port `5432`
- Redis on port `6379`

### 3. Verify

```bash
curl http://localhost:8000/health
# {"status":"healthy","service":"sales-conversation-analytics-engine"}
```

Open the analytics dashboard at [http://localhost:8000/](http://localhost:8000/). It shows call totals, average score, outcome and language distributions, and provides semantic transcript search.

## API Endpoints

### POST /calls — Ingest a call

**Text transcript:**
```bash
curl -X POST http://localhost:8000/calls \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Sales Rep: Hi, I am calling about our enterprise plan. Customer: What is the pricing? Sales Rep: Our professional tier starts at $299/month. Customer: That seems high. Sales Rep: I understand the concern. We offer a 14-day free trial and flexible payment terms. Would you like to schedule a demo next week? Customer: Yes, lets do Tuesday. Sales Rep: Perfect, I will send the invite."
  }'
```

**Audio file:**
```bash
# First, generate sample audio (optional)
python scripts/generate_tts_samples.py

# Upload audio
curl -X POST http://localhost:8000/calls/audio \
  -F "audio=@data/sample_audio/en_sales_sample.mp3"
```
Audio uploads use `POST /calls/audio` and are limited to 25 MiB. Empty uploads are rejected.

**Response:**
```json
{
  "call_id": "a1b2c3d4-...",
  "language": "en",
  "language_confidence": 0.95,
  "rubric_scores": {
    "pricing_discussed": true,
    "objection_handled": true,
    "next_steps_confirmed": true,
    "discovery_questions_asked": false,
    "product_demo_offered": true,
    "urgency_created": false,
    "competitor_mentioned": false,
    "overall_score": 0.75
  },
  "crm_record": {
    "customer_name_mentioned": null,
    "company_mentioned": null,
    "pricing_discussed": true,
    "objections_raised": ["Price seems high"],
    "next_steps_confirmed": true,
    "follow_up_date_mentioned": "Next Tuesday",
    "predicted_outcome": "conversion",
    "key_topics": ["enterprise plan", "pricing", "free trial", "demo"]
  },
  "transcript": "..."
}
```

### GET /calls/{id} — Full call details

```bash
curl http://localhost:8000/calls/a1b2c3d4-...
```

### POST /search — Semantic search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "calls where customer objected to price but still converted", "n_results": 5}'
```

### GET /analytics/summary — Aggregate stats

```bash
curl http://localhost:8000/analytics/summary
```

**Response:**
```json
{
  "total_calls": 50,
  "avg_overall_score": 0.72,
  "outcome_distribution": {"conversion": 30, "no_conversion": 20},
  "avg_rubric_scores": {
    "pricing_discussed": 0.82,
    "objection_handled": 0.64,
    "next_steps_confirmed": 0.70,
    "overall_score": 0.72
  },
  "language_distribution": {"en": 48, "hi": 2}
}
```

## Multilingual Demo

Generate synthetic audio in English, Hindi, and Tamil:

```bash
python scripts/generate_tts_samples.py
```

Then upload each file:
```bash
curl -X POST -F "audio=@data/sample_audio/en_sales_sample.mp3" http://localhost:8000/calls/audio
curl -X POST -F "audio=@data/sample_audio/hi_sales_sample.mp3" http://localhost:8000/calls/audio
curl -X POST -F "audio=@data/sample_audio/ta_sales_sample.mp3" http://localhost:8000/calls/audio
```

> **Note:** For real multilingual audio, you can download clips from the [Mozilla Data Collective](https://datacollective.mozillafoundation.org/) (CC0, free). Run `python scripts/download_sample_audio.py` for instructions.

## Evaluation

Run the scoring pipeline against labeled dataset outcomes:

```bash
# Evaluate a shuffled stream of 50 samples without downloading the full dataset.
# Reports per-class precision/recall/F1, specificity, balanced accuracy, and MCC.
python scripts/evaluate.py --sample-size 50 --delay 3

# Results are saved to evaluation_results.json
```

Output:
```
============================================================
EVALUATION RESULTS
============================================================
  Samples evaluated: 50
  Accuracy:          72.0%
  Precision:         75.0%
  Recall:            68.0%
  Specificity:       76.0%
  Balanced accuracy: 72.0%
  F1 Score:          71.3%
  Macro F1:          72.0%
  MCC:               0.440
============================================================
```

> **Note:** Actual accuracy will vary depending on the LLM model and dataset sample. Results above are illustrative.

## Seed Demo Data

Pre-populate the database for demo purposes:

```bash
python scripts/seed_dataset.py --count 20 --delay 3
```
The seeder streams only the requested number of rows instead of downloading the complete dataset first.

## Development

### Run tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

### Run locally (without Docker)

```bash
# Start PostgreSQL and Redis (via Docker)
docker compose up postgres redis -d

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
```

## Project Structure

```
Sales-Conversation-Analytics-Engine/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point + lifespan
│   ├── dashboard.html       # Lightweight analytics and semantic search UI
│   ├── config.py             # Pydantic Settings (env vars)
│   ├── schemas.py            # Pydantic models (rubric, CRM, API contracts)
│   ├── graph.py              # LangGraph StateGraph pipeline definition
│   ├── vectorstore.py        # ChromaDB (embedded) wrapper
│   ├── cache.py              # Redis async cache
│   ├── db/
│   │   ├── __init__.py
│   │   ├── engine.py         # Async SQLAlchemy engine
│   │   ├── models.py         # ORM models (CallRecord)
│   │   └── crud.py           # DB operations + analytics aggregation
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── ingest.py         # Input routing (audio vs text)
│   │   ├── transcribe.py     # faster-whisper transcription
│   │   ├── detect_language.py # Language detection
│   │   ├── score_call.py     # Rubric scoring (Groq LLM)
│   │   ├── extract_crm_fields.py  # CRM extraction (Groq LLM)
│   │   └── embed.py          # ChromaDB + PostgreSQL + Redis storage
│   └── routes/
│       ├── __init__.py
│       ├── calls.py          # POST /calls, GET /calls/{id}
│       ├── search.py         # POST /search
│       └── analytics.py      # GET /analytics/summary
├── scripts/
│   ├── generate_tts_samples.py   # gTTS audio generation (EN/HI/TA)
│   ├── download_sample_audio.py  # Mozilla Common Voice instructions
│   ├── seed_dataset.py           # Pre-populate DB from HF dataset
│   └── evaluate.py               # Accuracy evaluation vs labeled data
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Fixtures + test client
│   ├── test_schemas.py       # Pydantic model tests
│   ├── test_scoring.py       # Pipeline node tests (mocked LLM)
│   └── test_api.py           # API endpoint integration tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Cost Confirmation

✅ **Every component in this project is free:**

- **Groq API**: Free tier, no credit card required ([console.groq.com](https://console.groq.com))
- **faster-whisper**: Open-source, runs locally, no API
- **ChromaDB**: Open-source, embedded local storage
- **HuggingFace Embeddings**: Open-source model, runs locally
- **PostgreSQL**: Official Docker image, local container
- **Redis**: Official Docker image, local container
- **gTTS**: Free Google TTS wrapper, no API key
- **DeepMostInnovations dataset**: Public HuggingFace dataset, free download
- **Mozilla Common Voice**: CC0 licensed, free (download from [MDC](https://datacollective.mozillafoundation.org/))

## License

MIT
