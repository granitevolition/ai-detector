# AI Content Detector API

This API analyzes text to determine whether it was written by a human or an AI model. It provides simple binary classification (0 for human, 1 for AI) for text inputs.

## Features

- REST API for text classification
- Binary prediction (0 = human, 1 = AI)
- Sentence-level analysis for longer documents
- Standardization of input text
- Detailed results for document analysis

## Requirements

- Python 3.7+
- Predibase API access
- Required Python packages (install via pip):
  - predibase
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - tqdm
  - fastapi
  - uvicorn
  - nltk (optional, for better sentence tokenization)

## Configuration

The `config.py` file contains important settings for the API and model:

- API_TOKEN: Your Predibase API token
- MODEL_NAME: Name of the fine-tuned model
- TEMPERATURE: Inference temperature (default: 0.0 for deterministic outputs)

## API Usage

### Starting the API

```bash
# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### 1. Health Check

```
GET /
```

Returns status information about the API.

#### 2. Single Text Detection

```
POST /detect
```

Analyze a single piece of text and get a binary prediction (0 for human, 1 for AI).

**Request Body:**
```json
{
  "text": "Text to analyze",
  "adapter_id": "optional_adapter_id/version"
}
```

**Response:**
```json
{
  "prediction": 0,  // 0 = human, 1 = AI
  "text": "Text to analyze"
}
```

#### 3. Document Analysis

```
POST /detect-document
```

Analyze a document by breaking it into sentences and detecting each one.

**Request Body:**
```json
{
  "text": "This is a longer document with multiple sentences. Each sentence will be analyzed separately.",
  "adapter_id": "optional_adapter_id/version",
  "min_words": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "sentence": "This is a longer document with multiple sentences.",
      "prediction": 0,
      "index": 0
    },
    {
      "sentence": "Each sentence will be analyzed separately.",
      "prediction": 1,
      "index": 1
    }
  ],
  "overall_prediction": 1,
  "human_count": 1,
  "ai_count": 1,
  "unknown_count": 0,
  "total_sentences": 2,
  "analyzed_sentences": 2
}
```

## Deployment

This project includes a Procfile for easy deployment to platforms like Heroku, Railway, etc.

```
web: uvicorn api:app --host 0.0.0.0 --port $PORT
```

## Command-Line Interface

The original command-line interface is still available through `inference.py`:

```bash
# Analyze a single piece of text
python inference.py --text "This is a sample text to analyze."

# Analyze text from a file
python inference.py --file path/to/document.txt

# Process as a document with sentence-level analysis
python inference.py --file path/to/document.txt --document
```
