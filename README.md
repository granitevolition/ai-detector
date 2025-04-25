# AI Content Detector

This tool analyzes text to determine whether it was written by a human or an AI model. It can process individual sentences, paragraphs, or whole documents.

## Features

- Binary classification (Human vs AI) for text content
- Sentence-level analysis for longer documents
- Standardization of input text
- Support for asynchronous processing of multiple sentences
- Detailed debug mode
- Interactive CLI mode
- Configurable model parameters

## Requirements

- Python 3.7+
- Predibase API access
- Required Python packages (install via pip):
  - predibase
  - asyncio
  - nltk (optional, for better sentence tokenization)

## Configuration

The `config.py` file contains important settings for the API and model:

- API_TOKEN: Your Predibase API token
- MODEL_NAME: Name of the fine-tuned model
- TEMPERATURE: Inference temperature (default: 0.0 for deterministic outputs)

## Usage

### Basic Usage

```bash
# Analyze a single piece of text
python inference.py --text "This is a sample text to analyze."

# Analyze text from a file
python inference.py --file path/to/document.txt

# Process as a document with sentence-level analysis
python inference.py --file path/to/document.txt --document

# Interactive mode
python inference.py
```

### Advanced Options

```bash
# Specify minimum words for sentence analysis
python inference.py --file document.txt --document --min-words 10

# Use a specific adapter
python inference.py --adapter custom_model/2

# Save results to a file
python inference.py --file document.txt --document --output results.json

# Disable debug mode
python inference.py --text "Sample text" --no-debug

# Set max concurrent requests for async processing
python inference.py --file large_document.txt --document --max-concurrent 20
```

## Output

For single texts, the output includes:
- Binary prediction (0 for human, 1 for AI)
- Full model response

For documents, the output includes:
- Sentence-by-sentence analysis
- Overall document classification
- Statistics on human vs AI sentences

## License

MIT
