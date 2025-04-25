"""
Simple Flask web server for the AI Content Detector.
Use this for web deployments, or use inference.py directly for CLI.
"""

import os
from flask import Flask, request, jsonify, render_template_string
import asyncio
from predibase import Predibase
from config import API_TOKEN, MODEL_NAME
from inference import detect_ai_text_async, process_document_async, format_document_results

app = Flask(__name__)

# Initialize Predibase client
pb = Predibase(api_token=API_TOKEN)
deployment_name = "llama-3-1-8b-instruct"
adapter_id = f"{MODEL_NAME}/1"  # Default to version 1

# Create the client
try:
    client = pb.deployments.client(deployment_name)
    print(f"Connected to Predibase using deployment: {deployment_name}")
except Exception as e:
    print(f"Error connecting to Predibase: {e}")
    client = None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Content Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            border-bottom: 1px solid #ccc;
            padding-bottom: 10px;
        }
        textarea {
            width: 100%;
            height: 200px;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-family: inherit;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f9f9f9;
            white-space: pre-wrap;
        }
        .human {
            color: green;
            font-weight: bold;
        }
        .ai {
            color: red;
            font-weight: bold;
        }
        .mixed {
            color: orange;
            font-weight: bold;
        }
        .loading {
            display: none;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <h1>AI Content Detector</h1>
    <p>This tool analyzes text to determine if it was written by a human or an AI model.</p>
    
    <form id="detector-form" method="POST" action="/analyze">
        <textarea id="text-input" name="text" placeholder="Enter text to analyze...">{{ text }}</textarea>
        <button type="submit">Analyze Text</button>
        <div id="loading" class="loading">Analyzing... This may take a few moments.</div>
    </form>
    
    {% if result %}
    <div class="result">
        {{ result|safe }}
    </div>
    {% endif %}

    <script>
        document.getElementById('detector-form').addEventListener('submit', function() {
            document.getElementById('loading').style.display = 'block';
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '')
    
    if not text.strip():
        return render_template_string(HTML_TEMPLATE, text=text, 
                                     result="Please enter some text to analyze.")
    
    if client is None:
        return render_template_string(HTML_TEMPLATE, text=text,
                                     result="Error: Could not connect to Predibase. Please check your configuration.")
    
    # Determine if it's a document or single text
    if '.' in text and text.count('.') > 1:
        # Process as document
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results, summary = loop.run_until_complete(
            process_document_async(client, text, adapter_id, debug=False, min_words=5)
        )
        result = format_document_results(results, summary)
        
        # Add HTML formatting
        result = result.replace("HUMAN-WRITTEN", "<span class='human'>HUMAN-WRITTEN</span>")
        result = result.replace("AI-GENERATED", "<span class='ai'>AI-GENERATED</span>")
        result = result.replace("HUMAN", "<span class='human'>HUMAN</span>")
        result = result.replace("AI", "<span class='ai'>AI</span>")
        result = result.replace("MIXED", "<span class='mixed'>MIXED</span>")
    else:
        # Process as single text
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        prediction, response = loop.run_until_complete(
            detect_ai_text_async(client, text, adapter_id, debug=False)
        )
        
        if prediction == 0:
            result = "PREDICTION: <span class='human'>HUMAN-WRITTEN (0)</span>\n\nModel response: " + response
        elif prediction == 1:
            result = "PREDICTION: <span class='ai'>AI-GENERATED (1)</span>\n\nModel response: " + response
        else:
            result = "PREDICTION: UNKNOWN\n\nModel response: " + response
    
    return render_template_string(HTML_TEMPLATE, text=text, result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
