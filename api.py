"""
AI Content Detector API

Provides a simple REST API for detecting whether text is human-written or AI-generated.
Returns 0 for human-written text and 1 for AI-generated text.
"""

import re
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from predibase import Predibase
from config import API_TOKEN, MODEL_NAME, TEMPERATURE

# Set up FastAPI app
app = FastAPI(
    title="AI Content Detector API",
    description="Detects whether text was written by a human (0) or an AI model (1)",
    version="1.0.0"
)

# Initialize Predibase client
pb = None
lorax_client = None
MAX_NEW_TOKENS = 1

def initialize_clients():
    """Initialize the Predibase client and model."""
    global pb, lorax_client
    if pb is None:
        try:
            pb = Predibase(api_token=API_TOKEN)
            lorax_client = pb.deployments.client("llama-3-1-8b-instruct")
            return True
        except Exception as e:
            print(f"Error initializing client: {e}")
            return False
    return True

def standardize_text(text):
    """
    Standardize the text by:
    1. Removing all trailing whitespace
    2. Always ending with a period
    """
    # Strip any trailing whitespace
    text = text.strip()
    
    # If empty text, return just a period
    if not text:
        return "."
    
    # Add a period if the text doesn't already end with one
    if not text.endswith(".") and not text.endswith("!") and not text.endswith("?"):
        text = text + "."
    
    return text

def extract_prediction(response_text):
    """
    Extract the binary prediction (0 for human, 1 for AI) from the model's response.
    """
    # Clean up the response and look for the binary label
    clean_response = response_text.strip()
    
    # First look for exact matches to 0 or 1
    if clean_response == "0" or clean_response.lower() in ["0", "0.0", "zero", "human"]:
        return 0
    elif clean_response == "1" or clean_response.lower() in ["1", "1.0", "one", "ai"]:
        return 1
    
    # If no exact match, try to find the first number in the response
    match = re.search(r'\b[01]\b', clean_response)
    if match:
        return int(match.group(0))
    
    # If still no match, look for keywords
    if "human" in clean_response.lower() and "ai" not in clean_response.lower():
        return 0
    elif "ai" in clean_response.lower() or "model" in clean_response.lower():
        return 1
    
    # Default to unknown (represented as -1)
    return -1

def detect_ai_text(text, adapter_id=None):
    """
    Use the fine-tuned model to detect if text is human or AI-generated.
    Returns 0 for human, 1 for AI, -1 for unknown.
    """
    if not initialize_clients():
        raise Exception("Failed to initialize clients")
    
    # Standardize the text
    text = standardize_text(text)
    
    instruction = f"Determine if the following text was written by a human or an AI model:\n\n{text}"
    
    try:
        if adapter_id:
            # Using adapter
            response = lorax_client.generate(
                instruction,
                adapter_id=adapter_id,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE
            )
        else:
            # No adapter - just base model
            adapter_id = f"{MODEL_NAME}/1"  # Default to version 1
            response = lorax_client.generate(
                instruction,
                adapter_id=adapter_id,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE
            )
        
        # Extract prediction (0, 1, or -1 for unknown)
        return extract_prediction(response.generated_text)
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        raise Exception(f"Error during inference: {str(e)}")

def simple_sentence_tokenize(text):
    """
    A simple sentence tokenizer that splits by period, exclamation mark, and question mark.
    """
    # First replace common abbreviations to avoid splitting them
    text = text.replace("Mr.", "Mr")
    text = text.replace("Mrs.", "Mrs")
    text = text.replace("Dr.", "Dr")
    text = text.replace("etc.", "etc")
    text = text.replace("e.g.", "eg")
    text = text.replace("i.e.", "ie")
    
    # Split on sentence ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore the periods in abbreviations
    sentences = [s.replace("Mr", "Mr.") for s in sentences]
    sentences = [s.replace("Mrs", "Mrs.") for s in sentences]
    sentences = [s.replace("Dr", "Dr.") for s in sentences]
    sentences = [s.replace("etc", "etc.") for s in sentences]
    sentences = [s.replace("eg", "e.g.") for s in sentences]
    sentences = [s.replace("ie", "i.e.") for s in sentences]
    
    return sentences

# Request models
class TextRequest(BaseModel):
    text: str
    adapter_id: Optional[str] = None
    min_words: Optional[int] = 5

class DocumentRequest(BaseModel):
    text: str
    adapter_id: Optional[str] = None
    min_words: Optional[int] = 5

# Response models
class TextResponse(BaseModel):
    prediction: int
    text: str

class SentenceResult(BaseModel):
    sentence: str
    prediction: int
    index: int

class DocumentResponse(BaseModel):
    results: List[SentenceResult]
    overall_prediction: int
    human_count: int
    ai_count: int
    unknown_count: int
    total_sentences: int
    analyzed_sentences: int

@app.get("/")
async def root():
    """API health check."""
    return {"status": "online", "message": "AI Content Detector API is running"}

@app.post("/detect", response_model=TextResponse)
async def detect_single(request: TextRequest):
    """
    Detect whether a single piece of text was written by a human or AI.
    
    Returns:
        0 for human-written text
        1 for AI-generated text
        -1 if classification is uncertain
    """
    try:
        prediction = detect_ai_text(request.text, request.adapter_id)
        return {
            "prediction": prediction,
            "text": request.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-document", response_model=DocumentResponse)
async def detect_document(request: DocumentRequest):
    """
    Analyze a document by breaking it into sentences and detecting each one.
    
    Returns detailed results for each sentence and an overall prediction.
    """
    try:
        # Split into sentences
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(request.text)
        except Exception:
            sentences = simple_sentence_tokenize(request.text)
        
        results = []
        human_count = 0
        ai_count = 0
        unknown_count = 0
        sentence_count = 0
        
        # Process each sentence that meets the minimum word count
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < request.min_words:
                continue
            
            sentence_count += 1
            prediction = detect_ai_text(sentence, request.adapter_id)
            
            if prediction == 0:
                human_count += 1
            elif prediction == 1:
                ai_count += 1
            else:
                unknown_count += 1
            
            results.append({
                "sentence": sentence,
                "prediction": prediction,
                "index": i
            })
        
        # Determine overall classification
        if human_count > ai_count:
            overall = 0  # Human
        elif ai_count > human_count:
            overall = 1  # AI
        else:
            # Tie or all unknown
            if ai_count > 0:
                overall = 1  # AI if there's any AI content
            elif human_count > 0:
                overall = 0  # Human if there's any human content
            else:
                overall = -1  # Unknown if all unknown
        
        return {
            "results": results,
            "overall_prediction": overall,
            "human_count": human_count,
            "ai_count": ai_count,
            "unknown_count": unknown_count,
            "total_sentences": len(sentences),
            "analyzed_sentences": sentence_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Initialize the client when starting the API
    initialize_clients()
    # Run the server with uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
