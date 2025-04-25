"""
Run inference with the AI content detector model.

This script provides an interface to test your fine-tuned model with custom inputs.
It analyzes text to determine whether it was written by a human or an AI model.
It can process individual sentences, paragraphs, or whole documents.
"""

import argparse
import re
import json
import os
import asyncio
from predibase import Predibase
from config import API_TOKEN, MODEL_NAME, TEMPERATURE
from concurrent.futures import ThreadPoolExecutor

# Set max_new_tokens to 2 as requested
MAX_NEW_TOKENS = 1

def extract_prediction(response_text):
    """
    Extract the binary prediction (0 for human, 1 for AI) from the model's response.
    """
    # Clean up the response and look for the binary label
    clean_response = response_text.strip()
    
    # First look for exact matches to 0 or 1
    if clean_response == "0" or clean_response.lower() in ["0", "0.0", "zero", "human"]:
        return 0, clean_response
    elif clean_response == "1" or clean_response.lower() in ["1", "1.0", "one", "ai"]:
        return 1, clean_response
    
    # If no exact match, try to find the first number in the response
    match = re.search(r'\b[01]\b', clean_response)
    if match:
        return int(match.group(0)), clean_response
    
    # If still no match, look for keywords
    if "human" in clean_response.lower() and "ai" not in clean_response.lower():
        return 0, clean_response
    elif "ai" in clean_response.lower() or "model" in clean_response.lower():
        return 1, clean_response
    
    # Default to unknown
    return None, clean_response

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

def detect_ai_text(client, text, adapter_id=None, debug=True):
    """
    Synchronous version of the detect_ai_text function.
    """
    """
    Use the fine-tuned model to detect if text is human or AI-generated.
    Debug mode is enabled by default to show the standardization process.
    """
    # Show original text if debug is enabled
    if debug:
        print("\n--- DEBUG OUTPUT ---")
        print(f"Original text: \"{text}\"")
    
    # Standardize the text
    original_text = text
    text = standardize_text(text)
    
    # Show standardized text if debug is enabled
    if debug:
        print(f"Standardized text (period added if needed): \"{text}\"")
    
    instruction = f"Determine if the following text was written by a human or an AI model:\n\n{text}"
    
    # Show final prompt if debug is enabled
    if debug:
        print(f"Full prompt sent to model:\n{instruction}")
        print("--- END DEBUG ---\n")
    
    try:
        if adapter_id:
            # Using adapter
            print(f"Generating with adapter: {adapter_id}")
            response = client.generate(
                instruction,
                adapter_id=adapter_id,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE
            )
        else:
            # No adapter - just base model
            response = client.generate(
                instruction,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE
            )
        
        # Access generated_text consistently
        prediction, full_response = extract_prediction(response.generated_text)
        return prediction, full_response
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        print("This could be due to:")
        print("1. The adapter not being found (check the adapter name and version)")
        print("2. Rate limits if using shared endpoints")
        print("3. The deployment not being ready yet")
        
        return None, f"Error: {str(e)}"

def simple_sentence_tokenize(text):
    """
    A simple sentence tokenizer that splits by period, exclamation mark, and question mark.
    This is used as a fallback if NLTK is not available.
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

async def detect_ai_text_async(client, text, adapter_id=None, debug=True, request_id=None, semaphore=None):
    """
    Asynchronous version of the detect_ai_text function.
    Uses a semaphore to limit concurrent requests.
    """
    if semaphore:
        async with semaphore:
            if request_id is not None:
                print(f"\nProcessing request {request_id} (async)...")
            # Use ThreadPoolExecutor to run the synchronous client in a separate thread
            # This prevents blocking the event loop
            with ThreadPoolExecutor() as executor:
                return await asyncio.get_event_loop().run_in_executor(
                    executor, detect_ai_text, client, text, adapter_id, debug
                )
    else:
        # If no semaphore is provided, still run in a separate thread but without concurrency control
        with ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor, detect_ai_text, client, text, adapter_id, debug
            )

async def process_document_async(client, text, adapter_id=None, debug=True, min_words=5, max_concurrent=99):
    """
    Asynchronous version of process_document.
    Processes up to max_concurrent sentences in parallel.
    """
    # Try to use NLTK for sentence tokenization, but fall back to simple tokenizer if not available
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        print(f"Using NLTK tokenizer. Found {len(sentences)} sentences in the document.")
    except Exception as e:
        print(f"NLTK not available or error: {e}")
        print("Using simple sentence tokenizer as fallback.")
        sentences = simple_sentence_tokenize(text)
        print(f"Found {len(sentences)} sentences in the document.")
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    print(f"Processing with up to {max_concurrent} concurrent requests...")
    
    # Store results for each sentence
    results = []
    tasks = []
    valid_sentences = []
    valid_indices = []
    
    # Create tasks for each sentence that has enough words
    for i, sentence in enumerate(sentences):
        # Skip sentences with fewer than min_words
        if len(sentence.split()) < min_words:
            print(f"\nSkipping sentence {i+1} (fewer than {min_words} words): \"{sentence}\"")
            continue
        
        valid_sentences.append(sentence)
        valid_indices.append(i)
        
        # Create an async task for this sentence
        task = detect_ai_text_async(
            client, sentence, adapter_id, debug, i+1, semaphore
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    print(f"Submitting {len(tasks)} tasks for processing...")
    predictions_responses = await asyncio.gather(*tasks)
    
    # Process the results
    sentence_count = len(valid_sentences)
    human_count = 0
    ai_count = 0
    unknown_count = 0
    
    # Process each result
    for i, (idx, sentence, (prediction, response)) in enumerate(zip(valid_indices, valid_sentences, predictions_responses)):
        # Determine the classification
        if prediction == 0:
            classification = "HUMAN"
            human_count += 1
        elif prediction == 1:
            classification = "AI"
            ai_count += 1
        else:
            classification = "UNKNOWN"
            unknown_count += 1
        
        print(f"Result for sentence {idx+1}: {classification}")
        
        # Store the result
        results.append({
            "index": idx,
            "sentence": sentence,
            "standardized": standardize_text(sentence),
            "prediction": prediction,
            "classification": classification,
            "response": response
        })
    
    # Calculate the overall document classification
    overall_classification = "HUMAN"
    if ai_count > human_count:
        overall_classification = "AI"
    elif ai_count == human_count:
        overall_classification = "MIXED"
    
    # Create a summary
    summary = {
        "total_sentences": len(sentences),
        "analyzed_sentences": sentence_count,
        "human_sentences": human_count,
        "ai_sentences": ai_count,
        "unknown_sentences": unknown_count,
        "human_percentage": human_count / sentence_count * 100 if sentence_count > 0 else 0,
        "ai_percentage": ai_count / sentence_count * 100 if sentence_count > 0 else 0,
        "overall_classification": overall_classification
    }
    
    return results, summary

def process_document(client, text, adapter_id=None, debug=True, min_words=5):
    """
    Process an entire document, analyzing each sentence.
    Returns a structured result with sentence-level analysis.
    """
    # Try to use NLTK for sentence tokenization, but fall back to simple tokenizer if not available
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        print(f"Using NLTK tokenizer. Found {len(sentences)} sentences in the document.")
    except Exception as e:
        print(f"NLTK not available or error: {e}")
        print("Using simple sentence tokenizer as fallback.")
        sentences = simple_sentence_tokenize(text)
        print(f"Found {len(sentences)} sentences in the document.")
    
    # Store results for each sentence
    results = []
    sentence_count = 0
    human_count = 0
    ai_count = 0
    unknown_count = 0
    
    # Process each sentence
    for i, sentence in enumerate(sentences):
        # Skip sentences with fewer than min_words
        if len(sentence.split()) < min_words:
            print(f"\nSkipping sentence {i+1} (fewer than {min_words} words): \"{sentence}\"")
            continue
        
        sentence_count += 1
        print(f"\nProcessing sentence {i+1}/{len(sentences)}: \"{sentence}\"")
        
        # Standardize and analyze the sentence
        prediction, response = detect_ai_text(client, sentence, adapter_id, debug)
        
        # Determine the classification
        if prediction == 0:
            classification = "HUMAN"
            human_count += 1
        elif prediction == 1:
            classification = "AI"
            ai_count += 1
        else:
            classification = "UNKNOWN"
            unknown_count += 1
        
        print(f"Prediction: {classification}")
        
        # Store the result
        results.append({
            "index": i,
            "sentence": sentence,
            "standardized": standardize_text(sentence),
            "prediction": prediction,
            "classification": classification,
            "response": response
        })
    
    # Calculate the overall document classification
    overall_classification = "HUMAN"
    if ai_count > human_count:
        overall_classification = "AI"
    elif ai_count == human_count:
        overall_classification = "MIXED"
    
    # Create a summary
    summary = {
        "total_sentences": len(sentences),
        "analyzed_sentences": sentence_count,
        "human_sentences": human_count,
        "ai_sentences": ai_count,
        "unknown_sentences": unknown_count,
        "human_percentage": human_count / sentence_count * 100 if sentence_count > 0 else 0,
        "ai_percentage": ai_count / sentence_count * 100 if sentence_count > 0 else 0,
        "overall_classification": overall_classification
    }
    
    return results, summary

def format_document_results(results, summary):
    """
    Format document results in a user-friendly way.
    """
    output = "=" * 50 + "\n"
    output += "       AI CONTENT DETECTOR RESULTS\n"
    output += "=" * 50 + "\n\n"
    
    # Add summary
    output += "SUMMARY:\n"
    output += f"Total sentences: {summary['total_sentences']}\n"
    output += f"Analyzed sentences: {summary['analyzed_sentences']} (skipped {summary['total_sentences'] - summary['analyzed_sentences']} short sentences)\n"
    output += f"Human-written sentences: {summary['human_sentences']} ({summary['human_percentage']:.1f}%)\n"
    output += f"AI-generated sentences: {summary['ai_sentences']} ({summary['ai_percentage']:.1f}%)\n"
    if summary['unknown_sentences'] > 0:
        output += f"Unknown classification: {summary['unknown_sentences']}\n"
    output += f"Overall classification: {summary['overall_classification']}\n\n"
    
    # Add sentence-by-sentence results
    output += "SENTENCE-BY-SENTENCE ANALYSIS:\n"
    for result in results:
        output += "-" * 50 + "\n"
        output += f"Sentence {result['index'] + 1}: "
        if result['classification'] == "HUMAN":
            output += "HUMAN-WRITTEN (0)\n"
        elif result['classification'] == "AI":
            output += "AI-GENERATED (1)\n"
        else:
            output += "UNKNOWN\n"
        
        output += f"Text: \"{result['sentence']}\"\n"
        output += f"Model response: {result['response']}\n"
    
    output += "=" * 50 + "\n"
    return output

def save_results(results, summary, output_file=None):
    """
    Save the analysis results to a file.
    """
    if not output_file:
        # Generate a timestamp-based filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"ai_detector_results_{timestamp}.json"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "summary": summary
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return output_file

def format_result(prediction, full_response):
    """Format the result in a user-friendly way"""
    if prediction == 0:
        result = "PREDICTION: HUMAN-WRITTEN (0)\n"
    elif prediction == 1:
        result = "PREDICTION: AI-GENERATED (1)\n"
    else:
        result = "PREDICTION: UNKNOWN\n"
        
    result += f"\nModel response: {full_response}"
    
    return result

def list_adapters(pb):
    """Try to list all available adapters for the user"""
    print("Attempting to list available adapters...")
    try:
        repos = pb.repos.list()
        print("\nAvailable repositories:")
        for repo in repos:
            print(f"- {repo.name}")
            try:
                versions = []
                for i in range(1, 6):  # Try versions 1-5
                    try:
                        adapter = pb.adapters.get(f"{repo.name}/{i}")
                        versions.append(str(i))
                    except:
                        pass
                if versions:
                    print(f"  Available versions: {', '.join(versions)}")
            except:
                print("  Unable to get versions")
    except Exception as e:
        print(f"Error listing adapters: {e}")

async def process_async(client, text, adapter_id=None, debug=True, min_words=5, max_concurrent=99):
    """
    Asynchronous wrapper for processing text.
    Determines whether to use async document processing or single text processing.
    """
    # If there are multiple sentences, use document processing
    if '.' in text and text.count('.') > 1:
        return await process_document_async(
            client, text, adapter_id, debug, min_words, max_concurrent
        )
    else:
        # For single sentences, use the async detect function but without a semaphore
        prediction, response = await detect_ai_text_async(client, text, adapter_id, debug)
        return prediction, response

async def main_async():
    """
    Asynchronous version of the main function.
    """
    parser = argparse.ArgumentParser(description='AI Content Detector')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing text to analyze')
    parser.add_argument('--output', type=str, help='Output file to save results (JSON format)')
    parser.add_argument('--min-words', type=int, default=5, help='Minimum number of words for a sentence to be analyzed')
    parser.add_argument('--deployment', type=str, help='Specific deployment name to use (default: llama-3-1-8b-instruct)')
    parser.add_argument('--adapter', type=str, help='Specific adapter ID to use (format: repo-name/version)')
    parser.add_argument('--list-adapters', action='store_true', help='List available adapters')
    parser.add_argument('--document', action='store_true', help='Process text as a document with sentence-level analysis')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output (debug is ON by default)')
    parser.add_argument('--max-concurrent', type=int, default=99, help='Maximum number of concurrent requests (default: 99)')
    args = parser.parse_args()
    
    # Debug mode is enabled by default, can be turned off with --no-debug
    debug_mode = not args.no_debug
    
    # Initialize Predibase client
    print("Initializing Predibase client...")
    pb = Predibase(api_token=API_TOKEN)
    
    print("Connected to Predibase")
    
    # List adapters if requested
    if args.list_adapters:
        list_adapters(pb)
        return
    
    # Get text to analyze
    text_to_analyze = None
    if args.text:
        text_to_analyze = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text_to_analyze = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    
    # Determine adapter ID
    adapter_id = args.adapter if args.adapter else f"{MODEL_NAME}/1"  # Default to version 1
    
    # Create client - use the exact pattern that works
    deployment_name = args.deployment if args.deployment else "llama-3-1-8b-instruct"
    print(f"Using deployment: {deployment_name}")
    
    try:
        # Create the lorax client directly using the pattern that works
        lorax_client = pb.deployments.client(deployment_name)
        
        # Test the client with a simple prompt if debug flag is set
        if debug_mode:
            print("Testing client connection...")
    except Exception as e:
        print(f"Error creating client: {e}")
        return
    
    if not text_to_analyze:
        # Interactive mode
        print("=" * 50)
        print("       AI CONTENT DETECTOR")
        print("=" * 50)
        print("This tool analyzes text to determine if it was written by a human or an AI model.")
        print("Enter text to analyze, or type 'exit' to quit.")
        print("=" * 50)
        
        try:
            while True:
                print("\nEnter text to analyze (or 'exit' to quit):")
                user_input = input("> ")
                
                if user_input.lower() == 'exit':
                    break
                
                if not user_input.strip():
                    print("Please enter some text to analyze.")
                    continue
                
                print("\nAnalyzing...")
                
                # Process as document if it has multiple sentences
                if args.document or '.' in user_input:
                    results, summary = await process_document_async(
                        lorax_client, user_input, adapter_id, debug=debug_mode, 
                        min_words=args.min_words, max_concurrent=args.max_concurrent
                    )
                    print("\n" + format_document_results(results, summary))
                    
                    # Save results if output file is specified
                    if args.output:
                        save_results(results, summary, args.output)
                else:
                    # Process as single sentence
                    prediction, full_response = await detect_ai_text_async(
                        lorax_client, user_input, adapter_id, debug=debug_mode
                    )
                    print("\n" + "=" * 50)
                    print(format_result(prediction, full_response))
                    print("=" * 50)
                
        except KeyboardInterrupt:
            print("\nExiting...")
    elif args.document:
        # Process as a complete document
        print("Analyzing document...")
        results, summary = await process_document_async(
            lorax_client, text_to_analyze, adapter_id, debug=debug_mode, 
            min_words=args.min_words, max_concurrent=args.max_concurrent
        )
        print("\n" + format_document_results(results, summary))
        
        # Save results if output file is specified
        if args.output:
            save_results(results, summary, args.output)
    else:
        # One-off analysis of a single piece of text
        print("Analyzing...")
        prediction, full_response = await detect_ai_text_async(
            lorax_client, text_to_analyze, adapter_id, debug=debug_mode
        )
        
        print("\n" + "=" * 50)
        print(format_result(prediction, full_response))
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='AI Content Detector')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing text to analyze')
    parser.add_argument('--output', type=str, help='Output file to save results (JSON format)')
    parser.add_argument('--min-words', type=int, default=5, help='Minimum number of words for a sentence to be analyzed')
    parser.add_argument('--deployment', type=str, help='Specific deployment name to use (default: llama-3-1-8b-instruct)')
    parser.add_argument('--adapter', type=str, help='Specific adapter ID to use (format: repo-name/version)')
    parser.add_argument('--list-adapters', action='store_true', help='List available adapters')
    parser.add_argument('--document', action='store_true', help='Process text as a document with sentence-level analysis')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output (debug is ON by default)')
    args = parser.parse_args()
    
    # Debug mode is enabled by default, can be turned off with --no-debug
    debug_mode = not args.no_debug
    
    # Initialize Predibase client
    print("Initializing Predibase client...")
    pb = Predibase(api_token=API_TOKEN)
    
    print("Connected to Predibase")
    
    # List adapters if requested
    if args.list_adapters:
        list_adapters(pb)
        return
    
    # Get text to analyze
    text_to_analyze = None
    if args.text:
        text_to_analyze = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text_to_analyze = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    
    # Determine adapter ID
    adapter_id = args.adapter if args.adapter else f"{MODEL_NAME}/1"  # Default to version 1
    
    # Create client - use the exact pattern that works
    deployment_name = args.deployment if args.deployment else "llama-3-1-8b-instruct"
    print(f"Using deployment: {deployment_name}")
    
    try:
        # Create the lorax client directly using the pattern that works
        lorax_client = pb.deployments.client(deployment_name)
        
        # Test the client with a simple prompt if debug flag is set
        if debug_mode:
            print("Testing client connection...")
    except Exception as e:
        print(f"Error creating client: {e}")
        return
    
    if not text_to_analyze:
        # Interactive mode
        print("=" * 50)
        print("       AI CONTENT DETECTOR")
        print("=" * 50)
        print("This tool analyzes text to determine if it was written by a human or an AI model.")
        print("Enter text to analyze, or type 'exit' to quit.")
        print("=" * 50)
        
        try:
            while True:
                print("\nEnter text to analyze (or 'exit' to quit):")
                user_input = input("> ")
                
                if user_input.lower() == 'exit':
                    break
                
                if not user_input.strip():
                    print("Please enter some text to analyze.")
                    continue
                
                print("\nAnalyzing...")
                
                # Process as document if it has multiple sentences
                if args.document or '.' in user_input:
                    results, summary = process_document(
                        lorax_client, user_input, adapter_id, debug=debug_mode, min_words=args.min_words
                    )
                    print("\n" + format_document_results(results, summary))
                    
                    # Save results if output file is specified
                    if args.output:
                        save_results(results, summary, args.output)
                else:
                    # Process as single sentence
                    prediction, full_response = detect_ai_text(lorax_client, user_input, adapter_id, debug=debug_mode)
                    print("\n" + "=" * 50)
                    print(format_result(prediction, full_response))
                    print("=" * 50)
                
        except KeyboardInterrupt:
            print("\nExiting...")
    elif args.document:
        # Process as a complete document
        print("Analyzing document...")
        results, summary = process_document(
            lorax_client, text_to_analyze, adapter_id, debug=debug_mode, min_words=args.min_words
        )
        print("\n" + format_document_results(results, summary))
        
        # Save results if output file is specified
        if args.output:
            save_results(results, summary, args.output)
    else:
        # One-off analysis of a single piece of text
        print("Analyzing...")
        prediction, full_response = detect_ai_text(lorax_client, text_to_analyze, adapter_id, debug=debug_mode)
        
        print("\n" + "=" * 50)
        print(format_result(prediction, full_response))
        print("=" * 50)

if __name__ == "__main__":
    import sys
    
    if sys.platform == 'win32':
        # Windows requires this to properly handle asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        # Use asyncio.run() to start the async main function
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        print("Using fallback synchronous execution...")
        main()