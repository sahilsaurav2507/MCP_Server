import google.generativeai as genai
import os
import logging
import time
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API key configuration
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    logger.warning("GEMINI_API_KEY environment variable not set. API calls will fail.")

# Model configuration
MODEL_NAME = "gemini-1.5-pro"  # Can also use "gemini-1.5-flash" for faster, less expensive responses

def initialize_gemini():
    """Initialize the Gemini API with the API key"""
    try:
        genai.configure(api_key=API_KEY)
        logger.info("Gemini API initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Gemini API: {str(e)}", exc_info=True)
        raise

def query_model(prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """Query the Gemini model with the given prompt"""
    try:
        # Initialize Gemini if API key is available
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        initialize_gemini()
        
        # Log the prompt (truncated for brevity)
        truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"Querying Gemini model with prompt: {truncated_prompt}")
        
        # Configure the model
        generation_config = {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": max_tokens,
        }
        
        # Create the model
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config
        )
        
        # Generate response
        start_time = time.time()
        response = model.generate_content(prompt)
        
        # Extract the text from the response
        if hasattr(response, 'text'):
            result = response.text
        else:
            # Handle different response formats
            result = str(response)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {len(result)} characters in {elapsed_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        logger.error(f"Error querying Gemini model: {str(e)}", exc_info=True)
        raise Exception(f"Failed to get response from Gemini: {str(e)}")

def process_response(response: str) -> str:
    """Process the response from the Gemini model"""
    # Gemini responses are usually well-formatted, but we can add any post-processing here if needed
    return response

# Simple test function
def test_model():
    response = query_model("Explain what the Gemini API is in one paragraph.")
    print(f"Test response: {response}")
    return response

if __name__ == "__main__":
    # Test the model if this file is run directly
    test_model()