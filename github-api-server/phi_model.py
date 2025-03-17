import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import time
import shutil
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store the model and tokenizer
model = None
tokenizer = None

# Model configuration - using a smaller model that works well on 6GB GPUs
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Much smaller than Phi-3
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "./model_cache")

def check_gpu_availability():
    """Check if CUDA is available and print GPU info"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        logger.info(f"GPU available: {gpu_name} with {gpu_memory:.2f} GB memory")
        return True
    else:
        logger.warning("CUDA not available. Using CPU instead.")
        return False

def download_model_from_hf():
    """Download the model files from Hugging Face Hub"""
    try:
        logger.info(f"Downloading model {MODEL_NAME}...")
        start_time = time.time()
        
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Clean up any existing partial downloads
        model_dir = os.path.join(CACHE_DIR, MODEL_NAME.split("/")[-1])
        if os.path.exists(model_dir):
            logger.info(f"Removing existing model directory: {model_dir}")
            shutil.rmtree(model_dir)
        
        # Download model files directly using snapshot_download
        model_path = snapshot_download(
            repo_id=MODEL_NAME,
            cache_dir=CACHE_DIR,
            local_dir=model_dir,
            local_dir_use_symlinks=False  # Don't use symlinks to avoid Windows issues
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Model download completed in {elapsed_time:.2f} seconds to {model_path}")
        
        # Verify the download by checking for key files
        if not os.path.exists(os.path.join(model_path, "config.json")):
            raise Exception(f"Model files not found in {model_path}. Download may have failed.")
            
        return model_path
    
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}", exc_info=True)
        raise

def load_model():
    """Load the model and tokenizer if not already loaded"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        logger.info(f"Loading {MODEL_NAME} model and tokenizer...")
        try:
            # Check GPU availability
            has_gpu = check_gpu_availability()
            
            # Check if model is already downloaded, if not download it
            model_dir = os.path.join(CACHE_DIR, MODEL_NAME.split("/")[-1])
            if not os.path.exists(os.path.join(model_dir, "config.json")):
                model_dir = download_model_from_hf()
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # Determine device and dtype based on available hardware
            if has_gpu:
                device = "cuda"
                logger.info("Loading model on GPU with half precision")
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16,  # Half precision is enough for this model
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                device = "cpu"
                logger.info("Loading model on CPU")
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise
    
    return model, tokenizer

def format_prompt(prompt):
    """Format the prompt for the TinyLlama chat model"""
    return f"<human>: {prompt}\n<assistant>:"

def query_model(prompt, max_new_tokens=1024, temperature=0.7):
    """Query the model with the given prompt"""
    try:
        # Load model if not already loaded
        llm, tok = load_model()
        
        # Log the prompt (truncated for brevity)
        truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"Querying model with prompt: {truncated_prompt}")
        
        # Format the prompt for the chat model
        formatted_prompt = format_prompt(prompt)
        
        # Tokenize input
        inputs = tok(formatted_prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(llm.device) for k, v in inputs.items()}
        
        # Generate response
        start_time = time.time()
        with torch.no_grad():
            outputs = llm.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tok.eos_token_id
            )
        
        # Decode the response
        full_response = tok.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<assistant>:" in full_response:
            response = full_response.split("<assistant>:")[-1].strip()
        else:
            response = full_response.replace(formatted_prompt, "").strip()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {len(response)} characters in {elapsed_time:.2f} seconds")
        
        return response
        
    except Exception as e:
        logger.error(f"Error querying model: {str(e)}", exc_info=True)
        raise Exception(f"Failed to get response from model: {str(e)}")

def process_response(response):
    """Process the response from the model"""
    # Remove any trailing model tokens that might appear
    if "<human>" in response:
        response = response.split("<human>")[0].strip()
    
    return response

# Function to unload model and free memory
def unload_model():
    global model, tokenizer
    
    if model is not None:
        del model
        model = None
        
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
        
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info("Model unloaded and memory freed")

# Simple test function
def test_model():
    response = query_model("Explain what a transformer model is in one paragraph.")
    print(f"Test response: {response}")
    return response

if __name__ == "__main__":
    # Test the model if this file is run directly
    test_model()
    unload_model()
