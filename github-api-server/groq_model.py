import requests
import os

# Set up the Groq API URL and key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Update with the correct Groq API URL
GROQ_API_KEY = "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"  # Your Groq API key directly
def query_groq_ai(prompt: str):
    """
    Function to send a prompt to the Groq AI model and retrieve the response.
    
    Parameters:
    - prompt (str): The input text or prompt for the AI model.
    
    Returns:
    - dict: The response from the Groq AI model.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt  # The input text that we send to the model
    }

    try:
        response = requests.post(GROQ_API_URL, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()  # Return the model's response
        else:
            return {"error": f"Failed to get response: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def process_groq_response(response: dict):
    """
    Process the response from the Groq AI model.
    
    Parameters:
    - response (dict): The response from the Groq AI model.
    
    Returns:
    - str: The processed output.
    """
    if "error" in response:
        return f"Error: {response['error']}"
    else:
        return response.get('output', 'No output from model.')
