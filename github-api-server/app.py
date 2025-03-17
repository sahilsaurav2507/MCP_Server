from flask import Flask, request, jsonify
from flask_cors import CORS
from github_operations import get_repo_info, star_repo, create_readme
from gemini_model import query_model, process_response
import os
import logging
import json
import traceback
from werkzeug.exceptions import BadRequest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app, resources={r"/mcp": {"origins": os.environ.get("ALLOWED_ORIGINS", "*")}})

@app.route('/mcp', methods=['POST'])
def handle_mcp_request():
    """
    Main endpoint for handling MCP (Model Control Protocol) requests.
    Supports actions: query_ai, get_repo_info, star_repo, create_readme
    """
    try:
        # Log request details for debugging
        logger.info(f"Request headers: {dict(request.headers)}")
        request_data = request.get_data(as_text=True)
        logger.info(f"Request data preview: {request_data[:200]}...")
        
        # Validate request has content
        if not request_data:
            return jsonify({"error": "Empty request body", "success": False}), 400
            
        # Validate content type is JSON
        content_type = request.headers.get('Content-Type', '')
        if 'application/json' not in content_type.lower():
            return jsonify({
                "error": f"Content-Type must be application/json, got {content_type}", 
                "success": False
            }), 415
        
        try:
            data = json.loads(request_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return jsonify({"error": f"Invalid JSON format: {str(e)}", "success": False}), 400
            
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object", "success": False}), 400

        action = data.get("action")
        params = data.get("params", {})
        
        if not action:
            return jsonify({"error": "Missing required field: action", "success": False}), 400
        
        if not isinstance(params, dict):
            return jsonify({"error": "Params must be a JSON object", "success": False}), 400
        
        # Process based on the action type
        if action == "query_ai":
            return handle_query_ai(params)
        elif action == "get_repo_info":
            return handle_get_repo_info(params)
        elif action == "star_repo":
            return handle_star_repo(params)
        elif action == "create_readme":
            return handle_create_readme(params)
        else:
            return jsonify({"error": f"Invalid action: {action}", "success": False}), 400

    except BadRequest as e:
        logger.error(f"Bad request error: {str(e)}")
        return jsonify({"error": f"Bad request: {str(e)}", "success": False}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"An error occurred: {str(e)}", "success": False}), 500

def handle_query_ai(params):
    """Handle AI query requests"""
    prompt = params.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing required parameter: prompt", "success": False}), 400
    
    # parameters 
    max_tokens = params.get("max_tokens", 1024)
    temperature = params.get("temperature", 0.7)
    
    logger.info(f"Processing query_ai request with prompt length: {len(prompt)}")
    try:
        # Query 
        ai_response = query_model(prompt, max_tokens, temperature)
        processed_response = process_response(ai_response)
        return jsonify({
            "success": True, 
            "ai_response": processed_response
        })
    except Exception as e:
        logger.error(f"AI model error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "success": False}), 500

def handle_get_repo_info(params):
    """Handle GitHub repository info requests"""
    owner = params.get("owner")
    repo = params.get("repo")
    
    if not owner or not repo:
        return jsonify({"error": "Missing required parameters: owner and repo", "success": False}), 400
    
    logger.info(f"Getting repo info for {owner}/{repo}")
    try:
        result = get_repo_info(owner, repo)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error getting repo info: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

def handle_star_repo(params):
    """Handle GitHub repository starring requests"""
    owner = params.get("owner")
    repo = params.get("repo")
    
    if not owner or not repo:
        return jsonify({"error": "Missing required parameters: owner and repo", "success": False}), 400
    
    logger.info(f"Starring repo {owner}/{repo}")
    try:
        result = star_repo(owner, repo)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error starring repo: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

def handle_create_readme(params):
    """Handle GitHub README creation requests"""
    owner = params.get("owner")
    repo = params.get("repo")
    
    if not owner or not repo:
        return jsonify({"error": "Missing required parameters: owner and repo", "success": False}), 400
    
    logger.info(f"Creating README for {owner}/{repo}")
    try:
        result = create_readme(owner, repo)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error creating README: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0"
    }), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to get information about the AI model being used"""
    return jsonify({
        "model": "gemini-1.5-pro",
        "type": "Google Gemini API",
        "context_length": "1M tokens",
        "parameters": "Multi-modal LLM",
        "provider": "Google"
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    logger.info(f"Starting server on port {port} with debug={debug_mode}")

    logger.info(f"API key configured: {'Yes' if os.environ.get('GEMINI_API_KEY') else 'No'}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
