# Model Context Protocol (MCP) Server for GitHub Operations with Gemini API

## **Introduction to MCP**
The **Model Context Protocol (MCP)** is an open protocol that standardizes how applications interact with AI models, tools, and data sources. MCP provides a **unified interface** for AI-driven applications, making it easier to integrate **tools, resources, prompts, and execution environments**.

MCP operates on a **client-server architecture**, where:
- **MCP Clients** initiate requests for AI-based tasks.
- **MCP Servers** expose functionalities such as fetching resources, invoking tools, and handling prompts.
- **AI Models** process these tasks, providing intelligent responses or executing actions.

This project implements an **MCP-based server** to interact with **GitHub** using the **Gemini API** to enable AI-driven automation.

---

## **Project Overview**
This MCP server allows AI models to:
- **Fetch GitHub repository information**
- **Star repositories** 
- **Create dynamic README files** 
- **Enable AI-driven interactions** via **Gemini API** 
- **Provide a chat-based web interface** for user interaction 

---

## **Setup & Installation**
### **Prerequisites**
Before running this project, ensure you have:
- **Python 3.7+**
- **A GitHub Personal Access Token (PAT)** for API authentication
- **A Gemini API Key** for AI-powered responses

### **Clone the Repository**
```bash
git clone https://github.com/your-username/github-mcp-server.git
cd github-mcp-server
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Set Up Environment Variables**
Create a `.env` file in the project root and add the following:

```ini
GITHUB_TOKEN=your_github_personal_access_token_here
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## **Running the MCP Server**
Start the Flask server by running:
```bash
python app.py
```
The server will be available at:  
**`http://127.0.0.1:5000/`**

---

## **API Endpoints**
### **Query Gemini AI**
**Endpoint**: `/mcp`  
**Method**: `POST`  
**Request Body**:
```json
{
    "action": "query_ai",
    "params": {
        "prompt": "Tell me about the repository 'octocat/Hello-World'."
    }
}
```
**Response Example**:
```json
{
    "ai_response": "The repository 'Hello-World' is a public repository owned by octocat."
}
```

### **Get GitHub Repository Information**
**Endpoint**: `/mcp`  
**Method**: `POST`  
**Request Body**:
```json
{
    "action": "get_repo_info",
    "params": {
        "owner": "octocat",
        "repo": "Hello-World"
    }
}
```
**Response Example**:
```json
{
    "name": "Hello-World",
    "owner": "octocat",
    "description": "My first repository on GitHub!",
    "stars": 100,
    "forks": 50,
    "open_issues": 2
}
```

### **Star a Repository**
**Endpoint**: `/mcp`  
**Method**: `POST`  
**Request Body**:
```json
{
    "action": "star_repo",
    "params": {
        "owner": "octocat",
        "repo": "Hello-World"
    }
}
```
**Response Example**:
```json
{
    "message": "Repository starred successfully"
}
```

### **Create a README File**
**Endpoint**: `/mcp`  
**Method**: `POST`  
**Request Body**:
```json
{
    "action": "create_readme",
    "params": {
        "owner": "octocat",
        "repo": "Hello-World"
    }
}
```
**Response Example**:
```json
{
    "message": "README created successfully"
}
```

---

## ** Using the Chat Interface**
A static **HTML chat interface** (`chat_interface.html`) is provided to interact with the MCP server.

### **How to Use**
1. Open `chat_interface.html` in your browser.
2. Type a prompt (e.g., `"Describe GitHub"`).
3. Click **Send** and receive a response from the Gemini API.

---

## **License**
This project is **MIT Licensed**. You can use, modify, and distribute it freely.

---

## **Contributing**
Feel free to submit **issues** or **pull requests** on [GitHub](https://github.com/sahilsaurav2507/github-mcp-server).

---

## **Future Improvements**
-  Add support for **AI-powered code generation**.
-  Enhance **error handling** for GitHub API failures.
-  Implement **OAuth-based authentication** for better security.

---

## **Author**
Created by **Sahil Saurav**  
Contact: sahilsaurav2507@gmail.com
GitHub: [sahilsaurav2507](https://github.com/sahilsaurav2507)

---

## **Credits**
- Uses **Google's Gemini API** for AI-based responses.
- Built with **Flask** for backend API handling.
- Integrated with **GitHub API** for repository operations.

---

## **Show Your Support**
If you like this project, please  the repository!
