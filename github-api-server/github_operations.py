import requests
import os

GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = "XXXXXXXXXXXXXX" # GitHub token ::>  (Please use yours)

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    
    "Accept": "application/vnd.github.v3+json"
}

def get_repo_info(owner: str, repo: str):
    """
    Fetch repository information from GitHub.
    
    Parameters:
    - owner (str): The owner of the repository.
    - repo (str): The name of the repository.
    
    Returns:
    - dict: Repository information.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        repo_data = response.json()
        return {
            "name": repo_data["name"],
            "owner": repo_data["owner"]["login"],
            "description": repo_data["description"],
            "stars": repo_data["stargazers_count"],
            "forks": repo_data["forks_count"],
            "open_issues": repo_data["open_issues_count"]
        }
    else:
        return {"error": "Failed to fetch repository data"}

def star_repo(owner: str, repo: str):
    """
    Star a repository on GitHub.
    
    Parameters:
    - owner (str): The owner of the repository.
    - repo (str): The name of the repository.
    
    Returns:
    - dict: The result of the star operation.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/star"
    response = requests.put(url, headers=HEADERS)
    
    if response.status_code == 204:
        return {"message": "Repository starred successfully"}
    else:
        return {"error": "Failed to star the repository"}

def create_readme(owner: str, repo: str):
    """
    Create a README file in the repository.
    
    Parameters:
    - owner (str): The owner of the repository.
    - repo (str): The name of the repository.
    
    Returns:
    - dict: The result of the README creation.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/README.md"
    content = {
        "message": "Create README",
        "content": "IyMgVGhpcyBpcyBhIG5hb2ZpbGUgZm9yIHRoZSBwcm9qZWN0Lg=="  # encoded for Base64
    }
    response = requests.put(url, json=content, headers=HEADERS)
    
    if response.status_code == 201:
        return {"message": "README created successfully"}
    else:
        return {"error": "Failed to create README"}
