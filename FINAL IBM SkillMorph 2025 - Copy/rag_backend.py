import requests
import textwrap
from bs4 import BeautifulSoup

# Your IBM Credentials
API_KEY = "w7SE47lLgerfBjXFKvX0CV7xnhVTYNROSuWvBjIl5uvV"
PROJECT_ID = "bf842e0c-d435-4525-86b1-6ea4d2ba9d9a"
MODEL_ID = "ibm/granite-3-3-8b-instruct"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def get_access_token(api_key):
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "apikey": api_key,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def query_granite_chat(access_token, prompt_text):
    url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    body = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "parameters": {
            "temperature": 0.2,
            "max_tokens": 800,
            "top_p": 1.0
        },
        "messages": [{"role": "user", "content": prompt_text}]
    }
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def search_links_bing(query, max_links=5):
    print("Searching Bing for:", query)
    search_url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
    response = requests.get(search_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for li in soup.find_all('li', class_='b_algo'):
        a = li.find('a', href=True)
        if a and ("coursera.org" in a['href'] or "internshala.com" in a['href'] or "linkedin.com" in a['href']):
            links.append(a['href'])
        if len(links) >= max_links:
            break
    return links

def generate_rag_response(parsed_output):
    try:
        skills = parsed_output.get("skills", [])
        projects = parsed_output.get("projects", [])

        if not skills and not projects:
            return {
                "summary": "No sufficient resume data found.",
                "recommendations": "Please upload a more detailed resume."
            }

        skill_list = ", ".join(skills)
        project_list = ", ".join(projects)

        prompt = f"""
You are a career assistant. Analyze the following extracted resume data:

Skills: {skill_list if skill_list else 'N/A'}
Projects: {project_list if project_list else 'N/A'}

Give a short summary of the candidate's profile and personalized recommendations for career or learning paths.
"""

        access_token = get_access_token(API_KEY)
        ai_response = query_granite_chat(access_token, prompt)

        return {
            "summary": ai_response,
            "recommendations": "Explore more opportunities on: " + ", ".join(search_links_bing(skill_list)) if skills else "No suggestions found."
        }

    except Exception as e:
        return {
            "summary": "RAG processing failed.",
            "recommendations": str(e)
        }
