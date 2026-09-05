import json
import os
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="PS04 - Customer Support Resolution Assistant")

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_data(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

ARTICLES = load_data("articles.json")
CUSTOMERS = load_data("customers.json")

class SupportRequest(BaseModel):
    account_id: str
    message: str

class ResolutionResponse(BaseModel):
    status: str
    response: str
    cited_articles: List[str] = []
    escalation_summary: Optional[dict] = None

def get_customer(account_id: str):
    return next((c for c in CUSTOMERS if c["account_id"] == account_id), None)

def search_articles(query: str):
    matched = []
    for art in ARTICLES:
        if any(word.lower() in art.get("content", "").lower() or word.lower() in art.get("title", "").lower() 
               for word in query.split()):
            matched.append(art)
    return matched if matched else ARTICLES

@app.post("/api/resolve", response_model=ResolutionResponse)
def resolve_ticket(req: SupportRequest):
    client = get_gemini_client()
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set.")

    customer = get_customer(req.account_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer account not found.")

    relevant_articles = search_articles(req.message)
    
    prompt = f"""
You are a resolution assistant for a broadband and mobile provider.

Customer Context:
- Name: {customer.get('name', 'Customer')}
- Plan: {customer.get('plan', 'Standard')}
- Status: {customer.get('billing_status', 'Active')}
- Recent Tickets: {customer.get('recent_tickets', [])}

Customer Message:
"{req.message}"

Knowledge Base Articles Available:
{json.dumps(relevant_articles, indent=2)}

Task:
1. Determine if the customer query can be answered accurately using the Knowledge Base Articles.
2. If YES: Draft a polite response. Ground every claim directly in the matching article and list article IDs cited.
3. If NO (complex issue, missing article, edge case, or user needs human intervention): Escalate to a human agent.

Output JSON Format strictly:
{{
  "action": "RESOLVE",
  "reply_text": "Ground response to customer or explanation for transfer",
  "cited_article_ids": ["KB001"],
  "escalation_summary": null
}}
OR for escalation:
{{
  "action": "ESCALATE",
  "reply_text": "Ground response to customer or explanation for transfer",
  "cited_article_ids": [],
  "escalation_summary": {{
      "issue_summary": "Brief summary of issue",
      "established_facts": "What was confirmed",
      "attempted_solutions": "What was tried or looked up"
  }}
}}
"""

    models_to_try = ['gemini-3.7-flash']
    response = None
    last_error = None

    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                if response:
                    break
            except Exception as e:
                last_error = e
                time.sleep(2)  # Wait 2 seconds before retry
        if response:
            break

    try:
        if not response:
            raise last_error

        result = json.loads(response.text)

        if result.get("action") == "RESOLVE":
            return ResolutionResponse(
                status="RESOLVED",
                response=result.get("reply_text", ""),
                cited_articles=result.get("cited_article_ids", [])
            )
        else:
            return ResolutionResponse(
                status="ESCALATED",
                response=result.get("reply_text", "Transferring ticket to human support desk."),
                cited_articles=[],
                escalation_summary=result.get("escalation_summary")
            )

    except Exception as e:
        return ResolutionResponse(
            status="ESCALATED",
            response="An unexpected issue occurred. Routing your ticket directly to a human agent.",
            escalation_summary={
                "issue_summary": req.message,
                "established_facts": f"Account ID: {req.account_id}",
                "attempted_solutions": f"System error: {str(e)}"
            }
        )

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)