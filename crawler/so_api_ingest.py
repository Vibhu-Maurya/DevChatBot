import os
import time
import json
import hashlib
import requests
from bs4 import BeautifulSoup

# Stack Exchange API Configuration
API_BASE = "https://api.stackexchange.com/2.3"
SITE = "stackoverflow"
PAGES_PER_TAG = 2  # 100 questions per page, 2 pages = 200 top questions per tag for now (can be increased)
PAGE_SIZE = 100

# We use the built-in 'withbody' filter which includes the HTML body for questions/answers
FILTER = "withbody"

TAGS = [
    "python",
    "fastapi",
    "docker",
    "kubernetes",
    "pytorch",
    "huggingface-transformers",
    "langchain",
    "amazon-web-services"
]

OUTPUT_FILE = "chunks.jsonl"

def clean_html(html_str):
    if not html_str:
        return ""
    soup = BeautifulSoup(html_str, "html.parser")
    return soup.get_text(separator="\n").strip()

def fetch_top_questions(tag):
    questions = []
    for page in range(1, PAGES_PER_TAG + 1):
        print(f"Fetching questions for tag '{tag}', page {page}...")
        url = f"{API_BASE}/search/advanced"
        params = {
            "site": SITE,
            "tagged": tag,
            "sort": "votes",
            "order": "desc",
            "accepted": "True",
            "pagesize": PAGE_SIZE,
            "page": page,
            "filter": FILTER
        }
        
        resp = requests.get(url, params=params)
        
        if resp.status_code != 200:
            print(f"Error fetching questions: {resp.text}")
            break
            
        data = resp.json()
        
        if "backoff" in data:
            sleep_time = data["backoff"] + 1
            print(f"Backoff requested by API. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            
        questions.extend(data.get("items", []))
        
        if not data.get("has_more"):
            break
            
        # Polite delay to avoid IP ban (without API key, limit is 300 req/day)
        time.sleep(1.5)
        
    return questions

def fetch_answers(answer_ids):
    answers = {}
    # SE API allows up to 100 IDs per request
    batch_size = 100
    for i in range(0, len(answer_ids), batch_size):
        batch = answer_ids[i:i+batch_size]
        ids_str = ";".join(map(str, batch))
        
        url = f"{API_BASE}/answers/{ids_str}"
        params = {
            "site": SITE,
            "filter": FILTER
        }
        
        print(f"Fetching {len(batch)} answers...")
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"Error fetching answers: {resp.text}")
            continue
            
        data = resp.json()
        
        if "backoff" in data:
            sleep_time = data["backoff"] + 1
            print(f"Backoff requested by API. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            
        for item in data.get("items", []):
            answers[item["answer_id"]] = item
            
        time.sleep(1.5)
        
    return answers

def main():
    print(f"Starting Stack Overflow Ingestion for tags: {TAGS}")
    
    total_chunks = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for tag in TAGS:
            questions = fetch_top_questions(tag)
            if not questions:
                continue
                
            # Extract accepted answer IDs
            answer_ids = [q["accepted_answer_id"] for q in questions if "accepted_answer_id" in q]
            if not answer_ids:
                continue
                
            # Fetch the answers
            answers_dict = fetch_answers(answer_ids)
            
            # Combine into chunks
            for q in questions:
                ans_id = q.get("accepted_answer_id")
                if not ans_id or ans_id not in answers_dict:
                    continue
                    
                ans = answers_dict[ans_id]
                
                q_title = clean_html(q.get("title", ""))
                q_body = clean_html(q.get("body", ""))
                a_body = clean_html(ans.get("body", ""))
                
                # Format chunk
                content = f"Question: {q_title}\n\n{q_body}\n\nAccepted Answer:\n{a_body}"
                
                chunk_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                url = q.get("link", "")
                
                chunk_record = {
                    "chunk_id": f"so_{q['question_id']}",
                    "chunk_hash": chunk_hash,
                    "url": url,
                    "domain": "stackoverflow.com",
                    "source": "stackoverflow",
                    "title": q_title,
                    "section": f"Tag: {tag}",
                    "content": content
                }
                
                f.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                total_chunks += 1
                
    print(f"Successfully processed and appended {total_chunks} Stack Overflow Q&A pairs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
