import requests
from bs4 import BeautifulSoup
import json

url = "https://www.python.org/doc/"
html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

data = []

for h2 in soup.find_all("h2"):
    section = h2.get_text(strip=True)
    ul = h2.find_next_sibling("ul")
    if not ul:
        continue

    for li in ul.find_all("li", recursive=False):
        a = li.find("a")
        if not a or not a.get("href"):
            continue
        title = a.get_text(strip=True)
        href = a["href"]
        # text after the link in the same li (short description if present)
        desc = li.get_text(" ", strip=True)
        if desc.startswith(title):
            desc = desc[len(title):].strip(" -–: ")
        data.append(
            {
                "section": section,
                "title": title,
                "url": href,
                "description": desc,
            }
        )

with open("python_doc_index.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
