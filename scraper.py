import os
import re
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

def fetch_html(url):
    """Fetch HTML content from a URL."""
    response = requests.get(url, headers=HEADERS)
    return response.text if response.status_code == 200 else ""

def clean_html_content(html_content):
    """Extract meaningful text from HTML while removing unnecessary elements."""
    soup = BeautifulSoup(html_content, "html.parser")
    for element in soup(["header", "footer", "nav", "script", "style", "aside", "sup", "table"]):
        element.decompose()

    main_content = soup.find("div", id="bodyContent") or soup.find("main") or soup.find("article")
    if not main_content:
        all_divs = soup.find_all("div")
        main_content = max(all_divs, key=lambda div: len(div.get_text()), default=soup)

    paragraphs = [p.get_text(" ", strip=True) for p in main_content.find_all("p")]
    text = "\n\n".join(paragraphs)

    text = re.sub(r"\[\d+\]|\[edit\]", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)

    return text

def scrape_and_chunk(urls, chunk_size=300, chunk_overlap=50):
    """Scrape text from URLs and split into smaller chunks for better FAISS retrieval."""
    all_chunks = []
    
    for url in urls:
        print(f"🔍 Fetching content from: {url}")
        html_content = fetch_html(url)
        cleaned_text = clean_html_content(html_content)

        # ✅ Ensure chunking is small enough to generate multiple vectors
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(cleaned_text)

        all_chunks.extend(chunks)

    print(f"✅ Total Chunks Created: {len(all_chunks)}")  # ✅ Debugging: Check chunk count
    return all_chunks

