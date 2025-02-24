# 🔍 AI Knowledge Retrieval System  

## 📖 Overview  
This project is a **semantic search engine** that scrapes AI-related content, indexes it using **FAISS**, and allows users to query it through an AI chatbot powered by **Google Gemini API**.

---

## 🚀 Features  
✅ **Web Scraper** – Extracts AI-related content from multiple sources.  
✅ **FAISS Vector Search** – Stores & retrieves relevant text chunks efficiently.  
✅ **Gemini API Integration** – Generates AI-powered responses.  
✅ **Streamlit UI** – User-friendly interface for querying AI knowledge.  

---

## ⚙️ Installation & Setup  

### 📌 Prerequisites  
- **Python 3.8+**  
- **Git**  
- **Streamlit**  
- **Google Gemini API Key**  
- **FAISS (Vector Database)**  

### 📌 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Sanigababu/Text-Scraping-RAG.git
cd Text-Scraping-RAG
```
### 📌 2️⃣ Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # For Mac/Linux
# OR
.venv\Scripts\activate      # For Windows

```
### 📌 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt

```
### 📌 4️⃣ Set Up API Key
1.Create a .env file in the project root.
2.Add your Gemini API Key inside .env:
```bash
GEMINI_API_KEY=your_secret_key_here

```
3.Add your Gemini API Key inside .env


## 🚀  Running the Project

### 📌 1️⃣ Run the Web Scraper & FAISS Indexing
```bash
python Web_scraper.py
```
### 📌 2️⃣ Run the Streamlit UI
```bash
streamlit run app.py
```

## 🌍 Deploying on Streamlit Cloud
url: https://aipoweredknowledgesearch.streamlit.app/

## 📌 Example Queries
```
🔍 Question: "What is generative artificial intelligence?"
🧠 AI Response: "Generative AI is a subset of artificial intelligence that uses models to produce text, images, and videos..."
```
```
🔍 Question: "Latest AI research trends?"
🧠 AI Response: "Recent AI advancements include LLMs, multimodal models, and self-supervised learning..."
```


