# QA-Chatbot: Local File Question Answering with LLM APIs

💬 **Query PDFs, Excel, and Word files locally using Hugging Face LLMs. Secure, fast, and user-friendly.**

---

## Overview

QA-Chatbot allows users to **upload documents locally** and interact with them through **natural language queries** using large language models (LLMs). Unlike typical RAG-based chatbots:

- ❌ **No vector database**
- ❌ **No chunking or embedding**
- ⚡ Files are processed **on-the-fly** for instant answers
- 🔐 Users provide their **own API keys**, ensuring security and no premium requirement

---

## Key Features

- **Local File Querying:** Upload PDFs, Excel sheets, or Word documents.  
- **LLM-powered Answers:** Uses Hugging Face models for on-demand queries.  
- **Safe API Key Usage:** Users supply their own tokens — keys are **never stored**.  
- **Optional Test Mode:** `test.py` verifies API connectivity and answer functionality.  
- **Excel Intelligence:** Supports natural language filtering, aggregation, trends, and basic visualization.  
- **Optimized Excel Handling:** Many queries can be performed **directly on Excel files** without requiring heavy LLM processing, making regular data operations faster and more efficient.  

---

## How It Works

1. **File Upload & Storage:**  
   - Files are stored **temporarily in `uploads/`**.  
   - No permanent storage; data exists only for the current session.  
   - **Excel files** can often be queried directly for counts, sums, averages, trends, etc., without needing API calls — this reduces unnecessary LLM usage.

2. **Query Processing:**  
   - Excel: Sheets parsed, columns normalized, filters and aggregations applied.  
   - PDF/Word: Text extracted and queried directly by LLM.

3. **LLM Interaction:**  
   - Uses Hugging Face LLMs to answer queries when needed.  
   - Users enter API keys **securely in the app**, never saved.

4. **Answer Display:**  
   - Streamlit shows answers directly.  
   - Excel results can be downloaded as CSV or Excel files.

---

## Why This Is Different

- **Not a typical RAG chatbot:** No embeddings, vector DB, or chunking.  
- **On-demand querying:** Each question processed independently.  
- **User-controlled security:** Only user API keys are used — safe and flexible.

---

## Setting Up API Keys

1. **Hugging Face:**  
   - Create an account at [Hugging Face](https://huggingface.co).  
   - Go to Profile → **Access Tokens → Create New Token**.  
   - Select **Fine-grained**, enable **Inference API calls**, and generate the token.

2. **OpenAI (optional):**  
   - Generate an API key in OpenAI dashboard if needed.

> ⚠️ **Never share or commit API keys.** The app is designed for **user-supplied keys only**.

---

## Getting Started

**Step 1: Clone the Repository**

sh

     git clone https://github.com/DeepakGowda-Official/QA_Chatbot.git
     cd QA_Chatbot

**Step 2: Set Up a Python Virtual Environment**
sh

     python -m venv venv
     venv\Scripts\activate 
     pip install -r requirements.txt

**Step 3: Run the App**
sh

     streamlit run app.py

**Step 4: Test API Key (Optional)**
sh

     python test.py

**Step 5: Upload Files & Query**

- Upload PDFs, Excel, or Word files through the app interface.
- For Excel, many queries can be processed directly **without calling the LLM**, making operations faster.
- Enter your **Hugging Face API key securely** when prompted — keys are never stored.

---


## Tech Stack

- **Python, Streamlit** – Frontend and backend interface  
- **Pandas, openpyxl** – Excel parsing and manipulation  
- **PyMuPDF (fitz), python-docx** – PDF and Word text extraction  
- **Hugging Face Hub (InferenceClient)** – LLM API calls  
- **Matplotlib** – Optional Excel visualization  

---

## Security & Privacy

- 🔐 **No API keys stored** — users provide their own tokens at runtime  
- 🗂 **Local processing** — uploaded files handled in session storage only  
- ✅ **Safe sharing** — no secrets are committed 

---

## Summary

QA-Chatbot is a **lightweight, secure, LLM-powered local file chatbot**:

- Query documents instantly  
- Safely use your own API keys  
- Avoid heavy infrastructure like vector databases  

It is **user-centric, secure, and easy to deploy**.

