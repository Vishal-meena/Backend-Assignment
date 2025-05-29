# 🏥 AI-Powered Medical Claim Processing System

This project is a fully automated agentic system for processing health insurance claims by extracting, validating, and adjudicating information from medical documents (PDFs like bills, discharge summaries, ID cards) using Gemini LLM and CrewAI agents.

---

## 🚀 Architecture Overview

The system is built using:

- **FastAPI (Optional Web Layer)** – for exposing RESTful endpoints (not included in the current version)
- **CrewAI + Gemini-Pro** – orchestrated agents with memory to handle tasks like classification, extraction, validation, and decision-making
- **ChromaDB** – vector store used as a medical validation knowledge base
- **PyPDF2** – for PDF text extraction
- **Pydantic** – for structured response models

### 🔁 Pipeline Flow

1. **PDF Text Extraction**  
   Raw PDFs are processed using PyPDF2 to extract text.

2. **Document Classification Agent**  
   Identifies each document as a bill, discharge summary, ID card, or other.

3. **Specialized Extraction Agents**  
   Based on type, agents extract structured data (e.g., hospital name, total cost, admission date).

4. **Validation Agent**  
   Uses a medical knowledge base (stored in ChromaDB) to detect:
   - Missing documents
   - Inconsistent patient info
   - Date mismatches

5. **Claim Decision Agent**  
   Makes a final call on **approval or rejection**, providing:
   - Reason
   - Confidence score

---

## 🤖 AI Tools Used

### ✅ Gemini-Pro
Used via `langchain_google_genai.ChatGoogleGenerativeAI` for:
- Document classification
- Structured extraction from unstructured text
- Logical validation and decision-making

### 🧠 Vector DB (ChromaDB)
Used to store and retrieve validation rules, such as:
- Required fields in a bill or discharge summary
- Consistency checks across documents

---

