# MultiModel RAG Pipeline

## Project Summary

This project implements a Multimodal Retrieval-Augmented Generation (RAG) pipeline that combines text and image understanding for intelligent document question-answering. The pipeline processes PDF documents, extracts both text and images, and embeds them using OpenAI’s CLIP model. These embeddings are stored in a FAISS vector database, enabling efficient similarity search for both modalities.

When a user submits a query, the system retrieves the most relevant text chunks and images from the PDF using unified CLIP embeddings. The retrieved content is then formatted and sent to Google Gemini (via the `langchain-google-genai` integration), which generates a context-aware answer leveraging both textual and visual information.

---

## Key Features

- **PDF Parsing:** Extracts text and images from PDF files using PyMuPDF (`fitz`).
- **Multimodal Embedding:** Uses CLIP to create unified embeddings for both text and images.
- **Vector Search:** Stores embeddings in FAISS for fast similarity-based retrieval.
- **Generative QA:** Integrates Google Gemini to answer questions using retrieved context.
- **Environment Configuration:** Supports API key management via `.env` file for secure authentication.
- **Extensible Design:** Easily adaptable to other document types or models.

---

## Example Use Cases

- Summarizing sections of a book or report that include both text and diagrams.
- Answering questions about historical events described and illustrated in a PDF.
- Extracting and explaining visual information (e.g., charts, figures) alongside text.

---

## Technologies Used

- Python, Jupyter Notebook
- LangChain, langchain-google-genai
- OpenAI CLIP (via HuggingFace Transformers)
- FAISS (Facebook AI Similarity Search)
- PyMuPDF (fitz), Pillow
- dotenv for environment management

---

## How It Works

1. **Load PDF:** The pipeline opens a PDF and iterates through each page.
2. **Extract & Embed:** Text is chunked and embedded; images are extracted, converted, and embedded.
3. **Store Embeddings:** All embeddings are stored in a FAISS vector store.
4. **Query:** User query is embedded and used to retrieve the most similar text and images.
5. **Generate Answer:** Retrieved context is sent to Gemini, which generates a comprehensive answer.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/multimodel_rag.git
cd multimodel_rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

- Add your Gemini API key to a `.env` file:
  ```
  GOOGLE_API_KEY=your-gemini-api-key
  ```
- Or set it in your notebook/script:
  ```python
  import os
  os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"
  ```

### 4. Prepare your PDF

- Place your PDF (e.g., `multimodel_sample.pdf`) in the project directory.

---

## Usage

Open and run the notebook `multimodel_rag.ipynb`. The pipeline will:

- Load and process the PDF
- Embed text and images
- Store embeddings in FAISS
- Retrieve relevant chunks for a query
- Generate answers using Gemini

### Example Queries

```python
queries = [
    "What happened between 1988 to 2013?",
    "Summarize the main findings of The DESIGN of EVERYDAY THINGS",
]
for query in queries:
    print(multimodal_pdf_rag_pipeline(query))
```

---

## File Structure

- `multimodel_rag.ipynb` — Main notebook with code and pipeline
- `multimodel_sample.pdf` — Example PDF for processing
- `.env` — Store your Gemini API key here

---

## Requirements

- Python 3.8+
- langchain
- langchain-google-genai
- transformers
- faiss
- pillow
- python-dotenv
- numpy
- scikit-learn
- pymupdf (fitz)
# RAG_MultiModel
