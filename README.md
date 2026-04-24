# IdeaStar

## Overview

IdeaStar is an intelligent research topic generation tool that helps students and researchers efficiently develop research topics in two distinct ways.

### Features

**Basic Version** - For Research Beginners
- AI-powered automated search on Google Scholar based on your research interests
- Generates targeted, relevant research topics tailored to your field
- Perfect for students exploring new areas of study
- Simple, intuitive interface for quick topic discovery

**Advanced Version** - For Experienced Researchers
- Upload your own documents (PDF, DOCX, Markdown, etc.)
- AI analyzes your literature collection to extract research gaps and opportunities
- Generates precise research directions within your customized research scope
- Ideal for researchers with preliminary literature collections

## Project Structure

```
IdeaStar/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Requirements

- Python 3.8+

## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

## Key Dependencies

- **Streamlit**: Web application framework
- **ChromaDB**: Vector database for document embeddings
- **Sentence Transformers**: BAAI/bge-small-en-v1.5 for semantic embeddings
- **Hugging Face Hub**: Access to pre-trained models
- **SERP API**: Google Scholar search integration
- **Firecrawl**: Web content extraction
- **PyPDF**: PDF document processing
- **docx2txt**: Microsoft Word document processing
- **BeautifulSoup**: HTML parsing

## API Keys Configuration

The application requires the following API keys:

- **Hugging Face Token**: For model access
- **Firecrawl API Key**: For web scraping capabilities
- **SERP API Key**: For Google Scholar searches

Configure these as environment variables before running:

```bash
export HUGGING_FACE_KEY="your_key_here"
export FIRECRAWL_API_KEY="your_key_here"
export SERP_API_KEY="your_key_here"
```

## Technology Stack

- **Backend**: Python with Streamlit
- **NLP**: Hugging Face Transformers, ChromaDB
- **Search**: SERP API for Google Scholar
- **Document Processing**: PyPDF, docx2txt, Markdown
- **Vectorization**: Sentence Transformers

## Acknowledgments

- Built with Streamlit
- Powered by Hugging Face models
- Uses ChromaDB for vector storage
