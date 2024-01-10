# Chat with your PDF using AWS Bedrock and Anthropic Claude LLM

This is a simple Streamlit chatbot app for talking to your uploaded PDF File(s)

## Features

- **PDF File Upload:** Upload PDF files to the web app, and store the generated embeddings in the Vector DB
- **Embedding Generation:** Embeddings are generated using Sentence Transformers - all-MiniLM-L6-v2
- **Chatbot Page:** Ask questions about the uploaded PDF File(s).

## Prerequisites

Before running the app, make sure you have the following:

- Python (version >= 3.7)
- AWS S3 credentials (Access Key and Secret Key)
- AWS Bedrock Claude Instant access
- Streamlit
- LangChain
- sentence-transformers
- boto3
- pdfminer.six
- pgvector
- InstructorEmbedding
- psycopg2-binary
- python-dotenv
