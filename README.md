# Chat with your PDF using AWS Bedrock and Anthropic Claude LLM

This is a simple Streamlit chatbot app for talking to your uploaded PDF File(s)

## Features

- **PDF File Upload:** Upload PDF files to the web app, and store the generated embeddings in the Vector DB
- **Embedding Generation:** Embeddings are generated using Sentence Transformers - all-MiniLM-L6-v2
- **Chatbot Page:** Ask questions about the uploaded PDF File(s).

## Application Demo
![Application Demo](static/Bedrock_Recording.gif)

## Flow Diagram
![alt text](https://github.com/mano361/aws-bedrock-pdf-chatbot/blob/main/static/bedrock_app_flow_diagram.jpg)

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

## TO RUN

- Local: ```streamlit run streamlit_web_app_sample.py```
- EC2: ```python3 -m streamlit run streamlit_web_app_sample.py``` (Don't forget to enable the port 8501 in EC2)
