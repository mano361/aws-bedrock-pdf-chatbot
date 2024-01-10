import os
#from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import PDFMinerLoader
from dotenv import load_dotenv
load_dotenv()


def create_connection():
    """ For creating the AWS PostgreSQL PG Vector DB connection """
    connection_string = PGVector.connection_string_from_db_params(
        driver='psycopg2',
        user=os.environ['AWS_DB_USERNAME'],
        password=os.environ['AWS_DB_PASSWORD'],
        host=os.environ['AWS_DB_HOSTNAME'],
        port=int(os.environ['AWS_DB_PORT']),
        database=os.environ['AWS_DB_NAME']
    )

    return connection_string


def func_pdf_embeddings(pdf_path, connection_str):
    """ For creating the embeddings and storing the uploaded file embeddings in the Vector DB """
    try:
        collection_name = os.environ['PG_VECTOR_COLLECTION_NAME']

        loader = PDFMinerLoader(pdf_path)

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        hf_embeddings = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2")

        db = PGVector.from_documents(
            embedding=hf_embeddings,
            documents=texts,
            collection_name=collection_name,
            connection_string=connection_str,
        )
        print("Embeddings completed")
        return True

    except Exception as e:
        print(e)
        return False

