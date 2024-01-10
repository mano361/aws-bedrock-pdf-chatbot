import json
import os
from datetime import datetime

import streamlit as st

from langchain.vectorstores.pgvector import PGVector
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain_community.llms import Bedrock
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

import boto3
from botocore.exceptions import NoCredentialsError

from pdf_embedding_generator import create_connection, func_pdf_embeddings

from dotenv import load_dotenv
load_dotenv()

# AWS S3 Configuration
AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
AWS_SECRET_KEY = os.environ["AWS_SECRET_KEY"]
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]

bedrock_client = boto3.client("bedrock-runtime", 'us-east-1', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                              aws_secret_access_key=os.environ['AWS_SECRET_KEY'])

max_token_limit = 100

vector_db_collection_name = os.environ['PG_VECTOR_COLLECTION_NAME']
hf_embeddings = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2")

model = Bedrock(client=bedrock_client, model_id="anthropic.claude-instant-v1")


def copy_to_local(uploaded_file):
    """ For copying the files to the local """

    local_folder = "user_uploaded_files"
    os.makedirs(local_folder, exist_ok=True)
    local_path = os.path.join(local_folder, uploaded_file.name)
    with open(local_path, "wb") as local_file:
        local_file.write(uploaded_file.getvalue())
    return os.path.dirname(os.path.abspath(local_path))


# Function to upload file to AWS S3
def upload_to_s3(file_path_var, bucket_name, object_name=None):
    """ For copying the user uploaded file(s) to S3 bucket to keep a track of user uploads """

    folder_name = "webapp_completed_files"
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
        )
        modified_obj_name = f"{folder_name}/{object_name}"
        s3.upload_file(file_path_var, bucket_name, modified_obj_name)
        return True
    except FileNotFoundError:
        st.error("The file was not found.")
        return False
    except NoCredentialsError:
        st.error("Credentials not available.")
        return False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False


# Chatbot Page with Continuous Chat
def chatbot_page(vector_store_obj):
    """ This function is responsible for the entire chatbot logic """
    st.title("Chatbot Page")

    # Initialize chat history
    if "messages" not in st.session_state:
        print("True for Messages")
        st.session_state.messages = []

    for each_conversation in st.session_state.messages:
        with st.chat_message('user'):
            st.markdown(each_conversation[0])
        with st.chat_message('assistant'):
            st.markdown(each_conversation[1])

    # Accept user input
    if prompt := st.chat_input("Enter your question here!"):
        # Add user message to chat history
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=model, retriever=vector_store_obj.as_retriever(), return_source_documents=True)

            chain_output = qa_chain(
                {"question": prompt, "chat_history": st.session_state.messages})

            assistant_response = chain_output['answer']

            st.markdown(assistant_response)
        st.session_state.messages.append((prompt, assistant_response))

    # Reset button to clear chat history
    if st.session_state.messages and st.button("Reset Chat"):
        # Clear chat history
        st.session_state.messages = []
        st.rerun()


# Page 1: File Upload
def page_file_upload(connection_str):
    """ Copies the uploaded file to the S3 Bucket and stores the embeddings in the DB """

    st.title("PDF File Upload")
    local_file_path = ''

    # File Upload Section
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)

    # Streamlit App Logic for File Upload
    if uploaded_file:
        st.success("File(s) successfully uploaded!")

        for each_uploaded_file in uploaded_file:

            local_file_path = copy_to_local(each_uploaded_file)

    else:
        st.info("Please upload a file.")

    # Upload to AWS S3
    if uploaded_file and st.button("Generate & Store Embeddings"):
        for each_file in os.listdir(local_file_path):
            with st.spinner('Generating and Storing Embeddings for %s' % each_file):
                current_file_path = os.path.join(local_file_path, each_file)

                output = func_pdf_embeddings(current_file_path, connection_str)
                if not output:
                    st.error("Unable to generate embeddings for the uploaded file. Retry again!")
                elif output and upload_to_s3(current_file_path, S3_BUCKET_NAME, each_file):
                    os.remove(current_file_path)
                else:
                    st.error("File copy to AWS S3 failed. Please check the error message.")
            print("Completed!")
        st.success('Generated and Stored Embeddings in the Vector DB for all the uploaded file(s)')


# Streamlit App
def main():
    """ Main function for routing the user requests to the clicked page """

    st.sidebar.title("Navigation")
    pages = ["File Upload", "Chatbot Page"]

    selection = st.sidebar.radio("Go to", pages)

    if selection == "File Upload":
        st.session_state.messages = []
        connection_string = create_connection()
        page_file_upload(connection_string)
    elif selection == "Chatbot Page":
        connection_string = create_connection()
        db = PGVector(
            embedding_function=hf_embeddings,
            collection_name=vector_db_collection_name,
            connection_string=connection_string,
        )
        chatbot_page(db)


if __name__ == "__main__":
    main()
