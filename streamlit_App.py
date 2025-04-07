import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="StudyBot", layout="wide")

# This is the first API key input; no need to repeat it in the main function.
# api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")
api_key = ""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks
        chunk_overlap=200  # More overlap helps context flow
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant. Use the context below to answer the question as thoroughly and accurately as possible.
    If the answer is not present in the context, respond with: "The answer is not available in the context." Do not attempt to guess or fabricate an answer.
    Use bullet points or concise paragraphs for clarity if the answer includes multiple parts.
    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.max_marginal_relevance_search(user_question, k=5)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])
    # docs_and_scores = new_db.similarity_search_with_score(user_question, k=5)
    # for doc, score in docs_and_scores:
    #     st.write("Score:", score)
    #     st.write("Content:", doc.page_content)

def main():
    st.header("studyBot - Your AI study companion")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")
import google.generativeai as genai

genai.configure(api_key=api_key)
for m in genai.list_models():
    print(m.name)

if __name__ == "__main__":
    main()
