import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from htmlTemplates import css
from streamlit_feedback import streamlit_feedback


# GET PDF TEXT
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# GET THE TEXT CHUNKS
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        # separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# CREATE VECTOR STORE
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstores = FAISS.from_texts(texts = text_chunks, embedding= embeddings)
    return vectorstores

# CREATE CONVERSATION CHAIN
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o", temperature=1)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='result')
    convesation_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    return convesation_chain

# HANDLE USER QUESTIONS (from input)
def handle_user_question(user_question):
    response = st.session_state.conversation({"query": user_question + "\nPlease include references and sources for your answers when available. Provide citations and sources if possible. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. If you don't have enough context to confidently answer the question, state: “Looks like I don't have enough information to answer that question. Feel free to try another prompt, and I will take another look at my knowledge bases, PULIN.”"})
    st.session_state.chat_history = response["chat_history"]
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            question_and_prompt = message.content.split('\n')
            message.content = question_and_prompt[0]
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)
    # feedback = streamlit_feedback(
    #     feedback_type="thumbs",
    #     optional_text_label="[Optional] Please provide an explanation",align="flex-start",
    # )

def main():
    load_dotenv()
    
    os.environ["OPENAI_API_KEY"]=os.environ.get("OPENAI_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"]="true"
    os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:", initial_sidebar_state="collapsed")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header(":rainbow[Chat with multiple PDFs] :books:")
    user_question = st.text_input(":violet[Ask a question about your documents:]")
    if user_question:
        handle_user_question(user_question)

    with st.sidebar:
        st.subheader(":blue[Your documents]")
        pdf_docs = st.file_uploader("Upload your PDFs and click on 'Process'", accept_multiple_files=True)
        if st.button(":blue[Process]"):
            with st.spinner("Processing..."):
                # Get PDF Text
                raw_text = get_pdf_text(pdf_docs)

                # Get the Text Chunks
                text_chunks = get_text_chunks(raw_text)

                # Create Vector Store
                vectorstore = get_vectorstore(text_chunks)
                st.success("Processing Completed :white_check_mark:")

                # Create Conversation Chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()