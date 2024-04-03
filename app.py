import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from htmlTemplates import css


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
    llm = ChatOpenAI(temperature=1)
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
    response = st.session_state.conversation({"query": user_question})
    st.session_state.chat_history = response["chat_history"]
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)
            

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:", initial_sidebar_state="collapsed")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_question(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get PDF Text
                raw_text = get_pdf_text(pdf_docs)

                # Get the Text Chunks
                text_chunks = get_text_chunks(raw_text)

                # Create Vector Store
                vectorstore = get_vectorstore(text_chunks)
                st.info("Processing Completed :white_check_mark:")

                # Create Conversation Chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()