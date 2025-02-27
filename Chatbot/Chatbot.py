import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
# from langchain_community.chat_models import ChatOpenAI
import os

# OPENAI_API_KEY = "ENTER YOUR OWN GENERATED OPENAI API KEY"
GEMINI_API_KEY = "ENTER YOUR OWN GENERATED GEMINI API KEY"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# First step is to upload the files for training the model
st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Use the file for asking questions", type="pdf")

# Second step is to extract the text and break it into chunks

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        # The text of the entire pdf is now stored in text
        text += page.extract_text()
        # st.write(text)

    # Break the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = "\n",
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # Third step is to generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    # Fourth step is to store these embeddings in a vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Fifth step is taking an input question from the user
    input_query = st.text_input("Type your question here")

    # Sixth step is to do a similarity search between the user question and the vector DB and it will
    # return chunks of data which have the similarities
    if input_query:
        match = vector_store.similarity_search(input_query)
        # st.write(match)

        # Define the LLM
        llm = ChatGoogleGenerativeAI(
            model = "gemini-1.5-flash",
            temperature = 0.0,
            max_tokens = 1000
        )
        # llm = ChatOpenAI(
        #     openai_api_key = OPENAI_API_KEY,
        #     # The below three variables are fine tuned for specific use cases. As temperature increases,
        #     # randomness increases and we get long answers with not specific details.
        #     temperature = 0,
        #     max_tokens = 1000,
        #     model_name = "gpt-3.5-turbo"
        # )

        # Output the results
        chain = load_qa_chain(llm, chain_type = "stuff")
        response = chain.run(input_documents = match, question = input_query)
        st.write(response)