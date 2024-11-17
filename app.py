import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import faiss
import logging

# Configure logging
logging.basicConfig(
    filename="error.log", 
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [TWYN](https://www.twyn.org/)')

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def main():
    st.header("Chat with PDF")

    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf is not None:
        st.write(f"Uploaded PDF: {pdf.name}")
        try:
            pdf_reader = PdfReader(pdf)
            text = ""

            # Extract text from each page of the PDF
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            store_name = pdf.name[:-4]

            # Attempt to load existing FAISS index
            VectorStore = None
            if os.path.exists(f"{store_name}_faiss.index") and os.path.exists(f"{store_name}_docstore.pkl"):
                try:
                    index = faiss.read_index(f"{store_name}_faiss.index")
                    with open(f"{store_name}_docstore.pkl", "rb") as f:
                        docstore = pickle.load(f)
                    VectorStore = FAISS(index=index, docstore=docstore, index_to_docstore_id=None)
                    st.write('Embedding Loaded from Disk')
                except Exception as e:
                    logging.error(f"Error loading FAISS index: {str(e)}")
                    #st.warning("Unable to load previous embeddings. Computing new ones...")

            # Compute new embeddings if necessary
            if VectorStore is None:
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                faiss.write_index(VectorStore.index, f"{store_name}_faiss.index")
                with open(f"{store_name}_docstore.pkl", "wb") as f:
                    pickle.dump(VectorStore.docstore, f)
                st.write('Embedding Computation Completed and FAISS Index & Docstore Saved')

            # Handle user query
            query = st.text_input("Ask a question about the PDF:")
            if query:
                try:
                    docs = VectorStore.similarity_search(query=query, k=3)
                    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=query)
                    st.write(response)
                except Exception as e:
                    logging.error(f"Error during query processing: {str(e)}")
                    st.error("An error occurred while processing your query. Please try again.")
        except Exception as e:
            logging.error(f"Error processing PDF: {str(e)}")
            st.error("An error occurred while processing the PDF. Please check the file and try again.")

if __name__ == '__main__':
    main()
