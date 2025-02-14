import asyncio
import datetime
import os
import json
import time
import uuid
import docx2txt

from pathlib import Path
from docx import Document
from collections import defaultdict

import streamlit as st
from streamlit.components.v1 import html

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Document, load_index_from_storage
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.prompts import LangchainPromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings


DEBUG = int(os.getenv("DEBUG", 0)) == 1
#logging.basicConfig(level=logging.INFO) #Streamlit handles logging

# Initialize LLM and Embed Model (move outside for persistence)
@st.cache_resource
def load_llm():
    return Ollama(model="llama3.2", request_timeout=60.0)

@st.cache_resource
def load_embed_model():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

llm = load_llm()
embed_model = load_embed_model()


INDEXED_FILES_RECORD = "indexed_files.json"

def load_indexed_files():
    """Load the record of indexed files."""
    if Path(INDEXED_FILES_RECORD).exists():
        with open(INDEXED_FILES_RECORD, 'r') as f:
            return json.load(f)
    return []

def save_indexed_files(indexed_files):
    """Save the record of indexed files."""
    with open(INDEXED_FILES_RECORD, 'w') as f:
        json.dump(indexed_files, f)

def process_pdf_files(pdf_files, pdf_dir, indexed_files):
    """Process PDF files to create document objects."""
    all_documents = []
    for pdf_file in pdf_files:
        if pdf_file not in indexed_files:
            pdf_file_path = os.path.join(pdf_dir, pdf_file)
            documents = SimpleDirectoryReader(input_files=[pdf_file_path]).load_data()
            document = Document(text="\n\n".join([doc.text for doc in documents]), metadata={"filename": pdf_file})
            all_documents.append(document)
            st.info(f"Processing file: {pdf_file}")
        else:
            st.info(f"Skipping already indexed file: {pdf_file}")
    return all_documents

def process_docx_files(docx_files, docx_dir, indexed_files):
    """Process DOCX files to create document objects."""
    all_documents = []
    for docx_file in docx_files:
        if docx_file not in indexed_files:
            docx_file_path = os.path.join(docx_dir, docx_file)
            text = docx2txt.process(docx_file_path)
            document = Document(text=text, metadata={"filename": docx_file})
            all_documents.append(document)
            st.info(f"Processing file: {docx_file}")
        else:
            st.info(f"Skipping already indexed file: {docx_file}")
    return all_documents

def process_files(pdf_files, docx_files, file_dir, indexed_files):
    """Process both PDF and DOCX files to create document objects."""
    all_documents = []
    all_documents.extend(process_pdf_files(pdf_files, file_dir, indexed_files))
    all_documents.extend(process_docx_files(docx_files, file_dir, indexed_files))
    
    return all_documents


@st.cache_resource
def build_or_update_sentence_window_index(
        documents,
        indexed_files,
        llm,
        embed_model, 
        sentence_window_size,
        save_dir="sentence_index"
):
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(documents)
        sentence_index.storage_context.persist(persist_dir=save_dir)

        for doc in documents:
            indexed_files.append(doc.metadata['filename'])
        st.info("Initial documents indexed:" + str(indexed_files))
        save_indexed_files(indexed_files)
    else:
        try:
            sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir))
        except Exception as e:
            st.error(f"Error loading index from storage: {e}. Rebuilding index...")
            sentence_index = VectorStoreIndex.from_documents(documents)
            sentence_index.storage_context.persist(persist_dir=save_dir)

            for doc in documents:
                indexed_files.append(doc.metadata['filename'])
            st.info("Initial documents indexed:" + str(indexed_files))
            save_indexed_files(indexed_files)

            return sentence_index

        new_documents = [doc for doc in documents if doc.metadata.get('filename') not in indexed_files]
        if new_documents:
            for doc in new_documents:
                sentence_index.insert(doc)
            sentence_index.storage_context.persist(persist_dir=save_dir)

            indexed_files.extend([doc.metadata['filename'] for doc in new_documents])
            st.info("Updated indexed_files:" + str(indexed_files))
            save_indexed_files(indexed_files)
        else:
            st.info("No new documents to index.")


    return sentence_index


@st.cache_resource
def get_sentence_window_query_engine(sentence_index, similarity_top_k, rerank_top_n):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model="BAAI/bge-reranker-base")

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


@st.cache_resource
def create_query_engine(sentence_index):
    """Create and configure the query engine."""
    query_engine = get_sentence_window_query_engine(sentence_index, similarity_top_k=15, rerank_top_n=6)

    my_prompt = PromptTemplate.from_template(
        """
        Using the document, answer the question based only on the following context: {context}
        If you don't know, say that you don't know.

        Question: {question}
        """
    )

    lc_prompt_tmpl = LangchainPromptTemplate(
        template=my_prompt,
        template_var_mappings={"query_str": "question", "context_str": "context"},
    )

    query_engine.update_prompts({"response_synthesizer:text_qa_template": lc_prompt_tmpl})
    return query_engine


# Initialize or load chats and query engine
class Chats:
    def __init__(self, cache_display_name, cache_minutes):
        self._chats = defaultdict(list)
        self._sessions = {}
        self._messages_in_progress = {}
        self._cache_minutes = cache_minutes
        self._cache_display_name = cache_display_name
        pdf_files = []
        docx_files = []
        uploaded_files = []

        indexed_files = load_indexed_files()
        file_dir = "uploaded_files"

        all_documents = process_files(pdf_files, docx_files, file_dir, indexed_files)
    
        self.sentence_index = build_or_update_sentence_window_index(
            all_documents,
            indexed_files,
            llm=llm,
            embed_model=embed_model,
            sentence_window_size=3,
            save_dir="./sentence_index"
        )

        self.query_engine = create_query_engine(self.sentence_index)

    def update_query_engine(self, all_documents, indexed_files):
        """Update the query engine with newly indexed documents."""
        self.sentence_index = build_or_update_sentence_window_index(
            all_documents,
            indexed_files,
            llm=llm,
            embed_model=embed_model,
            sentence_window_size=3,
            save_dir="./sentence_index"
        )
        self.query_engine = create_query_engine(self.sentence_index)

    def send_message(self, message: str, chat_id: str):
        self._chats[chat_id].append({"role": "human", "text": message, "dt": now()})
        if chat_id not in self._sessions:
            self._start_chat_session(chat_id)

        response = self._send_message(message, chat_id)
        self._chats[chat_id].append({"role": "AI", "text": response, "dt": now()})
        return response

    def _send_message(self, message, chat_id):
        session = self._sessions[chat_id]
        try:
            response = self.query_engine.query(message)
            return response.response
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return f"Error: {e}"

    def get_chat_history(self, chat_id):
        return self._chats.get(chat_id, [])  # Return chat history or an empty list if chat_id doesn't exist

    def print(self):
        for chat_id, messages in self._chats.items():
            print("\n\n\n\n" + "-" * 80)
            print(chat_id)

            for message in messages:
                print(f"  - {message['role']}: {message['text']}")
                if len(message["text"]) > 100:
                    print("\n\n" + "-" * 80)

        print("\n\n")

    def _start_chat_session(self, chat_id):
        if chat_id not in self._sessions:
            self._sessions[chat_id] = {"initialized": True}


    @staticmethod
    def _load_context(source_dir, context_path):
        if Path(context_path).exists():
            with open(context_path) as f:
                return f.read()

        context = []
        for path in sorted(os.listdir(source_dir)):
            if path.startswith("context"):
                continue

            with open(f"{source_dir}/{path}") as f:
                context.append(f"# FILE: {path}\n{f.read()}")

        context = "\n\n===\n".join(context)
        with open(context_path, "w") as f:
            f.write(context)

        return context

def now():
    return datetime.datetime.now().isoformat()


# Streamlit App
def main():
    st.set_page_config(page_title="FIA RAG", page_icon="/static/fia.png")
    st.title("FIA RAG")

    # Initialize chat history
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
    chat_id = st.session_state.chat_id

    if 'chats' not in st.session_state:
         st.session_state.chats = Chats(
            cache_display_name=os.getenv("GEMINI_CACHE_DISPLAY_NAME"),
            cache_minutes=int(os.getenv("GEMINI_CACHE_MINUTES", "30")),
    )

    chats = st.session_state.chats



    # Sidebar for file upload and indexed files list
    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader("Upload Documents (PDF, DOCX)", accept_multiple_files=True)

        if uploaded_files:
            uploaded_dir = "uploaded_files"
            os.makedirs(uploaded_dir, exist_ok=True)

            file_names = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(uploaded_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                file_names.append(uploaded_file.name)

            indexed_files = load_indexed_files()
            all_documents = process_files(file_names, [], uploaded_dir, indexed_files)
            chats.update_query_engine(all_documents, indexed_files)

            save_indexed_files(indexed_files)
            st.success("Files uploaded and indexed successfully!")

        st.subheader("Indexed Files")
        indexed_files = load_indexed_files()
        if indexed_files:
            for file in indexed_files:
                st.markdown(f"- {file}")
        else:
            st.info("No files indexed yet.")



    # Chat interface
    if "messages" not in st.session_state:
         st.session_state.messages = []
    chat_history = chats.get_chat_history(chat_id)

    # Display chat messages from history on app rerun
    for message in chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["text"])



    if prompt := st.chat_input("Type your question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "human", "text": prompt})
        # Display user message in chat message container
        with st.chat_message("human"):
            st.markdown(prompt)


        # Get the response
        with st.spinner("Thinking..."):
                response = chats.send_message(prompt, chat_id)

        # Display assistant response in chat message container
        with st.chat_message("AI"):
            st.markdown(response)
        st.session_state.messages.append({"role": "AI", "text": response})

if __name__ == "__main__":
    main()