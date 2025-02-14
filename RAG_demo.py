import asyncio
import time
import json
import os
import torch
import docx2txt
import streamlit as st
from pathlib import Path
from docx import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.prompts import LangchainPromptTemplate
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings

torch.classes.__path__ = []
INDEXED_FILES_RECORD = "indexed_files.json"
device = torch.device("cpu")
print(device)

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
    print("Indexed file")

def process_pdf_files(pdf_files, pdf_dir):
    """Process PDF files to create document objects."""
    all_documents = []
    for pdf_file in pdf_files:
        pdf_file_path = os.path.join(pdf_dir, pdf_file)
        documents = SimpleDirectoryReader(input_files=[pdf_file_path]).load_data()
        document = Document(text="\n\n".join([doc.text for doc in documents]), metadata={"filename": pdf_file})
        all_documents.append(document)
        print(f"Processing file: {pdf_file}")
    return all_documents

def process_docx_files(docx_files, docx_dir):
    """Process DOCX files to create document objects."""
    all_documents = []
    for docx_file in docx_files:
        docx_file_path = os.path.join(docx_dir, docx_file)
        text = docx2txt.process(docx_file_path)
        document = Document(text=text, metadata={"filename": docx_file})
        all_documents.append(document)
        print(f"Processing file: {docx_file}")
    return all_documents


def process_files(pdf_files, docx_files, file_dir):
    """Process both PDF and DOCX files to create document objects."""
    all_documents = []
    all_documents.extend(process_pdf_files(pdf_files, file_dir))
    all_documents.extend(process_docx_files(docx_files, file_dir))
    return all_documents

def build_or_update_sentence_window_index(
        documents,
        indexed_files,
        llm,
        embed_model,
        sentence_window_size,
        save_dir="sentence_index"
):
    node_parser = SentenceWindowNodeParser.from_defaults(
        # window_size-ul da numarul de propozitii de inainte si dupa cea cautata pentru context (3 = 1 inainte si 1 dupa cea selectata)
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # Cele 3 linii inlocuiesc Sentence_context
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    if not os.path.exists(save_dir):
        # Face indexarea
        sentence_index = VectorStoreIndex.from_documents(documents)
        # Salveaza indexarea in folder
        sentence_index.storage_context.persist(persist_dir=save_dir)

        for doc in documents:
            if doc.metadata['filename'] not in indexed_files:  # Double check for safety.
                indexed_files.append(doc.metadata['filename'])
        print("Initial documents indexed:", indexed_files)
        save_indexed_files(indexed_files) # Now save the updated list
    else:
        # Daca exista deja, incarca indexarea
        sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir))

        new_documents = [doc for doc in documents if doc.metadata.get('filename') not in indexed_files]

        if new_documents:
            for doc in new_documents:
                sentence_index.insert(doc)
                indexed_files.append(doc.metadata['filename']) # Add the filename here.
            sentence_index.storage_context.persist(persist_dir=save_dir)

            print("Updated indexed_files:", indexed_files)
            save_indexed_files(indexed_files) # Always save after inserting.

    return sentence_index

def get_sentence_window_query_engine(sentence_index, similarity_top_k, rerank_top_n):
    # Adauga propozitiile de langa
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    # Recalculeaza scorurile contextelor dupa adaugarea propozitiilor de langa
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model="BAAI/bge-reranker-base")

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine

def create_query_engine(sentence_index):
    """Create and configure the query engine."""
    query_engine = get_sentence_window_query_engine(sentence_index, similarity_top_k=15, rerank_top_n=6)

    template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)

    qa_template = PromptTemplate(template)

    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_template})
    return query_engine

llm = Ollama(model="llama3.2", request_timeout=60.0)
# embed_model= OllamaEmbedding(model_name="llama3.1:latest",base_url=base_url,ollama_additional_kwargs={"mirostat": 0})
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

def main():
    start_time = time.time()
    st.set_page_config(page_title="ZegaSoftware Demo", layout="wide")

    col1, col2 = st.columns([8, 1])
    with col1:
        st.title("Document Querying")
    with col2:
        logo_path = "logo.jpg"
        st.image(logo_path, width=180)

    uploaded_dir = "uploaded_files"
    os.makedirs(uploaded_dir, exist_ok=True)

    uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

    indexed_files = load_indexed_files()  # Load previously indexed files
    all_documents = []  # Initialize all_documents here

    if uploaded_files:
        new_files = []  # Track newly uploaded files to process
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in indexed_files:  # Only process if not indexed
                file_path = os.path.join(uploaded_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())  # Save file before processing
                new_files.append(uploaded_file.name)
            else:
                print(f"File {uploaded_file.name} already indexed, skipping.")

        # Only process new files that are not in the indexed_files list
        all_documents = process_files(new_files, [], uploaded_dir) #Don't pass indexed_files here


        # Build or update the sentence window index
        sentence_index = build_or_update_sentence_window_index(
            all_documents,
            indexed_files,
            llm=llm,
            embed_model=embed_model,
            sentence_window_size=3,
            save_dir="./sentence_index"
        )

        indexed_files = load_indexed_files() # REFRESH the list.
        query_engine = create_query_engine(sentence_index)

        # Now we have updated documents and the query engine, so we can proceed to querying
        st.success("Files uploaded and indexed successfully!")

        # indexed_files.storage_context.persist(persist_dir="sentence_index")

    else: #Handles app start with no file upload
        if os.path.exists("./sentence_index"):
            # sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./sentence_index"),  llm= llm, embed_model=embed_model)
            # query_engine = create_query_engine(sentence_index)
            sentence_index = build_or_update_sentence_window_index(
            all_documents,
            indexed_files,
            llm=llm,
            embed_model=embed_model,
            sentence_window_size=3,
            save_dir="./sentence_index"
        )
            query_engine=create_query_engine(sentence_index)
        else:
            sentence_index = None
            query_engine = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with st.sidebar:
        st.subheader("Indexed Files")
        if indexed_files:
            st.markdown(
            '''
            <style>
                .sidebar .sidebar-content {
                    width: 250px;
                }
                .indexed-files {
                    font-size: 18px;
                    height: 300px;
                    overflow-y: auto;
                    padding-left: 10px;
                    text-align: center;
                }
            </style>
            <div class="indexed-files">
                ''' + "<br>".join(indexed_files) + '''
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.write("Please upload a file to start querying.")

    query = st.text_input("Enter your query:")
    if query:
        if query_engine:
            retrieval_context = []
            response = query_engine.query(query)  # Ensure query_engine is initialized
            st.subheader("Response:")
            if isinstance(response, str):
                st.write(response)
            else:
                st.write(response.response)

            context_used = [
                node.node.metadata.get('original_text', 'N/A')
                for node in response.source_nodes if node.node.text.strip()
            ]
            retrieval_context.append(context_used)
        else:
            st.warning("Please upload and index documents before querying.")

    end_time = time.time()
    total_run_time = end_time - start_time
    total_run_time_str = f"{total_run_time:.2f} seconds"
    print(f'Total Run Time: {total_run_time_str}')

if __name__ == "__main__":
    main()