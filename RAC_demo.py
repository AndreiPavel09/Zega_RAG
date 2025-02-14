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
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings

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

def process_pdf_files(pdf_files, pdf_dir, indexed_files):
    """Process PDF files to create document objects."""
    all_documents = []
    for pdf_file in pdf_files:
        if pdf_file not in indexed_files:
            pdf_file_path = os.path.join(pdf_dir, pdf_file)
            documents = SimpleDirectoryReader(input_files=[pdf_file_path]).load_data()
            document = Document(text="\n\n".join([doc.text for doc in documents]), metadata={"filename": pdf_file})
            all_documents.append(document)
            print(f"Processing file: {pdf_file}")
        else:
            print(f"Skipping already indexed file: {pdf_file}")
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
            print(f"Processing file: {docx_file}")
        else:
            print(f"Skipping already indexed file: {docx_file}")
    return all_documents

def process_files(pdf_files, docx_files, file_dir, indexed_files):
    """Process both PDF and DOCX files to create document objects."""
    all_documents = []
    all_documents.extend(process_pdf_files(pdf_files, file_dir, indexed_files))
    all_documents.extend(process_docx_files(docx_files, file_dir, indexed_files))
    
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
            indexed_files.append(doc.metadata['filename'])
        print("Initial documents indexed:", indexed_files)
        save_indexed_files(indexed_files)
    else:
        # Daca exista deja, incarca indexarea
        sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir))
        new_documents = [doc for doc in documents if doc.metadata.get('filename') not in indexed_files]
        if new_documents:
            for doc in new_documents:
                sentence_index.insert(doc)
            sentence_index.storage_context.persist(persist_dir=save_dir)

            indexed_files.extend([doc.metadata['filename'] for doc in new_documents])
            print("Updated indexed_files:", indexed_files)

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

llm = Ollama(model="gemma2:9b", request_timeout=60.0)
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def main():
    start_time = time.time()
    st.title("RAG (Retrieval Augmented Generation)")

    # pdf_files = ["253_2024.pdf", "254_2019.pdf", "255_2019.pdf", "256_2024.pdf", "260_2019.pdf"]
    # pdf_dir = "sentence_index_upload_files"

    uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

    pdf_files = []
    docx_files = []

    if uploaded_files:
        os.makedirs("uploaded_files", exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join("uploaded_files", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            if uploaded_file.name.endswith(".pdf"):
                pdf_files.append(uploaded_file.name)
            elif uploaded_file.name.endswith(".docx"):
                docx_files.append(uploaded_file.name)

    indexed_files = load_indexed_files()

    all_documents = process_files(pdf_files, docx_files, uploaded_files, indexed_files)
    
    sentence_index = build_or_update_sentence_window_index(
        all_documents,
        indexed_files,
        llm=llm,
        embed_model=embed_model,
        sentence_window_size=3,
        save_dir="./sentence_index"
    )

    query_engine = create_query_engine(sentence_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    query = st.text_input("Enter your query:")
    if query:
        retrieval_context = []
        response = query_engine.query(query)
        st.subheader("Response:")
        if isinstance(response, str):
            st.write(response)
        else:
            st.write(response.response)

        context_used = [
            node.node.metadata.get('original_text', 'N/A')
            for node in response.source_nodes if node.node.text.strip()
        ]   
        st.subheader("Source Articles for the Response:")
        for context in context_used:
            st.write(f"Article: {context}")
            st.write("=" * 50)
        
        retrieval_context.append(context_used)

    end_time = time.time()
    total_run_time = end_time - start_time
    total_run_time_str = f"{total_run_time:.2f} seconds"
    print(f'Total Run Time: {total_run_time_str}')

    return query_engine

if __name__ == "__main__":
    main()