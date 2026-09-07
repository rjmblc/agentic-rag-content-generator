from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(directory:str):
    documents=[]

    pdf_files = Path(directory).glob("*.pdf")

    for pdf_file in pdf_files:
        print(f"Loading:{pdf_file}")

        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        documents.extend(docs)

    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    return chunks
