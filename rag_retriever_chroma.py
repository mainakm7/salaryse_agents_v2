from langchain_community.document_loaders import WebBaseLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.documents import Document
import os
import pandas as pd
import ast

db_dir = os.path.join(os.getcwd(), "db")
persistent_directory = os.path.join(db_dir, "vectorDB_for_RAG_chroma3")
data_dir = os.path.join(os.getcwd(), "metadata")

urls = [
    "https://www.salaryse.com/",
    "https://salaryse.com/privacy-policy",
    "https://salaryse.com/terms-conditions"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [doc for sublist in docs for doc in sublist]

csv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".csv")]
csv_docs = []

for file in csv_files:
    df = pd.read_csv(file)
    
    for _, row in df.iterrows():
        row_content = " ".join([str(value) for value in row[:-1]]) 
        
        try:
            metadata_dict = ast.literal_eval(row['metadata'])  
            metadata = {k: str(v) for k, v in metadata_dict.items()}  
        except (ValueError, SyntaxError):
            metadata = {}

        document = Document(
            page_content=row_content,
            metadata={"source": file, "row_index": _, **metadata}  
        )
        
        csv_docs.append(document)

all_docs = docs_list + csv_docs

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
doc_splits = []

for doc in all_docs:
    splits = text_splitter.split_documents([doc])
    for split in splits:
        split.metadata = doc.metadata  
        doc_splits.append(split)

embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

if not os.path.exists(persistent_directory):
    db = Chroma.from_documents(documents=doc_splits, embedding=embeddings, persist_directory=persistent_directory)
else:
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever_chroma = db.as_retriever()
