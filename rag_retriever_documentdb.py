from langchain_community.document_loaders import WebBaseLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.documentdb import DocumentDBVectorSearch
from pymongo import MongoClient
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.documents import Document
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import time
import boto3
from botocore.exceptions import ClientError
import ast 

load_dotenv(find_dotenv())


def get_secret():
    secret_name = os.getenv("DOCUMENTDB_CLUSTER_NAME")
    region_name = os.getenv("AWS_REGION")

    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e

    db_password = get_secret_value_response['SecretString']
    return db_password


db_username = os.getenv("DOCUMENTDB_USERNAME")
db_uri = os.getenv("DOCUMENTDB_URI")
db_password = get_secret()
MONGO_URI = f"mongodb://{db_username}:{db_password}@{db_uri}/?tls=true&tlsCAFile=global-bundle.pem&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false"


db_name = "vectorDB_for_RAG_documentdb"
index_name = "salaryse_rag_vector"
collection_name = "salaryse_rag_collection"


db_dir = os.path.join(os.getcwd(), "db")
data_dir = os.path.join(os.getcwd(), "data")


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
    loader = CSVLoader(file_path=file)
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

client = MongoClient(MONGO_URI)
db = client[db_name]
collection = db[collection_name]

vectorstore = DocumentDBVectorSearch.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    collection=collection,
    index_name=index_name,
)

vectorstore.create_index()

retriever_documentdb = vectorstore.as_retriever()

