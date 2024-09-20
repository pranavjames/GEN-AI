import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter

BASE_URL = "http://172.27.59.54:11434"
BASE_URL = "http://50.19.87.84:11434"

MODEL_NAME = "llama3"



# PersistentClient =  to make database persistent
chroma_client = chromadb.PersistentClient(path=".")
collection = chroma_client.create_collection("text_collection")
from langchain.schema import Document

embedding_model = OllamaEmbeddings(model="nomic-embed-text",base_url=BASE_URL)  


def get_chunks_built(text_content, chunk_size:int=70, chunk_overlap:int=0):
    """Get successive n-sized chunks from text."""
    documents = [Document(page_content=text_content)]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def get_embeddings(chunks, embedding_model:OllamaEmbeddings):
    return  [embedding_model.embed_documents([chunk.page_content])[0] for chunk in chunks]  


def store_data(text:str):
    """Store text in collection with embeddings."""
    chunks = get_chunks_built(text)



    embeddings = get_embeddings(chunks,embedding_model)  # Extract text from Document

    # Convert Document to string
    documents = [chunk.page_content for chunk in chunks]

    # Ensure embeddings are in the correct format
    for id, (document, embedding) in enumerate(zip(documents, embeddings)):
        # Check embedding type and convert if necessary
        if not isinstance(embedding, list):
            embedding = embedding.tolist()  # Convert numpy array to list if needed



        collection.add(ids=[str(id)], documents=[document], embeddings=[embedding])


file_path = "sample.txt"
with open(file_path, "r", encoding='utf-8') as file:
        text = file.read()

store_data(text)
    