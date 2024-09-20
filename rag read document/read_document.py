
import chromadb
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import gradio as gr

BASE_URL = "http://172.27.59.54:11434"

MODEL_NAME = "llama3"

chroma_client = chromadb.PersistentClient(path=".")
collection = chroma_client.get_collection(name="text_collection")


embedding_model = OllamaEmbeddings(model="nomic-embed-text",base_url=BASE_URL)  
answer_model = Ollama(model=MODEL_NAME, base_url=BASE_URL)




def retrieve_data(query, embedding_model, collection:chromadb.Collection, top_k:int=5):
    """Retrieve data from collection."""
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embedding)
    return results['documents']

def generate_answer(answer_model,question,context):
    """Generate answer to question from context."""
    response = answer_model(f"Question: {question}\nContext: {context}\nAnswer:") 
    
    return response 


def process_document_and_query( question:str):
    """Process document and query."""
   

    print("and question = {} and collection = {} and embedding_model = {} and answer_model = {}".format(question, collection, embedding_model, answer_model))
   
    
    relevant_chunk_data = retrieve_data(question,embedding_model, collection, 5)

    
    full_context = " ".join(relevant_chunk_data[0])
    answer = generate_answer(answer_model, question, full_context)

    return answer


interface = gr.Interface(
    fn=process_document_and_query,
    inputs=[
        "textbox"
    ],
    outputs="text"
)

interface.launch()
















