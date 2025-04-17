from bertopic import BERTopic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


DATA_PATH = 'demodataPDFs/'
DB_FAISS_PATH = 'vectorstore/db_faiss'


topic_model = BERTopic(
    embedding_model="Jsevere/bertopic-admissions-mmr-keybert",
    language="english",
    calculate_probabilities=True,
    verbose=True,
)

#Retrieves the topic_model from BERTopic pipeline 
topic_id = topic_model.get_topic_info()["Topic"].tolist()
doc_info = topic_model.get_document_info()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

topic_content_dict = {}
# Example: Assigning topic_id
for i, topic_id in enumerate(topic_model.topics_):
    content = doc_info.iloc[i]["Document"]  # Access content from doc_info
    if topic_id not in topic_content_dict: # Check if the topic_id has not already been created in the dictionary before passing it into the dictionary
        topic_content_dict[topic_id] = []
        topic_content_dict[topic_id].extend(content)
    
#Print the dictionary 
print(f"Document {i} belongs to Topic {topic_content_dict}")


# Combine all chunks across topics if you want a single FAISS store (could create a store for each topic)
all_chunks = {}
for i, topic_id in enumerate(topic_id):
     doc = doc_info[i]
     chunks = text_splitter.split_text(content)
     if topic_id not in all_chunks: #Check if a topic id has not already been created in all_chunks. If not, create a new entry for it
        all_chunks[topic_id] = [] #Currently empty
        all_chunks[topic_id].extend(chunks) #Appends chunks to corresponding topic



# 4. Prepare embeddings and vectorstore for each topic group
embeddings = OpenAIEmbeddings()

# Combine all chunks into a single list for embedding
combined_chunks = [chunk for chunks in all_chunks.values() for chunk in chunks]

# Create a FAISS vector store using the embeddings
faiss_store = FAISS.from_texts(combined_chunks, embeddings)

    
# Save the FAISS index for later use
faiss_store.save_local(DB_FAISS_PATH)


# Load the FAISS index (later for faster retrieval w/o recomputing)
faiss_store = FAISS.load_local(DB_FAISS_PATH)