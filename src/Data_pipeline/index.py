from bertopic import BERTopic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd

#DATA_PATH = 'demodataPDFs/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

#Load the dataset
df = pd.read_csv("Combined Admissions Data.csv") #Add /Users/josephsevere/Downloads/ in front if not running on AWS EC2 instance

docs = df['Content'].tolist()
section_headers = df['Section Header'].tolist() #Loading the section headers

#Loading the BERTopic model 
topic_model = BERTopic.load("Jsevere/bertopic-admissions-mmr-keybert")


#Getting topic information 
topic_info_df = topic_model.get_topic_info()
doc_info = topic_model.get_document_info(docs)
topic_ids = topic_info_df["Topic"].tolist()

#Initializing the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

topic_content_dict = {}
# Example: Assigning topic_id
for i, tid in enumerate(topic_model.topics_):
    content = doc_info.iloc[i]["Document"]
    header = section_headers[i]
    combined_content = f"{header}\n{content}"
    
      # Access content from doc_info
    if tid not in topic_content_dict: # Check if the topic_id has not already been created in the dictionary before passing it into the dictionary
        topic_content_dict[tid] = []
        topic_content_dict[tid].append(combined_content)
    
#Print the dictionary 
print(f"Document {i} belongs to Topic {topic_content_dict}")


#  Split documents into chunks by topic
all_chunks = {}
for tid, docs in topic_content_dict.items():
    all_chunks[tid] = []
    for doc in docs:
        chunks = text_splitter.split_text(doc)
        all_chunks[tid].extend(chunks)


# 4. Prepare embeddings and vectorstore for each topic group
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Combine all chunks into a single list for embedding
combined_chunks = [chunk for chunks in all_chunks.values() for chunk in chunks]

# Create a FAISS vector store using the embeddings
faiss_store = FAISS.from_texts(combined_chunks, embeddings)

    
# Save the FAISS index for later use
faiss_store.save_local(DB_FAISS_PATH)

# Load the FAISS index (later for faster retrieval w/o recomputing)
faiss_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Check if the FAISS index is loaded successfully
print("FAISS index loaded successfully.")
