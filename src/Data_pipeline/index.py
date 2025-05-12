from bertopic import BERTopic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS as LCFAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import numpy as np
import faiss
import os

DB_FAISS_PATH = 'vectorstore/db_faiss'
#os.makedirs(DB_FAISS_PATH, exist_ok=True)  # Ensure the directory exists

# Load dataset
df = pd.read_csv('Combined Admissions Data.csv')
docs = df['Content'].tolist()
section_headers = df['Section Header'].tolist()

# Load BERTopic model
topic_model = BERTopic.load("Jsevere/bertopic-admissions-mmr-keybert")
topic_info_df = topic_model.get_topic_info()
doc_info = topic_model.get_document_info(docs)
topic_ids = topic_info_df["Topic"].tolist()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

# Group content by topic
topic_content_dict = {}
for i, tid in enumerate(topic_model.topics_):
    content = doc_info.iloc[i]["Document"]
    header = section_headers[i]
    combined_content = f"{header}\n{content}"
    topic_content_dict.setdefault(tid, []).append(combined_content)

# Split documents by topic into chunks
all_chunks = {}
for tid, docs in topic_content_dict.items():
    all_chunks[tid] = []
    for doc in docs:
        chunks = text_splitter.split_text(doc)
        all_chunks[tid].extend(chunks)
        
# Prepare embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
combined_chunks = [chunk for chunks in all_chunks.values() for chunk in chunks]
vectors = embeddings.embed_documents(combined_chunks)
vector_array = np.array(vectors).astype("float32")

# GPU FAISS setup
try:
    res = faiss.StandardGpuResources()
    d = vector_array.shape[1]
    nlist = 100
    quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    gpu_index.train(vector_array)
    gpu_index.add(vector_array)
except Exception as e:
    raise RuntimeError(f"Failed to initialize FAISS GPU resources: {e}")

# Wrap with LangChain FAISS store
faiss_store = LCFAISS.from_texts(
    texts = combined_chunks,
    embedding=embeddings,
    index=gpu_index,
)

# Save index
try:
    faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), f"{DB_FAISS_PATH}/index.faiss")
    print("FAISS index saved to disk in CPU format.")
except Exception as e:
    raise RuntimeError(f"Failed to save FAISS index: {e}")


# Optional: Transfer the index to GPU when needed
cpu_index = faiss.read_index(f"{DB_FAISS_PATH}/index.faiss")
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
print("FAISS index transferred to GPU.")


#faiss_store.save_local(DB_FAISS_PATH)
#print("FAISS index loaded successfully.")

# Reload for confirmation
#cpu_index = faiss.read_index(f"{DB_FAISS_PATH}/index.faiss")
#faiss_store = LCFAISS(
    #embedding_function=embeddings,
    #index=cpu_index,
    #docstore=faiss_store.docstore,
    #index_to_docstore_id={i: str(i) for i in range(len(combined_chunks))}
#)

