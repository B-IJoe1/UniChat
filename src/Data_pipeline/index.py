from bertopic import BERTopic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
#import numpy as np
import faiss
import os
import torch

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Test PyTorch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

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
print("LangChain text splitter initialized successfully.")

# Group content by topic
topic_content_dict = {}
for i, tid in enumerate(topic_model.topics_):
    content = doc_info.iloc[i]["Document"]
    header = section_headers[i]
    combined_content = f"{header}\n{content}"
    topic_content_dict.setdefault(tid, []).append(combined_content)

# Split documents into chunks by topic
all_chunks = {}
for tid, docs in topic_content_dict.items():
    all_chunks[tid] = []
    for doc in docs:
        chunks = text_splitter.split_text(doc)
        all_chunks[tid].extend(chunks)

# Prepare embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
combined_chunks = [chunk for chunks in all_chunks.values() for chunk in chunks]
vectors = torch.tensor(embeddings.embed_documents(combined_chunks), dtype=torch.float32)

# GPU FAISS setup
try:
    res = faiss.StandardGpuResources()  # Initialize GPU resources
    d = vectors.shape[1]
    nlist = 100  # Number of clusters
    quantizer = faiss.IndexFlatL2(d)  # L2 distance quantizer
    gpu_index = faiss.IndexIVFFlat(quantizer, d, nlist)  # IVF index
    gpu_index = faiss.index_cpu_to_gpu(res, 0, gpu_index)  # Move index to GPU

    gpu_index.train(vectors.numpy())  # FAISS requires NumPy-like input
    gpu_index.add(vectors.numpy())
except Exception as e:
    raise RuntimeError(f"Failed to initialize FAISS GPU resources: {e}")

# Save index
try:
    faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), f"{DB_FAISS_PATH}/index.faiss")
    print("FAISS index saved to disk in CPU format.")
except Exception as e:
    raise RuntimeError(f"Failed to save FAISS index: {e}")

# Reload for confirmation
try:
    cpu_index = faiss.read_index(f"{DB_FAISS_PATH}/index.faiss")
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    print("FAISS index reloaded and transferred to GPU successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to reload FAISS index: {e}")