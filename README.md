# 🧠 UniChat: Topic-Aware University Q&A Chatbot

UniChat is an AI-powered chatbot designed to answer university-related questions with precision, context, and relevance. It uses topic modeling and vector search to retrieve the best answer from your knowledge base and presents it via a real-time interactive UI.

Built with:
- 🧩 **LangChain** (core + Hugging Face + community tools)
- 🔗 **Chainlit** for the frontend
- 🤗 **Hugging Face Transformers** for LLMs
- 🧠 **BERTopic** for topic clustering & chunking
- 🔍 **FAISS** for fast vector similarity search
- ⚡ **Accelerate** for GPU support on RunPod

---

## 🚀 Features

- Conversational interface powered by **Chainlit**
- Topic-aware chunking with **BERTopic**
- Smart document retrieval with **FAISS**
- Hugging Face-hosted fine-tuned LLM support
- Built for student help desks and university websites

---

## 🛠 Installation

Before running UniChat:

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Now pip install these packages:
```bash
pip install chainlit
pip install bertopic
pip install langchain
pip install langchain-community
pip install langchain-huggingface
pip install faiss-cpu
pip install accelerate  # Only required if running on RunPod or other GPU-based infra
```

3. Or you can use this easy install methos (Might take longer to run)

   ```bash
   pip install requirements.txt




