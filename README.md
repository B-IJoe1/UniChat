# 🧠 UniChat: University Q&A Chatbot

UniChat is an AI-powered chatbot designed to answer university-related questions with precision, context, and relevance. It uses topic modeling and vector search to retrieve the best answer from your knowledge base and presents it via a real-time interactive UI.

Built with:
- 🧩 **LangChain** (core + Hugging Face + community tools)
- 🔗 **Chainlit** for the frontend
- 🤗 **Hugging Face Transformers** for LLMs
- 🧠 **BERTopic** for topic clustering & chunking
- 🔍 **FAISS** for fast vector similarity search
- ⚡ **Accelerate** for GPU support on RunPod

---
## 🗂️ Documentation & Media

📓 **Step-by-step walkthrough** with visuals is available in:  
👉 [`UniChat Project Journal.pdf`](./UniChat%20Project%20Journal.pdf)

🎬 **Demo Video** (recommended to watch at 1.5x speed):  
👉 [`UniChat Movie Project demonstration.mp4`](./UniChat%20Movie%20Project%20demonstration.mp4)

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





