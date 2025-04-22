from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from Topic_Router import classify_topic_and_get_response
from Data_pipeline.index import DB_FAISS_PATH
from langchain_community.embeddings import HuggingFaceEmbeddings


# LLM loader
def load_llm():
    model_name = "Jsevere/llama2-7b-admissions-qa-merged"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None, torch_dtype="float32")
    print("Model loaded successfully!")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.1,
        top_k=3,
        device=0
    )
    return HuggingFacePipeline(pipeline=pipe)

# Prompt template
def custom_prompt():
    template = """You are a helpful chat assistant for Salem State University admissions. 
Only answer questions based on the information provided in the context. 
If you do not know the answer, say: "You have to email the University admissions office for personalized help."

Context: {context}

Question: {question}

Answer:"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Main QA bot setup
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = faiss_store.as_retriever(search_type="similarity", k=3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = load_llm()
    prompt = custom_prompt()

    return retriever, memory, llm, prompt
    # Return a callable function for Chainlit to use
def custom_chain(user_input, retriever, memory, llm, prompt):
        docs = retriever.get_relevant_documents(user_input)
        if docs:
            context = "\n".join([doc.page_content for doc in docs])
        else:
            context = classify_topic_and_get_response(user_input)

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            prompt=prompt,
            return_source_documents=True,
        )
        chain.invoke({"question": user_input, "context": context})
        
        return custom_chain  # 🟢 now returns a function ready to run