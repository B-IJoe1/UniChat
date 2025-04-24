from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from Topic_Router import classify_topic_and_get_response
from Data_pipeline.index import DB_FAISS_PATH
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

# LLM loader
def load_llm():
    model_id = "Jsevere/llama2-7b-admissions-qa-merged"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",  # bfloat16 on supported hardware, fallback otherwise
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
   
#Checking if the model is loaded correctly
    print("Model loaded successfully!")

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.1,
        top_k=3,
    )
    return HuggingFacePipeline(pipeline=pipe)

# Prompt template
def custom_prompt():
    template = """You are a helpful chat assistant for Salem State University admissions.
        Context: {context}
        Question: {question}
        """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

# Main QA bot setup
def create_qa_chain(load_llm,custom_prompt):
   embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
   db = FAISS.load_local(DB_FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
   
   
   qa_chain = RetrievalQA.from_chain_type(
       llm=load_llm,
       chain_type="stuff",
       retriever = db.as_retriever(search_type="similarity", k=3),
       #memory=ConversationBufferMemory(memory_key="chat_history", input_key="question"),
       return_source_documents=True,
       chain_type_kwargs={"prompt": custom_prompt}   
      
   )
   return qa_chain

print("QA bot initialized successfully with sentence transformer!")

# Return a callable function for Chainlit to use
async def qa_bot_answer(user_input, qa_chain, retriever):
    docs = retriever.get_relevant_documents(user_input)

    qa_bot_instance = qa_chain()
    bot_response = qa_bot_instance({"context": context, "question": user_input})
    
    if docs:
        context = "\n".join([doc.page_content for doc in docs])
    else:
        context = classify_topic_and_get_response(user_input)
        
        return bot_response



    #return await chain.acall(
        #{"context": context, "question": user_input},
       # callbacks=callbacks
   # )
