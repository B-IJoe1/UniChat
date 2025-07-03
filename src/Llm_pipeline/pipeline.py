from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from Topic_Router import classify_topic_and_get_response
from Data_pipeline.index import DB_FAISS_PATH
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.runnables.base import Runnable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser

# LLM loader
def load_llm():
    model_id = "Jsevere/llama2-7b-admissions-qa-merged"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",  # bfloat16 on supported hardware, fallback otherwise
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Transformers model loaded successfully.")

#Checking if the model is loaded correctly
    print("Model loaded successfully!")

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.1,
        top_k=1,
    )
    return HuggingFacePipeline(pipeline=pipe)

# Prompt template
def custom_prompt():
    template = """You are a helpful chat assistant for Salem State University admissions.
        Context: {context}
        Question: {input}
        """
    return PromptTemplate(template=template, input_variables=["context", "input"])
print(f"Custom prompt after PromptTemplate: {type(custom_prompt())}")

#os.environ["TOKENIZERS_PARALLELISM"] = "false" #disabling parallelism to avoid warnings
# Main QA bot setup
def create_qa_chain(load_llm, custom_prompt):
   embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
   db = FAISS.load_local(DB_FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
   retriever = db.as_retriever(search_kwargs={"k": 3}) 
   llm = load_llm() 
   prompt = custom_prompt()
   
    #Validate types
   #if not isinstance(llm, Runnable):
    #raise TypeError(f"Expected 'llm' to be a Runnable, got {type(llm)}")
   #if not isinstance(prompt, BasePromptTemplate):
      #raise TypeError(f"Expected 'prompt' to be a BasePromptTemplate, got {type(prompt)}")
   
   

   #memory = ConversationBufferMemory(return_messages=True,
                                    #memory_key="chat_history", 
                                    #input_key="input", 
                                    #output_key="answer")
   

   question_answer_chain = create_stuff_documents_chain(llm,prompt)
   
   print(f"Question_answer_chain before StrOutputParser: {type(question_answer_chain)}")
   qa_chain = create_retrieval_chain(retriever,
                                      question_answer_chain,
                                      ) 
   qa_chain = qa_chain | StrOutputParser() # This ensures the final output is a string
   print(f"QA chain after StrOutputParser: {type(qa_chain)}")
   return qa_chain

print("QA bot initialized successfully with sentence transformer!")

# Return a callable function for Chainlit to use
async def qa_bot_answer(user_input: str, qa_chain) -> str:
    bot_response = await qa_chain.acall({"input": user_input})
    print(f"bot_response type: {type(bot_response)}")
    print(f"bot_response value: {bot_response}") 
    return bot_response #No need to StrOutputParser here, as the chain already returns the string

