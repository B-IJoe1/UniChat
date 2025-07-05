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
from langchain_core.runnables.base import RunnableMap, RunnablePassthrough
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

def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3}) 
    print("Retriever loaded successfully.")
    return retriever


def create_qa_chain(load_llm, custom_prompt):
   llm = load_llm() 
   prompt = custom_prompt()
   retriever = load_retriever()
                                    
   
   # Create a document QA chain
   #print(f"Question_answer_chain before StrOutputParser: {type(question_answer_chain)}")

    #create_retrieval_chain expects a retriever and a chain that takes a question and context
   question_answer_chain = create_stuff_documents_chain(llm,prompt) #This chain will take a question and context, and return an answer
   #qa_chain = create_retrieval_chain(retriever,
                                      #question_answer_chain) #This retrieval chain will return a dictionary with the answer and the context used to generate it.

   
   qa_chain = RunnableMap({
        "context": retriever,
        "input": RunnablePassthrough() #You don't have user_input yet, so we use RunnablePassthrough() to pass the input directly for now
    }) | question_answer_chain | StrOutputParser() #This will create a chain that takes a question, retrieves context, and returns an answer

   #qa_chain = qa_chain.with_output_keys(["answer"])  # Ensure the output is a string
   return qa_chain


print("QA bot initialized successfully with sentence transformer!")

# Return a callable function for Chainlit to use
async def qa_bot_answer(user_input, qa_chain):
    #retriever = load_retriever()
    #docs = await retriever.ainvoke(user_input)
    #context = "\n".join([doc.page_content for doc in docs])

    bot_response = await qa_chain.ainvoke(user_input)

    print(bot_response)
    return bot_response #to StrOutputParser here, as the chain already returns the string

