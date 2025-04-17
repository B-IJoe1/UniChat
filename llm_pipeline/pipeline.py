from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import sentence_transformers
from Chatbot_flow.index import DB_FAISS_PATH

embedding_model = sentence_transformers("Jsevere/bertopic-admissions-mmr-keybert") 


model = ("Jsevere/llama2-7b-admissions-qa-merged")


def load_llm():
    # Load the locally downloaded model here
    llm = {
        "model": model,
        "device_map": "auto",
        "max_new_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.1,
        "top_k": 3
    }
    return llm


def custom_prompt():
    
  custom_prompt_template=   """Prompt template for QA retrieval for each vectorstore
    """
  prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
  
  return prompt



def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        embedding_model = embedding_model,  #Adding the embedding model into the chain!
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#QA Model Function IMPORTANT to run QA bot
def qa_bot():
    embeddings = OpenAIEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings) #Located in index.py
    llm = load_llm()
    qa_prompt = custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa