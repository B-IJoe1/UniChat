import chainlit as cl
from Llm_pipeline.pipeline import create_qa_chain, qa_bot_answer, load_llm, custom_prompt

# Create the chain globally
qa_chain, retriever = create_qa_chain(load_llm=load_llm, custom_prompt=custom_prompt)

@cl.on_chat_start
async def start():
    cl.user_session.set("qa_chain", qa_chain) # Store the chain in the user session
    cl.user_session.set("retriever", retriever) # Store the retriever in the user session
    await cl.Message(content="Hi! I’m your Salem State Admissions Assistant. How can I help?").send()

@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the chain from the user session
    qa_chain = cl.user_session.get("qa_chain")
    retriever = cl.user_session.get("retriever")
    # Call the QA chain using the user's input
    response = await qa_bot_answer(user_input=message.content, qa_chain=qa_chain, retriever=retriever)

    # Send the response back to the user
    await cl.Message(content=response).send()
