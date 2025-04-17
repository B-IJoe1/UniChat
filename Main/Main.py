import chainlit as cl 

 #chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting your gen AI bot!...")
    await msg.send()
    msg.content = "Welcome to Uni Chat!. Ask your question here:"
    await msg.update()



    # Set the chain in the user session
    # This allows us to access the chain in the on_message function
    # so we can call it with the user's message and get the answer.
    cl.user_session.set("chain", chain)
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["documents"]
    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"
    await cl.Message(content=answer).send()


    # This is the main entry point for the Chainlit app.
    "__main__" == __name__
   
   