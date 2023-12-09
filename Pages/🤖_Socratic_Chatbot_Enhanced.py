from apikey import apikey
import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
import pandas as pd
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from streamlit_feedback import streamlit_feedback
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
import helper

index_name = 'langchain-retrieval-agent-002'
pinecone.init(
    api_key=st.secrets("pinecone_api_key"),
    environment="asia-southeast1-gcp-free"
)

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

model_name = 'text-embedding-ada-002'
os.environ["OpenAI_API_KEY"] = st.secrets["openai_api_key"]
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.environ["OpenAI_API_KEY"]
)

vectorstore = Pinecone(
    index, embed, text_field
)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=os.environ["OpenAI_API_KEY"],
    temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo", streaming=True
)

# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

socraticKB_tool = Tool(
        name='Socratic Knowledge Base',
        func=qa.run,
        description=(
            'Use this tool when answering questions related to philosophy, philosophical conceptions and distinctions of cognitive content.' 
            'Do not use this tool if the question or the prompt is not related to philosophy.' 
            'If you have been asked to answer any other questions which are not about philosphy, do not answer. '
        )
    )

os.environ["OpenAI_API_KEY"] = apikey

# Prompt templates
socratic_template = PromptTemplate(
    input_variables=['user_prompt', 'socratic_research'],
    template = """
    You are an expert chatbot in philosophy and you are responsible to answer and respond to the user's question: {user_prompt} in the philosophy based on the context provided context from  
    the socratic research tool: {socratic_research}. After answering the question , you can cite sources from which you generated this information.
    If you do not the answer, respond with I don't know, don't generate any answers you are not sure about.
    """
)

# App Framework
st.title('🦜💡 Socratic Chatbot')
st.caption("Unveil the wisdom of philosophy with 'Socratic Chatbot' 🤖📚, a GenAI-powered conversational guide that provides insightful answers based on the vast knowledge from Stanford Philosophy Encyclopedia 🌍💡")

# create the message history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# render older messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize memory
socratic_memory = ConversationBufferMemory(input_key='user_prompt', memory_key='chat_history', ai_prefix='\nAI', human_prefix='\nHuman')

socratic_chain = LLMChain(llm=llm, prompt=socratic_template, verbose=True, output_key='socratic', memory=socratic_memory)

# Clear message history button
if st.sidebar.button("Clear message history"):
    print("Clearing message history")
    socratic_memory.clear()
    st.session_state.messages = []
    
# render the chat input
prompt = st.chat_input("Enter your message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # render the user's new message
    with st.chat_message("user"):
        st.markdown(prompt)

    # render the assistant's response
    with st.chat_message("assistant"):
        retrival_container = st.container()
        message_placeholder = st.empty()
        
        kb_results = socraticKB_tool.run(prompt)
        response = socratic_chain.run(user_prompt=prompt, socratic_research=kb_results)

        
        if "messages" in st.session_state:
            chat_history = [helper.convert_message(m) for m in st.session_state.messages[:-1]]
        else:
            chat_history = []

        full_response = response
        

        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        # feedback = streamlit_feedback(
        #     feedback_type="thumbs",
        #     optional_text_label="[Optional] Please provide an explanation",
        # )

        feedback = streamlit_feedback(
            feedback_type="faces",
            optional_text_label="[Optional] Please provide an explanation",
        )

    # add the full response to the message history
    st.session_state.messages.append({"role": "assistant", "content": full_response})