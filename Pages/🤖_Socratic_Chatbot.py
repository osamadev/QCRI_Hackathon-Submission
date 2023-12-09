import streamlit as st
import pinecone
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
import helper

# Initialize Pinecone with API key from Streamlit secrets
pinecone.init(
    api_key=st.secrets["pinecone_api_key"],
    environment="asia-southeast1-gcp-free"
)

index_name = 'langchain-retrieval-agent-002'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric='cosine', dimension=1536)

text_field = "text"
index = pinecone.Index(index_name)

# Using Streamlit secrets to access the OpenAI API key
openai_api_key = st.secrets["openai_api_key"]

embed = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    openai_api_key=openai_api_key
)

vectorstore = Pinecone(index, embed, text_field)

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo", streaming=True
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

socraticKB_tool = Tool(
    name='Socratic Knowledge Base',
    func=qa.run,
    description=(
        "This tool specializes in answering complex philosophical questions by leveraging the comprehensive and authoritative content from the Stanford Philosophy Encyclopedia. It's designed to provide in-depth, accurate responses to queries spanning a wide range of philosophical topics, making it an ideal resource for students, scholars, and anyone interested in exploring philosophical concepts and theories."
    )
)

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

# Initialize and display messages with unique state variable names
if "socratic_messages" not in st.session_state:
    st.session_state.socratic_messages = []

for message in st.session_state.socratic_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

socratic_memory = ConversationBufferMemory(input_key='user_prompt', memory_key='socratic_chat_history', ai_prefix='\nAI', human_prefix='\nHuman')

socratic_chain = LLMChain(llm=llm, prompt=socratic_template, verbose=True, output_key='socratic', memory=socratic_memory)

def handle_chat(user_input):
    st.session_state.socratic_messages.append({"role": "user", "content": user_input})

    kb_results = socraticKB_tool.run(user_input)
    response = socratic_chain.run(user_prompt=user_input, socratic_research=kb_results)

    full_response = response

    st.session_state.socratic_messages.append({"role": "assistant", "content": full_response})

    for message in st.session_state.socratic_messages[-2:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Clear message history button
if st.sidebar.button("Clear socratic message history"):
    socratic_memory.clear()
    st.session_state.socratic_messages = []

# Chat input and processing
prompt = st.chat_input("Enter your message...")
if prompt:
    handle_chat(prompt)
