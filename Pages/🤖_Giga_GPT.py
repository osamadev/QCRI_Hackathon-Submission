import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from streamlit import session_state
from streamlit_feedback import streamlit_feedback

pdf_folder_path = "dataset/Giga"
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

def process_pdf_documents(pdf_folder_path: str):
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100, 
        length_function=len, 
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""]
    )
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

def load_or_persist_chromadb(collection_name: str, persist_directory: str) -> Chroma:
    client = chromadb.Client()
    existing_collections = client.list_collections()
    if collection_name not in existing_collections:
        print("Creating new collection and persisting documents.")
        client.create_collection(collection_name)
        chunked_documents = process_pdf_documents(pdf_folder_path)
        vectordb = Chroma.from_documents(
            documents=chunked_documents,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory
        )
        vectordb.persist()
    else:
        print("Collection already exists. Loading from ChromaDB.")
        vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=OpenAIEmbeddings())
    return vectordb

def create_agent_chain(vectordb):
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name, temperature=0.1)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(), 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )
    return qa_chain

def get_llm_response(query):
    collection_name = "giga_collection"
    persist_directory = "vector_db"
    vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=OpenAIEmbeddings())
    chain = create_agent_chain(vectordb)
    response = chain(query)
    return response

def embed_documents():
    client = chromadb.Client()
    collection_name = "giga_collection"
    persist_directory = "vector_db"
    client.delete_collection(collection_name)
    chunked_documents = process_pdf_documents(pdf_folder_path)
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    vectordb.persist()
    st.success("Documents re-embedded and Chroma index updated.")

def process_llm_response(llm_response):
    search_results = llm_response['result']
    sources = {}
    for source in llm_response["source_documents"]:
        doc_name = source.metadata['source'].replace('dataset/Giga\\', '')
        page_num = source.metadata['page']
        if doc_name not in sources:
            sources[doc_name] = {page_num}
        else:
            sources[doc_name].add(page_num)
    formatted_sources = [f"{doc} (Pages {' ,'.join(map(str, pages))})" for doc, pages in sources.items()]
    return search_results, formatted_sources

def format_sources_as_markdown(sources):
    formatted_sources = "\n".join([f"- {source}" for source in sources])
    return formatted_sources

def list_and_manage_pdfs(pdf_folder_path):
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
    selected_pdf = st.sidebar.selectbox("Select a PDF to delete", pdf_files)
    if st.sidebar.button(f"Delete '{selected_pdf}'"):
        os.remove(os.path.join(pdf_folder_path, selected_pdf))
        st.sidebar.success(f"Deleted {selected_pdf}")
        session_state.update_file_list = True

def upload_pdf(pdf_folder_path):
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        with open(os.path.join(pdf_folder_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Uploaded {uploaded_file.name}")
        session_state.update_file_list = True

def display_search_results(results):
    if results:
        st.markdown("### Search Results")
        st.markdown(results)  # Assuming 'results' is a Markdown-compatible string
    else:
        st.markdown("No results found.")


# Initialize chat history and messages if they don't exist in the session state
if 'chat_history_giga' not in st.session_state:
    st.session_state.chat_history_giga = []
if 'messages_giga' not in st.session_state:
    st.session_state.messages_giga = []


st.set_page_config(page_title="Giga Enterprise GPT", page_icon="🤖")
st.header("🤖 Giga Enterprise GPT")
st.caption("Unleash the power of AI in every search with Giga Enterprise GPT – your gateway to lightning-fast, intelligent, and precise information retrieval!")


if st.sidebar.button("Embed Documents"):
    embed_documents()

with st.sidebar.expander("Manage PDFs"):
    list_and_manage_pdfs(pdf_folder_path)
    upload_pdf(pdf_folder_path)

if 'update_file_list' not in session_state:
    session_state.update_file_list = False

if session_state.update_file_list:
    st.experimental_rerun()

# Initialize chat history and messages if they don't exist in the session state



def handle_chat(user_chat):
    if user_chat:
        response = get_llm_response(user_chat)
        results, sources = process_llm_response(response)
        formatted_sources = format_sources_as_markdown(sources)
        st.session_state.messages_giga.append({"role": "user", "content": user_chat})
        st.session_state.messages_giga.append({"role": "ai", "content": results})
        st.session_state.messages_giga.append({"role": "sources", "content": formatted_sources})

def display_chat_message(message_info):
    role = message_info["role"]
    message = message_info["content"]
    if role == "user":
        st.markdown(f"👤 **You**: {message}")
    elif role == "ai":
        st.markdown(f"🤖 **AI**: {message}")
    elif role == "sources":
        with st.expander("Sources and Citation"):
            st.markdown(message)
    # st.markdown("---")  # Separator

# Display chat history
for message_info in st.session_state.messages_giga:
    display_chat_message(message_info)

# render the chat input
prompt = st.chat_input("Enter your message...")
if prompt:
    handle_chat(prompt)
    # Clear the prompt
    st.experimental_rerun()