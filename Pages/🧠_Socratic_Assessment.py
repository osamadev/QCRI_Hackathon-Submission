import os
import streamlit as st
import random
from langchain.llms import OpenAI
from langchain.chains import SequentialChain, LLMChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
import pinecone
import topics_helper

# Function to initialize Pinecone
def initialize_pinecone():
    index_name = 'langchain-retrieval-agent-002'
    pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment="asia-southeast1-gcp-free")

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric='cosine', dimension=1536)

    index = pinecone.Index(index_name)
    return index

# Function to create the agent
def create_agent(index):
    model_name = 'text-embedding-ada-002'
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    embed = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)
    vectorstore = Pinecone(index, embed, "text")

    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0.2)

    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history', input_key='input', output_key='output', k=5, return_messages=True)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    tools = [Tool(name='Socratic Knowledge Base', func=qa.run, description='...')]

    system_message = """
            You are an advanced AI assistant specialized in generating philosophical multiple-choice questions. Your primary task is to create engaging, thought-provoking questions based on various philosophical topics. Each question you generate should be followed by a set of possible answers, labeled as Option 1, Option 2, etc. 

            Key Guidelines:
            1. Ensure the questions are unique and not repetitive.
            2. Do not provide the correct answer within the options; maintain neutrality.
            3. Tailor each question to be specific to the philosophical topic provided.
            4. Maintain a formal and respectful tone suitable for educational purposes.
            5. The questions should encourage critical thinking and philosophical discussion.
            6. Keep your responses concise and focused on the question generation task.
            7. Adhere to a consistent format: a clear question followed by distinct options.

            Remember, the goal is to stimulate intellectual curiosity and facilitate a deeper understanding of philosophical concepts through these questions. Your role is crucial in creating an engaging and educational experience for the users.
            """

    return initialize_agent(agent='chat-conversational-react-description', tools=tools, llm=llm,
                            verbose=True, max_iterations=3, early_stopping_method='generate',
                            memory=conversational_memory, return_intermediate_steps=True,
                            handle_parsing_errors=True, agent_kwargs={"system_message": system_message})

# Function to generate questions
def generate_questions(agent, philosophical_topics):
    mcq_list = {}
    max_retries = 3

    for _ in range(3):
        current_topic = random.choice(philosophical_topics)
        prompt = f"""
        Your task is to generate a multiple-choice question about '{current_topic}'. 
        Please format the question and answers as follows:
        Start with the question text followed by a newline.
        Then list the answer options, each on a new line, starting with 'Option 1:', 'Option 2:', etc.
        For example:
        'What is the main theme of {current_topic}?\\n\\nOption 1: Theme A\\nOption 2: Theme B\\nOption 3: Theme C...'
        Ensure the question is unique and thought-provoking, suitable for philosophical discussions.
        """

        for attempt in range(max_retries):
            response = agent({"input": prompt})
            try: 
                question_and_answers = response['output']  
                question, answers = parse_question_and_answers(question_and_answers)
                mcq_list[question] = answers
                break  # Break out of the retry loop if successful
            except ValueError as e:
                if attempt == max_retries - 1:
                    st.warning(f"Unable to generate a properly formatted question after {max_retries} attempts.")
                # If not the last attempt, it will automatically retry

    return mcq_list

# Function to parse question and answers
def parse_question_and_answers(data):
    try:
        # Splitting into question and answers parts
        parts = data.split('\n\n', 1)
        if len(parts) < 2:
            raise ValueError("Question and answers format not as expected")

        question_part = parts[0]
        answers_part = parts[1] if len(parts) > 1 else ''

        # Splitting the answers part into a list of answers
        answers = answers_part.strip().split('\n')

        # Clean up each answer to remove the option prefix
        cleaned_answers = []
        for answer in answers:
            # Split only on the first occurrence of ': '
            split_answer = answer.split(': ', 1)
            if len(split_answer) == 2:
                cleaned_answers.append(split_answer[1])
            else:
                # If the answer does not match expected format, skip it
                continue

        return question_part, cleaned_answers
    except Exception as e:
        raise ValueError(f"Error parsing question and answers: {e}")

# Function to display questions in the Streamlit app
def display_questions(mcq_list, form_key):
    with st.form(key=f'question_form_{form_key}'):
        chosen_answers = {}
        for i, (question, answers) in enumerate(mcq_list.items()):
            st.write(question)
            answers_with_empty_option = [""] + answers
            chosen_answer = st.radio("Select one of the below choices:", answers_with_empty_option, key=f'question_{i}_{form_key}')
            chosen_answers[question] = chosen_answer

        submitted = st.form_submit_button("Submit Answers")

        if submitted:
            if all(answer != "" for answer in chosen_answers.values()):
                st.success("You've successfully submitted your answers!")
                st.session_state['assessment_completed'] = True
            else:
                st.error("Please select an answer for each question before submitting.")


# Initialize session state for assessment status
if 'assessment_started' not in st.session_state:
    st.session_state['assessment_started'] = False
if 'assessment_completed' not in st.session_state:
    st.session_state['assessment_completed'] = False
if 'form_key' not in st.session_state:
    st.session_state['form_key'] = 0
if 'agent_initialized' not in st.session_state:
    st.session_state['agent_initialized'] = False
    st.session_state['agent'] = None

# Function to initialize the agent
def initialize_agent_if_needed():
    if not st.session_state['agent_initialized']:
        index = initialize_pinecone()
        st.session_state['agent'] = create_agent(index)
        st.session_state['agent_initialized'] = True

# Main execution logic
if __name__ == "__main__":
    st.title("🧠 Socratic Assessment 💭")
    st.caption("Test your philosophical prowess with 'Socratic Assessment' 🤖🎓, a Gen AI app designed to evaluate your knowledge in various philosophy topics, offering a deep dive into the realms of wisdom and understanding! 📚💭")

    # Check for necessary API keys
    if "OPENAI_API_KEY" not in st.secrets or "PINECONE_API_KEY" not in st.secrets:
        st.error("API keys are missing. Please add them to your Streamlit secrets.")
    else:
        initialize_agent_if_needed()

        # Button to start or restart the assessment
        if st.button("Start New Assessment"):
            st.session_state['assessment_started'] = True
            st.session_state['assessment_completed'] = False
            st.session_state['form_key'] += 1  # Increment form key to ensure uniqueness
            mcq_list = generate_questions(st.session_state['agent'], topics_helper.philosophical_topics)
            display_questions(mcq_list, st.session_state['form_key'])

        elif st.session_state['assessment_started'] and not st.session_state['assessment_completed']:
            # If an assessment has started but not yet completed, continue displaying questions
            mcq_list = generate_questions(st.session_state['agent'], topics_helper.philosophical_topics)
            display_questions(mcq_list, st.session_state['form_key'])

        if st.session_state['assessment_completed']:
            # Display the humorous message
            st.markdown("### 🧙‍♂️✨ Keep going, future philosopher! Your insights today have paved the way for profound philosophical journeys ahead. Who knows? You might just be the next Socrates! 🌟")



mystyle = '''
    <style>
        div[role="radiogroup"] >  :first-child{
           display: none !important;
        }       
    </style>
    '''

st.markdown(mystyle, unsafe_allow_html=True)
