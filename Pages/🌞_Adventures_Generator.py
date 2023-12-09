import streamlit as st
import openai
import re

# Function to get the image URL
def get_image(prompt):
    try:
        response = openai.Image.create(prompt=prompt, n=1, size="512x512")
        image_url = response.data[0].url
    except Exception as ex:
        image_url = "https://pythonprogramming.net/static/images/imgfailure.png"
    return image_url

# Function to handle chat
def chat(input, message_history, role="user"):
    message_history.append({"role": role, "content": input})
    completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=message_history)
    response = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": response})
    return response, message_history

# Initialize OpenAI API key
openai.api_key = open('./key.txt', 'r').read().strip("\n")

# Initialize message history in session state
if "message_history_en" not in st.session_state:
    st.session_state.message_history_en = [{"role": "user", "content": """You are an interactive story game bot that proposes some hypothetical fantastical situation where the user needs to pick from 2-4 options that you provide. Once the user picks one of those options, you will then state what happens next and present new options, and this then repeats. If you understand, say, OK, and begin when I say "begin." When you present the story and options, present just the story and start immediately with the story, no further commentary, and then options like "Option 1:" "Option 2:" ...etc."""},
                        {"role": "assistant", "content": f"""OK, I understand. Begin when you're ready."""}]

# Title of the app
st.title("🚀 Adventures Generator 🌞")
st.caption("🌟✨ Embark on a magical journey with 'Adventures Generator', a child-friendly app that uses Gen AI to weave enchanting and personalized adventure stories 📚🐉, igniting young imaginations and bringing their fantasy worlds to life! 🚀")

# Start the game
if "start_game" not in st.session_state or st.button("Start Over"):
    st.session_state.start_game = True
    response, st.session_state.message_history_en = chat("begin", st.session_state.message_history_en)
    text = response.split("Option 1")[0]
    st.session_state.options_en = re.findall(r"Option \d: (.*)", response)
    st.session_state.options_en.insert(0, "")

# Display the story text and image
if "start_game" in st.session_state:
    with st.form("adventure_form"):
        text = st.session_state.message_history_en[-1]["content"].split("Option 1")[0]
        img_url = get_image(text)
        st.image(img_url, use_column_width=True)
        st.write(text)

        # Display radio buttons for options
        selected_option = st.radio("Choose your next step:", st.session_state.options_en)

        # Check if an option is selected
        submitted = st.form_submit_button(" Go ")
        if submitted:
            reply_content, st.session_state.message_history_en = chat(selected_option, st.session_state.message_history_en)
            st.session_state.options_en = re.findall(r"Option \d: (.*)", reply_content)
            st.session_state.options_en.insert(0, "")

mystyle = '''
    <style>
        div[role="radiogroup"] >  :first-child{
           display: none !important;
        }       
    </style>
    '''

st.markdown(mystyle, unsafe_allow_html=True)
