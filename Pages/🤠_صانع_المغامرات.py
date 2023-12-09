import streamlit as st
import openai
import re
from googletrans import Translator, LANGUAGES

# Initialize the translator
translator = Translator()

# Function to translate text from Arabic to English
def translate_arabic_to_english(text):
    # Translate only if the text is Arabic
    if 'ar' in LANGUAGES and LANGUAGES['ar'] in text:
        return translator.translate(text, src='ar', dest='en').text
    return text

# Function to get the image URL
def get_image(prompt):
    # Translate the prompt from Arabic to English
    translated_prompt = translate_arabic_to_english(prompt)

    try:
        response = openai.Image.create(prompt=translated_prompt, n=1, size="512x512")
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
if "message_history" not in st.session_state:
    st.session_state.message_history = [{"role": "user", "content": """أنت روبوت لعبة قصة تفاعلية تقترح بعض المواقف الخيالية الافتراضية حيث يحتاج المستخدم إلى الاختيار من بين 2-4 Options تقدمها. بمجرد أن يختار المستخدم أحد هذه الخيارات، ستذكر بعد ذلك ما سيحدث بعد ذلك وتقدم خيارات جديدة، ثم يتكرر ذلك. إذا فهمت، فقل، حسنًا، وابدأ عندما أقول "ابدأ". عندما تقدم القصة والخيارات، اعرض القصة فقط وابدأ فورًا بالقصة، دون مزيد من التعليق، ثم خيارات مثل "Option 1:" "Option 2:" ...إلخ. وستقوم بتأليف قصص باللغة العربية للأطفال على هذا المنوال. اللغة العربية هي اللغة الأساسية لك ولا يمكن تأليف القصة بلغات أخرى"""},
                                        {"role": "assistant", "content": """حسنا، أنا أفهم. ابدأ عندما تكون مستعدًا"""}]

# Title of the app
st.title("🤠 صانع المغامرات والحكايات 📚🐉")
st.caption("انطلق في رحلة خيالية مع 'صانع المغامرات والحكايات' 🌟✨، تطبيق ساحر للأطفال يستخدم الذكاء الاصطناعي لخلق قصص مغامرات شيقة وشخصية، لإثارة الخيال وإحياء عوالم الأحلام! 📚🐉")

# Start the game
if "start_game" not in st.session_state or st.button("ابدأ من جديد"):
    st.session_state.start_game = True
    response, st.session_state.message_history = chat("Begin", st.session_state.message_history)
    text = response.split("Option 1")[0]
    st.session_state.options = re.findall(r"Option \d: (.*)", response)
    st.session_state.options.insert(0, "")

# Display the story text and image
if "start_game" in st.session_state:
    with st.form("adventure_form"):
        text = st.session_state.message_history[-1]["content"].split("Option 1")[0]
        img_url = get_image(text)
        st.image(img_url, use_column_width=True)
        st.write(text)

        # Display radio buttons for options
        selected_option = st.radio("اختر خطوتك التالية", st.session_state.options)

        # Check if an option is selected
        submitted = st.form_submit_button("انطلق")
        if submitted:
            reply_content, st.session_state.message_history = chat(selected_option, st.session_state.message_history)
            st.session_state.options = re.findall(r"Option \d: (.*)", reply_content)
            st.session_state.options.insert(0, "")

mystyle = '''
    <style>
        p {
            text-align: right;
        }
        div {
            text-align: right;
        }
        div .block-container {
            text-align: right;
        }
        div[role="radiogroup"] {
            text-align: right;
        }
        .row-widget{
            text-align: right;
        }

        div[role="radiogroup"] >  :first-child{
           display: none !important;
        }       
    </style>
    '''

st.markdown(mystyle, unsafe_allow_html=True)
