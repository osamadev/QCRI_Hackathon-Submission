from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage

def convert_message(m):
    if m["role"] == "user":
        return HumanMessage(content=m["content"])
    elif m["role"] == "assistant":
        return AIMessage(content=m["content"])
    elif m["role"] == "system":
        return SystemMessage(content=m["content"])
    else:
        raise ValueError(f"Unknown role {m['role']}")
    

# In your helper.py file

def handle_user_prompt(prompt, socratic_chain, socraticKB_tool, session_state):
    """
    Handles the user's prompt, runs the Socratic chain and knowledge base tool,
    and updates the Streamlit session state.

    Args:
    prompt (str): The user's input message.
    socratic_chain (LLMChain): The Socratic LLM Chain for processing the prompt.
    socraticKB_tool (Tool): The Socratic Knowledge Base tool.
    session_state (SessionState): The Streamlit session state for storing messages.
    """
    # Append user prompt to the message history
    session_state.messages.append({"role": "user", "content": prompt})

    # Run the Socratic Knowledge Base tool
    kb_results = socraticKB_tool.run(prompt)

    # Run the Socratic chain
    response = socratic_chain.run(user_prompt=prompt, socratic_research=kb_results)

    # Append the full response to the message history
    full_response = response
    session_state.messages.append({"role": "assistant", "content": full_response})
