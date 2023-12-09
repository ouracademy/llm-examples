import streamlit as st
import streamlit.components.v1 as components
from task import mermaid, clean
from langchain_wrapper import invoque_llm, get_model
with st.sidebar:
    sample = st.selectbox(
    'Select the example',
    options = ["flowchart generator"], key="sample")
    model_id = st.selectbox(
    'Select the model',
    options = ['gpt-3.5-turbo'])
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/ouracademy/llm-examples)"

if sample == "flowchart generator":
    st.title("Flowchart generator using GPT")
    st.markdown("**Prompt:**")
    prompt = "Generate a flowchart TD using valid mermaid.js syntax that outlines the process with conditional steps of "
    st.markdown(prompt)
    process = st.selectbox(
        'Process:',
        ('order a pizza', 'build a startup', 'cook a cake'))

    if openai_api_key:
        model = get_model(model_id, apikey=openai_api_key,  config= { "temperature":0} )
        r = invoque_llm(prompt + process, model)
        code = r["text"]
        st.markdown("**GPT response**")
        st.code(code)
        code = clean(code)
        st.markdown("**Postprocesing GPT response**")
        st.code(code)
        html = mermaid(code)
        st.markdown("**Flowchart**")
        components.html(html, height=800, scrolling=True)
    else:
        st.error("You must set the api key")
