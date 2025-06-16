import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig
import torch


@st.cache_resource
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        device_map="auto",
        torch_dtype="auto"
    )
    model = PeftModel.from_pretrained(base_model, "./rft_model")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    return model, tokenizer

model, tokenizer = load_model()
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


# UI
st.title("ConvertAI")
st.text("Ask me to convert some units - Ex. 'What is 3.2km in miles?'")

prompt = st.chat_input("How may I assist you?")

if prompt:
    with st.chat_message("user"):
        # .md or .write
        st.write(prompt)

    # Dummy system response — replace this with your actual logic
    response = pipe(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9)[0]["generated_text"]
    reply = response[len(prompt):].strip()
    reply = reply.replace("<answer>", "").replace("</answer>", "")

    # Display system response
    with st.chat_message("assistant"):
        # .md or .write
        st.write(reply)
