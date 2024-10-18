import os
import openai
import streamlit as st
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

math_model_url = os.environ.get("MATH_MODEL", "http://localhost:8708")
writer_model_url = os.environ.get("WRITER_MODEL", "http://localhost:8706")

stt_model = WhisperModel("medium.en", device="cpu", compute_type="int8")
math_model = openai.OpenAI(base_url=math_model_url, api_key="sk-no-key-required")
writer_model = openai.OpenAI(base_url=writer_model_url, api_key="sk-no-key-required")
classifier_model = pipeline(
    "zero-shot-classification", model="tasksource/deberta-base-long-nli"
)
query_labels = ["sales transaction"]


st.set_page_config(page_title="Sansa")

st.markdown(
    """
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: show;}
        .stAppDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""",
    unsafe_allow_html=True,
)


def transcribe_transaction(audio_value):
    transcript = ""
    with st.spinner("Transcribing..."):
        segments, _ = stt_model.transcribe(audio_value)
        for s in segments:
            transcript = f"{transcript} {s.text}"

    st.markdown(f"#### Transaction transcript:\n\n{transcript}")
    return transcript


def call_llm(messages, client, temperature=0.7):
    res = client.chat.completions.create(
        model="local-llm",
        messages=messages,
        max_tokens=4096,
        temperature=temperature,
    )

    return res.choices[0].message.content


def parse_transaction(transcript):
    with st.spinner("Writing equation..."):
        msg_compute = [
            {
                "role": "system",
                "content": "You will write python code that are sympy-compatible algebraic expressions to solve the problems. The code should should be a function, and store the answer in the 'answer' variable",
            },
            {
                "role": "user",
                "content": f"Write an algebraic expression to solve this problem:\n\n\n{transcript}\n\n\nOutput should be directly the code block or function. No explanations.",
            },
        ]
        math_res = call_llm(msg_compute, writer_model)

    st.markdown(f"#### Transaction equation:\n\n{math_res}")
    return math_res


def compute_transaction(transcript):
    with st.spinner("Computing..."):
        msg_compute = [
            {
                "role": "system",
                "content": "You will assist in computing sales transactions in pesos as the currency or monetary unit. Reason step by step, then give your final answer within \\boxed{}.",
            },
            {
                "role": "user",
                "content": transcript,
            },
        ]
        math_res = call_llm(msg_compute, math_model)

    st.markdown(f"#### Transaction computed:\n\n{math_res}")


def entry():
    audio_value = st.experimental_audio_input("Record audio")

    if audio_value:
        st.audio(audio_value)

        transcript = transcribe_transaction(audio_value)

        output = classifier_model(transcript, query_labels)
        if output["scores"][0] > 0.8:
            expression = parse_transaction(transcript)

            e = expression.replace("```", "")
            code = e.replace("python", "")
            exec(code, globals())
            st.markdown(f"Answer using python code: {answer}")

            compute_transaction(transcript)

        else:
            st.markdown("Your query is not a sales transaction.")


if __name__ == "__main__":
    entry()
