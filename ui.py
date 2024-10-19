import os
import openai
import streamlit as st
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

math_model_url = os.environ.get("MATH_MODEL", "http://localhost:8708")
coder_model_url = os.environ.get("CODER_MODEL", "http://localhost:8706")

stt_model = WhisperModel("medium.en", device="cpu", compute_type="int8")
math_model = openai.OpenAI(base_url=math_model_url, api_key="sk-no-key-required")
coder_model = openai.OpenAI(base_url=coder_model_url, api_key="sk-no-key-required")
classifier_model = pipeline(
    "zero-shot-classification", model="tasksource/deberta-base-long-nli"
)
query_labels = ["order transaction"]


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


def compute_by_symbolic(transcript):
    with st.spinner("Writing equation and solving..."):
        msg_compute = [
            {
                "role": "system",
                "content": "You will write python code that are sympy-compatible algebraic expressions to solve the problems. The code should be a function, and store the answer in the 'answer' variable",
            },
            {
                "role": "user",
                "content": f"Write an algebraic expression to solve this problem:\n\n\n{transcript}\n\n\nStrictly no explanations outside the code block.",
            },
        ]
        math_res = call_llm(msg_compute, coder_model)

    st.markdown(f"#### Symbolic computation:\n\n{math_res}")
    return math_res


def compute_by_llm(transcript):
    with st.spinner("Understanding the problem and computing..."):
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

    st.markdown(f"#### LLM-based computation:\n\n{math_res}")
    return math_res


def verify_by_symbolic(problem_transcript, prelim_solution):
    with st.spinner("Verifying..."):
        msg_compute = [
            {
                "role": "system",
                "content": "You will write python code that are sympy-compatible algebraic expressions to verify solutions to problems. The code should be a function, and store the answer in the 'answer' variable",
            },
            {
                "role": "user",
                "content": f"The problem:\n\n\n{problem_transcript}\n\n\nPreliminary solution:\n\n\n{prelim_solution}\n\n\nWrite an algebraic expression to solve this problem and verify the preliminary solution. Strictly no explanations outside the code block.",
            },
        ]
        math_res = call_llm(msg_compute, coder_model)

    st.markdown(f"#### Symbolic verification:\n\n{math_res}")
    return math_res


def execute_code(expression):
    e = expression.replace("```", "")
    code = e.replace("python", "")
    exec(code, globals())
    st.markdown(f"Answer: {answer}")


def entry():
    audio_value = st.experimental_audio_input("Record audio")

    if audio_value:
        st.audio(audio_value)

        transcript = transcribe_transaction(audio_value)

        output = classifier_model(transcript, query_labels)
        if output["scores"][0] > 0.8:
            prelim = compute_by_llm(transcript)
            expression = verify_by_symbolic(transcript, prelim)
            execute_code(expression)

        else:
            st.markdown("Your query is not an order transaction.")


if __name__ == "__main__":
    entry()
