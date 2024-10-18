import os
import openai
import streamlit as st
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

math_model_url = os.environ.get("MATH_MODEL", "http://localhost:8708")

stt_model = WhisperModel("medium.en", device="cpu", compute_type="int8")
math_model = openai.OpenAI(base_url=math_model_url, api_key="sk-no-key-required")


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


def call_llm(messages, client, temperature=0.7):
    res = client.chat.completions.create(
        model="local-llm",
        messages=messages,
        max_tokens=4096,
        temperature=temperature,
    )

    return res.choices[0].message.content


def entry():
    audio_value = st.experimental_audio_input("Record audio")

    transcript = ""

    if audio_value:
        st.audio(audio_value)
        with st.spinner("Transcribing..."):
            segments, _ = stt_model.transcribe(audio_value)
            for s in segments:
                transcript = f"{transcript} {s.text}"

        st.markdown(f"#### Transaction transcript:\n\n{transcript}")

        with st.spinner("Computing..."):
            msg_compute = [
                {
                    "role": "system",
                    "content": "You will assist in computing sales transactions in pesos as the currency or monetary unit. Reason step by step, and give a preliminary answer. Review your steps, computation accuracy, and the preliminary answer. Then, give your final answer within \\boxed{}.",
                },
                {
                    "role": "user",
                    "content": transcript,
                },
            ]
            math_res = call_llm(msg_compute, math_model)

        st.markdown(f"#### Transaction computed:\n\n{math_res}")


if __name__ == "__main__":
    entry()
