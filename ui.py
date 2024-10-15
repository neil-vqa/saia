import os
import openai
import streamlit as st
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

math_model_url = os.environ.get("MATH_MODEL", "http://localhost:8707")
writer_model_url = os.environ.get("WRITER_MODEL", "http://localhost:8704")

stt_model = WhisperModel("medium.en", device="cpu", compute_type="int8")
math_model = openai.OpenAI(base_url=math_model_url, api_key="sk-no-key-required")
writer_model = openai.OpenAI(base_url=writer_model_url, api_key="sk-no-key-required")


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

        st.markdown(f"Transaction transcript:\n\n{transcript}")

        with st.spinner("Computing..."):
            msg_compute = [
                {
                    "role": "system",
                    "content": "You will assist in computing sales transactions. Reason step by step, and put your final answer within \\boxed{}.",
                },
                {
                    "role": "user",
                    "content": transcript,
                },
            ]
            math_res = call_llm(msg_compute, math_model)

            msg_edit = [
                {
                    "role": "system",
                    "content": "You will edit a transaction record to use the appropriate currency.",
                },
                {
                    "role": "user",
                    "content": f"Text:\n\n{math_res}\n\n\nTask: Identify the currency or unit being used in the text, then replace it to use Philippine Pesos as the currency or unit. Do not explain how, why, or what you edited. Do not paraphrase the text.",
                },
            ]
            writer_res = call_llm(msg_edit, writer_model, 0.4)

        st.markdown(f"Transaction computed:\n\n{writer_res}")


if __name__ == "__main__":
    entry()
