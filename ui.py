import ast
import os
import re
import openai
import streamlit as st
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

math_model_url = os.environ.get("MATH_MODEL", "http://localhost:8708")

stt_model = WhisperModel("medium.en", device="cpu", compute_type="int8")
math_model = openai.OpenAI(base_url=math_model_url, api_key="sk-no-key-required")
classifier_model = pipeline(
    "zero-shot-classification", model="tasksource/deberta-base-long-nli"
)
query_labels = ["order transaction"]

st.set_page_config(page_title="Saia | The Solopreneur's AI Assistant")
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

    return transcript


def call_llm(messages, client, temperature=0.7):
    res = client.chat.completions.create(
        model="local-llm",
        messages=messages,
        max_tokens=4096,
        temperature=temperature,
    )

    return res.choices[0].message.content


def get_last_assigned_variable_name_and_value(code):
    # Parse the code into an AST
    tree = ast.parse(code)

    # Traverse the AST to find the last assignment statement
    last_assignment = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            last_assignment = node

    # Extract the name of the last assigned variable
    if last_assignment:
        last_var_name = last_assignment.targets[0].id
    else:
        return None, None

    # Execute the code to create the variables in the local namespace
    local_vars = {}
    exec(code, {}, local_vars)

    # Retrieve the value of the last assigned variable
    last_var_value = local_vars.get(last_var_name)

    return last_var_name, last_var_value


def extract_python_code(response_text: str):
    """
    Extract and remove text within ````output ... ``` markers, while keeping the text within ````python ... ``` markers.

    :param response_text: The string containing the code and output blocks.
    :return: A tuple containing the extracted Python code and the modified string with the output block removed.
    """
    extracted_python = []
    modified_response_text = []
    in_python_block = False
    in_output_block = False
    last_boxed_sentence = None

    for line in response_text.split("\n"):
        if line.strip() == "```python":
            in_python_block = True
            extracted_python = []  # Reset the list to only keep the last Python block
        elif line.strip() == "```output":
            in_output_block = True
        elif line.strip() == "```":
            in_python_block = False
            in_output_block = False
        elif in_python_block:
            extracted_python.append(line)
        elif not in_output_block:
            if "\\boxed{" in line:
                last_boxed_sentence = line
            else:
                modified_response_text.append(line)

    extracted_python_text = "\n".join(extracted_python)
    modified_response_text = "\n".join(modified_response_text)

    return extracted_python_text, modified_response_text, last_boxed_sentence


def compute_by_llm_tir(transcript):
    messages = [
        {
            "role": "system",
            "content": "You will assist in computing sales transactions in pesos as the currency or monetary unit. Please integrate natural language reasoning with programs to solve the problems, and put your final answer within \\boxed{}.",
        },
        {
            "role": "user",
            "content": f"{transcript}\n\nImportant: When writing the python code, ALWAYS store the final answer in the 'answer' variable.",
        },
    ]

    res = math_model.chat.completions.create(
        model="local-llm",
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
    )

    return res


def entry():
    st.subheader("Saia - The Solopreneur's AI Assistant")
    with st.expander("Read the instructions"):
        st.markdown(
            """
            ***What is this for?*** This AI Assistant will compute how much the customer will pay based on their order.

            ***How to use this?*** Simply talk to the Assistant by saying the customer's order. You can say: "The customer ordered a Spanish Latte at 110 pesos with a 5% discount, an Americano at 100 pesos without a discount, and a Strawberry Matcha at 105 pesos with a 20% discount. How much will the customer pay?"

            ***Now what?*** To get started, just click the Microphone icon below to start recording, say the orders, then click the stop button when you're finished saying them. Don't forget to ask "How much will the customer pay?" at the end.
            """
        )

    audio_value = st.experimental_audio_input("Record audio")

    if audio_value:
        transcript = transcribe_transaction(audio_value)
        st.markdown(f"##### Transaction transcript:\n\n{transcript}")

        output = classifier_model(transcript, query_labels)
        if output["scores"][0] > 0.8:
            with st.spinner(
                "Understanding the problem, then outlining and executing a step-by-step plan..."
            ):
                response = compute_by_llm_tir(transcript)

                python_code, modified_response, last_boxed_sentence = (
                    extract_python_code(response.choices[0].message.content)
                )

                st.markdown(f"##### Transaction query solution:")

                with st.expander("Read the step-by-step plan"):
                    st.markdown(modified_response)

                with st.expander("View the code implementation"):
                    st.markdown(f"""```python\n{python_code}""")

                _, last_var_value = get_last_assigned_variable_name_and_value(
                    python_code
                )

                pattern = r"\\boxed\{([^\}]+)\}"
                new_text = re.sub(
                    pattern, f"{last_var_value}", last_boxed_sentence, count=1
                )
                st.markdown(f"> {new_text}")

        else:
            st.markdown("Your query is not an order transaction.")


if __name__ == "__main__":
    entry()
