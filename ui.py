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
                "content": "You will assist in computing sales transactions in pesos as the currency or monetary unit. Please integrate natural language reasoning with programs to solve the problems, and put your final answer within \\boxed{}.",
            },
            {
                "role": "user",
                "content": transcript,
            },
        ]
        math_res = call_llm(msg_compute, math_model)

    st.markdown(f"#### LLM-based computation:\n\n{math_res}")
    return math_res


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
    with st.spinner("Understanding the problem, solving, and verifying..."):
        msg_compute = [
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
            messages=msg_compute,
            max_tokens=4096,
            temperature=0.7,
        )

        python_code, modified_response, last_boxed_sentence = extract_python_code(
            res.choices[0].message.content
        )
        st.markdown(modified_response)
        with st.expander("See code"):
            st.markdown(f"""```python\n{python_code}""")

        _, last_var_value = get_last_assigned_variable_name_and_value(python_code)

        pattern = r"\\boxed\{([^\}]+)\}"
        new_text = re.sub(pattern, f"{last_var_value}", last_boxed_sentence, count=1)
        st.markdown(f"**{new_text}**")


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


def verify(problem_transcript, prelim_solution):
    try:
        expression = verify_by_symbolic(problem_transcript, prelim_solution)
        execute_code(expression)
        st.markdown(f"Answer: {answer}")
    except:
        verify(problem_transcript, prelim_solution)


def entry():
    audio_value = st.experimental_audio_input("Record audio")

    if audio_value:
        st.audio(audio_value)

        # transcript = transcribe_transaction(audio_value)
        transcript = "The customer ordered a Spanish latte at 110 pesos with a 5% discount and also a strawberry matcha at 120 pesos without a discount. How much will the customer pay?"
        st.markdown(f"#### Transaction transcript:\n\n{transcript}")

        output = classifier_model(transcript, query_labels)
        if output["scores"][0] > 0.8:
            # prelim = compute_by_llm(transcript)
            # verify(transcript, prelim)
            compute_by_llm_tir(transcript)

        else:
            st.markdown("Your query is not an order transaction.")


if __name__ == "__main__":
    entry()
