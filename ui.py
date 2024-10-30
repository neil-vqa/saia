import ast
import os
import difflib
import re
import openai
import streamlit as st
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from transformers import pipeline
from gliner import GLiNER

load_dotenv()

ie_model_url = os.environ.get("IE_MODEL", "http://localhost:8706")
math_model_url = os.environ.get("MATH_MODEL", "http://localhost:8708")

stt_model = WhisperModel("small.en", device="cpu", compute_type="int8")
math_model = openai.OpenAI(base_url=math_model_url, api_key="sk-no-key-required")
ie_model = openai.OpenAI(base_url=ie_model_url, api_key="sk-no-key-required")
ner_model = GLiNER.from_pretrained("numind/NuNerZero")
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

    return res.choices[0].message.content


def solution_pipeline(transcript, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = compute_by_llm_tir(transcript)

            python_code, modified_response, last_boxed_sentence = extract_python_code(
                response
            )

            _, last_var_value = get_last_assigned_variable_name_and_value(python_code)

            return python_code, modified_response, last_boxed_sentence, last_var_value
        except SyntaxError:
            if attempt < max_retries - 1:
                continue
            else:
                raise


def write_invoice(transcript, new_text):
    schema = {
        "items": [{"name": "", "price": "", "quantity": "", "discount": ""}],
        "total": "",
    }
    invoice_ie_prompt = [
        {
            "role": "system",
            "content": "You will write an invoice from an order transcript. You will output in JSON format strictly according to the schema specified.",
        },
        {
            "role": "user",
            "content": "Order transcript: {transcript}\n{new_text}\n\nTask: Extract the items ordered, their prices, quantities, and if any discounts, as well as the total amount to pay. Strictly follow this JSON schema: {schema}".format(
                transcript=transcript, new_text=new_text, schema=schema
            ),
        },
    ]

    res = ie_model.chat.completions.create(
        model="local-llm",
        messages=invoice_ie_prompt,
        max_tokens=4096,
        temperature=0.7,
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "price": {"type": "number"},
                                "quantity": {"type": "number"},
                                "discount": {"type": "number"},
                            },
                            "required": ["name", "price", "quantity", "discount"],
                        },
                    },
                    "total": {"type": "number"},
                },
                "required": ["items", "total"],
            },
        },
    )
    return res.choices[0].message.content


def merge_entities(entities):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity["entity"] == current["entity"] and (
            next_entity["start"] == current["end"] + 1
            or next_entity["start"] == current["end"]
        ):
            current["word"] += " " + next_entity["word"]
            current["end"] = next_entity["end"]
        else:
            merged.append(current)
            current = next_entity
    merged.append(current)
    return merged


def extract_ordered_items(transcript):
    labels = ["ordered_item"]

    r = {
        "text": transcript,
        "entities": [
            {
                "entity": entity["label"],
                "word": entity["text"],
                "start": entity["start"],
                "end": entity["end"],
                "score": 0,
            }
            for entity in ner_model.predict_entities(transcript, labels, threshold=0.3)
        ],
    }
    r["entities"] = merge_entities(r["entities"])

    items = ""
    for item in r["entities"]:
        items = f"{item['word']}, {items}"
    return items


def get_variants(ordered_items: str):
    ordered_items = ordered_items.lower()
    variant_pattern = re.compile(r"\b(large|medium|small)\b")

    items_witn_variants = []
    for item in ordered_items.split(",")[:-1]:
        item = item.strip()
        match = variant_pattern.search(item)
        if match:
            size = match.group(0)
            name = item.replace(size, "").strip()
            items_witn_variants.append((name, size))

    return items_witn_variants


def find_closest_match(input_str, choices):
    matches = difflib.get_close_matches(input_str, choices, n=1, cutoff=0.6)
    return matches[0] if matches else None


def get_price(product_name, variant, products, variant_class="size"):
    names = {product["name"] for product in products}
    variants = {product[variant_class] for product in products}

    # Find the closest matches
    closest_product_name = find_closest_match(product_name, names)
    closest_variant = find_closest_match(variant, variants) if variant else None

    if closest_product_name and closest_variant:
        # Filter the products to get the price
        for product in products:
            if (
                product["name"] == closest_product_name
                and product[variant_class] == closest_variant
            ):
                return product["price"]
    elif closest_product_name:
        # If variant is not provided, find the first match with the closest product_name
        for product in products:
            if product["name"] == closest_product_name:
                return product["price"]
    return None


products = [
    {"name": "Americano", "size": "large", "price": 90},
    {"name": "Americano", "size": "medium", "price": 80},
    {"name": "Americano", "size": "small", "price": 50},
    {"name": "Vanilla Latte", "size": "large", "price": 110},
    {"name": "Vanilla Latte", "size": "medium", "price": 100},
    {"name": "Vanilla Latte", "size": "small", "price": 50},
    {"name": "Strawberry Matcha", "size": "large", "price": 130},
    {"name": "Strawberry Matcha", "size": "medium", "price": 120},
    {"name": "Strawberry Matcha", "size": "small", "price": 50},
]


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
                "Understanding the problem, then outlining and executing a step-by-step plan to solve it..."
            ):
                items = extract_ordered_items(transcript)
                items_with_variants = get_variants(items)
                price_ref = ""
                for name, variant in items_with_variants:
                    price = get_price(name, variant, products)
                    price_ref = f"{name} {variant}: {price} pesos\n{price_ref}"

                transcript = f"{transcript}\n\n{price_ref}"
                python_code, modified_response, last_boxed_sentence, last_var_value = (
                    solution_pipeline(transcript=transcript)
                )

                st.markdown(f"##### Transaction query solution:")

                pattern = r"\\boxed\{([^\}]+)\}"
                new_text = re.sub(
                    pattern, f"{last_var_value}", last_boxed_sentence, count=1
                )
                st.markdown(f"> {new_text}")

                with st.expander("Read the step-by-step plan"):
                    st.markdown(modified_response)

                with st.expander("View the code implementation"):
                    st.markdown(f"""```python\n{python_code}""")

            with st.spinner("Writing invoice..."):
                res = write_invoice(transcript=transcript, new_text=new_text)
                with st.expander("View invoice"):
                    st.markdown(f"```json\n\n{res}")

        else:
            st.markdown("> Your query is not an order transaction.")


if __name__ == "__main__":
    entry()
