# saia

![Saia](/assets/saia-ss.png)

## Models

These are the models I run locally for the whole app to function.

- STT: openai/whisper-medium.en
- Math: Qwen/Qwen2.5-Math-7B-Instruct
- General info extractor (IE): Goekdeniz-Guelmez/Josiefied-Qwen2.5-1.5B-Instruct-abliterated-v1
- NER: numind/NuNER_Zero
- Classifier: tasksource/deberta-base-long-nli

## Running locally

Both Math and General IE models are served by llama.cpp. Default are ports 8706 (General IE) and 8708 (Math). They can be configured by setting IE_MODEL and MATH_MODEL environment variables (e.g. IE_MODEL=http://localhost:8706). All other models are downloaded (at initial run) from huggingface.

The easiest way to get up and running is by Docker. The instructions.txt provides quick copy-pastable commands. The app will be available on *http://localhost:8501*

Note: When running, the app consumes ~5GB RAM. Hahaha. I appreciate your suggestions on how to improve this.

## Todos

- [ ] Setup automated tests
- [ ] Remove hard-coded products price list. (Currently, they are coffee shop-specific products. They let the user to simply say like, "Strawberry Matcha large", and the app will get the corresponding price for that. Will possibly create a UI to set the ProductName-Variant-Price data.)
- [ ] Save the current transaction: the transcript, invoice
- [ ] Have a transactions history
