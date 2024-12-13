# HuggingFace LangChain Integration Project

This project demonstrates the integration of HuggingFace models with the LangChain framework for building applications powered by large language models (LLMs). The code explores text generation, answering questions, and constructing pipelines using HuggingFace models and LangChain's utilities.

## Features

- Use HuggingFace endpoints with LangChain.
- Generate text using custom models from HuggingFace's model hub.
- Perform question-answering tasks with step-by-step reasoning prompts.
- Leverage both CPU and GPU for text generation.

## Requirements

To run this project, you need the following:

- Python 3.7+
- `transformers` library
- `langchain` and `langchain_huggingface`
- A HuggingFace API token

Install the required libraries using pip:

```bash
pip install transformers langchain
```

## Usage

### Setting up HuggingFace API Token

Before using the HuggingFace endpoint, ensure you have a valid HuggingFace API token. Set it up in the environment variables:

```python
from google.colab import userdata
sec_key = userdata.get("HUGGINGFACEHUB_API_TOKEN")
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key
```

### Text Generation Using HuggingFace Endpoint

The code demonstrates how to connect a HuggingFace model via its endpoint:

```python
from langchain_huggingface import HuggingFaceEndpoint

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key)
response = llm.invoke("What is machine learning?")
print(response)
```

### Constructing Chains with LangChain

You can use LangChain to create a question-answering pipeline with step-by-step reasoning:

```python
from langchain import PromptTemplate, LLMChain

template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(llm=llm, prompt=prompt)
question = "Who won the Cricket World Cup in the year 2011?"
response = llm_chain.invoke(question)
print(response)
```

### Using HuggingFace Models Locally

If you prefer to use models locally, load them directly with Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
response = pipe("What is generative AI")
print(response)
```

### GPU Utilization

To enable GPU acceleration, configure the `device` parameter:

```python
from langchain_huggingface import HuggingFacePipeline

gpu_llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    device=-1,  # Use device_map="auto" for the accelerate library.
    pipeline_kwargs={"max_new_tokens": 100},
)
```

## Project Structure

- `main.py`: Contains the code for interacting with HuggingFace models and LangChain.
- `requirements.txt`: Lists the required Python packages.

## Contributing

Feel free to submit issues or pull requests for improvements or new features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

### Example Output

**Question:** What is machine learning?

**Answer:** Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. ...
