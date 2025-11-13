from transformers import pipeline
from automap.utils import setup_hf
from huggingface_hub import whoami
import torch
import os


setup_hf()

model_name = "google/gemma-3-12b-it"  # Using Gemma 2 12B Instruct from Hugging Face
ontology_path = f"{os.getenv('HOME')}/workspace/automap/datasets/blinkg/data/scenario1/ontology.ttl"
csv_path = f"{os.getenv('HOME')}/workspace/automap/datasets/blinkg/data/scenario1/1B/student.csv"

# Create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

with open(ontology_path, 'r') as f:
    ontology_content = f.read()

with open(csv_path, 'r') as f:
    csv_content = f.read()

prompt_content = f"""You task is to generatea a YARRRML mapping given a CSV file and an RDF schema ontology. You must write the YARRRML mapping in turtle language. Answer only with the mapping, I do not want anything else. The source csv file name is student.csv.

csv file:
{csv_content}

ontology:
{ontology_content}
"""

# Format prompt for chat model
messages = [
    {
        'role': 'user',
        'content': prompt_content,
    },
]

# Generate response using pipeline
response = pipe(
    messages,
    # max_new_tokens=2048,
    # temperature=0.0,
    do_sample=False,
    top_p=0.0
)

# Extract and print the generated text
print(response[0]['generated_text'][-1]['content'])
