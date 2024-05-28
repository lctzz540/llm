from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
from random import seed
from datasets import load_dataset
import numpy as np
import nltk
import evaluate

# Ensure necessary nltk data is downloaded
nltk.download('punkt')

# Set a seed for reproducibility
seed(42)
np.random.seed(42)

# Load evaluation dataset
eval_dataset = load_dataset("lctzz540/bunbo", split="train")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./ppo-model/")
model = AutoModelForCausalLM.from_pretrained("./ppo-merged/").to("cuda")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

def evaluate(sample):
    query = sample['query']
    
    # Generate output
    outputs = pipe(
        query,
        max_new_tokens=256,
        do_sample=True,  # Correct argument name
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Extract generated text
    generated_text = outputs[0]["generated_text"]
    predicted_answer = generated_text[len(query):].strip()
    
    # Compare with the reference answer
    reference_answer = sample['answer'].strip()
    
    return predicted_answer, reference_answer

# Evaluate the model on a subset of the dataset
number_of_eval_samples = 100
eval_indices = np.random.choice(len(eval_dataset), number_of_eval_samples, replace=False)

# Collect predictions and references
predictions = []
references = []

for idx in tqdm(eval_indices):
    idx = int(idx)
    predicted_answer, reference_answer = evaluate(eval_dataset[idx])
    predictions.append(predicted_answer)
    references.append(reference_answer)

# Calculate BLEU score (using the correct format)
bleu_references = [[ref] for ref in references]
bleu_predictions = predictions
bleu_result = bleu_metric.compute(predictions=bleu_predictions, references=bleu_references)
print(f"BLEU score: {bleu_result['bleu'] * 100:.2f}%")

# Calculate ROUGE score
rouge_result = rouge_metric.compute(predictions=bleu_predictions, references=bleu_references)
print(f"ROUGE scores:\n{rouge_result}")

# Calculate METEOR score
meteor_result = meteor_metric.compute(predictions=bleu_predictions, references=bleu_references)
print(f"METEOR score: {meteor_result['meteor'] * 100:.2f}%")
