from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import pandas as pd
from datasets import Dataset
from huggingface_hub import login

login()


def convert_to_hf_dataset(example):
    return {"text": example["text"]}


hf_dataset = Dataset.from_pandas(pd.read_csv("./data.csv"))
hf_dataset = hf_dataset.map(convert_to_hf_dataset)

model = AutoModelForCausalLM.from_pretrained(
    "ura-hcmut/ura-llama-7b", device_map="auto"
)
model.config.pretraining_tp = 1
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "ura-hcmut/ura-llama-7b", trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

dataset = hf_dataset

training_args = TrainingArguments(
    "output_dir",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="logs",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    packing=True,
    args=training_args,
)

trainer.train()
