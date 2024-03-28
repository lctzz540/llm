from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
import pandas as pd
from datasets import Dataset
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, get_peft_model


model_id = "ura-hcmut/GemSUra-7B"


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['query'])):
        text = f"### Question: {example['query'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts


hf_dataset = load_dataset("bunbo", split="train")

response_template = "### Answer:"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)




model.config.pretraining_tp = 1
model.eval()


tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

collator = DataCollatorForCompletionOnlyLM(
    response_template, tokenizer=tokenizer)
dataset = hf_dataset

training_args = TrainingArguments(
    "output_dir",
    per_device_train_batch_size=32,
    num_train_epochs=1,
    logging_dir="logs",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)


trainer.train()
