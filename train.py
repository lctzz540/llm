import os
from transformers.trainer_callback import TrainerCallback
from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, get_peft_model


login()

model_id = "ura-hcmut/ura-llama-7b"


def convert_to_hf_dataset(example):
    return {"text": example["text"]}


hf_dataset = Dataset.from_pandas(pd.read_csv("./data.csv"))
hf_dataset = hf_dataset.map(convert_to_hf_dataset)

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

dataset = hf_dataset

training_args = TrainingArguments(
    "output_dir",
    per_device_train_batch_size=64,
    num_train_epochs=10,
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


class SaveWeightsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model_to_save = state.trainer.model
        output_dir = os.path.join(args.output_dir, f"epoch_{state.epoch}")
        os.makedirs(output_dir, exist_ok=True)
        model_to_save.save_pretrained(output_dir)
        self.save_optimizer_and_scheduler(output_dir, state)

    def save_optimizer_and_scheduler(self, output_dir, state):
        state.trainer.optimizer.save_pretrained(output_dir)
        state.trainer.lr_scheduler.save_pretrained(output_dir)


trainer.add_callback(SaveWeightsCallback())

trainer.train()


trainer.train()
model.save_pretrained("ura-finetuned")
