from trl import RewardTrainer
from peft import LoraConfig
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from datasets import load_dataset
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("ura-hcmut/GemSUra-7B")

train_dataset = load_dataset("lctzz540/bunbo-reward-dataset", split="train")


def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_j = tokenizer(chosen, truncation=True)
        tokenized_k = tokenizer(rejected, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(
            tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(
            tokenized_k["attention_mask"])

    return new_examples


train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= 512 and len(
        x["input_ids_rejected"]) <= 512
)

quantization_config = BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "ura-hcmut/GemSUra-7B",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    num_labels=1,
)
model.config.use_cache = False

training_args = TrainingArguments(
    output_dir="./train_logs",
    per_device_train_batch_size=128,
    gradient_accumulation_steps=1,
    learning_rate=1.41e-5,
    optim="adamw_torch",
    save_steps=50,
    logging_steps=50,
    report_to="tensorboard",
    remove_unused_columns=False,
)

peft_config = LoraConfig(
    r=16, lora_alpha=16, bias="none", task_type="SEQ_CLS", modules_to_save=["scores"]
)

trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_length=512,
)

trainer.train()
trainer.model.save_pretrained("./reward_model")
