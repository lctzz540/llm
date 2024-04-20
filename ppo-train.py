import torch
from tqdm import tqdm
from transformers import pipeline
from trl import PPOConfig
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer


dataset = load_dataset("lctzz540/bunbo", split="train")
tokenizer = AutoTokenizer.from_pretrained("ura-hcmut/GemSUra-7B")

config = PPOConfig(
    model_name="lctzz540/bunbomerged",
    learning_rate=1.41e-5,
    batch_size=1,
    mini_batch_size=1,
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token

reward_model = pipeline("text-classification", model="lctzz540/bunbo-reward")


def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample


dataset = dataset.map(tokenize, batched=False)
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
)
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}


epochs = 10
for epoch in tqdm(range(epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(
            r.squeeze()) for r in response_tensors]

        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

ppo_trainer.save_model("ppo_model")
