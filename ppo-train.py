import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from trl import PPOConfig
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOTrainer, create_reference_model
from peft import LoraConfig, TaskType, get_peft_model
from trl.core import LengthSampler

device = "cuda"
output_length_sampler = LengthSampler(100, 800)
DEFAULT_REJECTED_SUMMARY_TEXT = "This is a bad summary"


def score_summaries(model, tokenizer, chosen_summary, rejected_summary):
    chosen_tokens = tokenizer(
        chosen_summary,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    rejected_tokens = tokenizer(
        rejected_summary,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    chosen_tokens.to(device)
    rejected_tokens.to(device)

    with torch.no_grad():
        chosen_logits = model(**chosen_tokens).logits
        rejected_logits = model(**rejected_tokens).logits

    chosen_probs = F.softmax(chosen_logits, dim=-1)
    rejected_probs = F.softmax(rejected_logits, dim=-1)

    chosen_score = chosen_probs[0][1].item()
    rejected_score = rejected_probs[0][1].item()

    chosen_logit = chosen_logits[0][1].item()
    rejected_logit = rejected_logits[0][1].item()

    return chosen_score, rejected_score, chosen_logit, rejected_logit


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def peft_model_ppo(policy_model_path):
    policy_model = T5ForConditionalGeneration.from_pretrained(policy_model_path, device_map="auto")
    policy_model.to(device)
    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["q", "v"],
        lora_dropout=0.10,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    policy_peft_model = get_peft_model(policy_model, lora_config)
    policy_peft_model.to(device)
    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        policy_peft_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        is_trainable=True,
    )
    ppo_model.to(device)
    ref_model = create_reference_model(policy_model)
    ref_model.to(device)
    return ppo_model, ref_model


dataset = load_dataset("lctzz540/bunbo", split="train")
tokenizer = AutoTokenizer.from_pretrained("ura-hcmut/GemSUra-7B")

config = PPOConfig(
    model_name="lctzz540/bunbomerged",
    learning_rate=1.41e-5,
    batch_size=1,
    mini_batch_size=1,
)
model, ref = peft_model_ppo(config.model_name)

tokenizer.pad_token = tokenizer.eos_token

rm_model = AutoModelForSequenceClassification.from_pretrained(
    "./checkpoint-500", device_map="auto", ignore_mismatched_sizes=True
)


def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample


dataset = dataset.map(tokenize, batched=False)
ppo_trainer = PPOTrainer(
    model=model,
    ref_model=ref,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
max_ppo_steps = 100

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if step >= max_ppo_steps:
        break

    prompt_tensors = batch["input_ids"]

    if isinstance(prompt_tensors, list) and all(
        isinstance(item, list) for item in prompt_tensors
    ):
        lengths = [len(seq) for seq in prompt_tensors]
        unique_lengths = set(lengths)

        if len(unique_lengths) > 1:
            max_length = max(unique_lengths)
            original_prompt_tensors = [
                seq + [0] * (max_length - len(seq)) for seq in prompt_tensors
            ]

        prompt_tensors = [torch.tensor(seq).to(device) for seq in prompt_tensors]

    summary_tensors = []

    for prompt_tensor in prompt_tensors:
        prompt_tensor = torch.tensor(prompt_tensor).to(device)
        max_new_tokens = output_length_sampler()
        generation_kwargs["max_new_tokens"] = max_new_tokens
        summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
        summary_tensors.append(summary.squeeze()[-max_new_tokens:])

    batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

    chosen_summaries = batch["response"]
    rejected_summaries = [DEFAULT_REJECTED_SUMMARY_TEXT] * len(batch["response"])

    reward_tensors = []

    for chosen_summary, rejected_summary in zip(chosen_summaries, rejected_summaries):
        chosen_score, _, _, _ = score_summaries(
            rm_model, tokenizer, chosen_summary, rejected_summary
        )
        reward_tensors.append(torch.tensor(chosen_score))

    stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)

    print(f'objective/kl: {stats["objective/kl"]}')
    print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
    print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
    print("-".join("" for x in range(100)))
ppo_trainer.save_model("ppo_model")
