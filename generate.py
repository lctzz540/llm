import csv
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("lctzz540/bunbo")

query_data = dataset["train"]["query"][:10000]
llm = LLM(model="ura-hcmut/GemSUra-7B",
          enable_lora=True, gpu_memory_utilization=0.98)

sampling_params = SamplingParams(
    temperature=0.8, max_tokens=256, stop=["[/assistant]"])


def generate_response(query):
    prompts = [
        f"[user] {query} [/user] [assistant]",
        f"[user] {query} [/user] [assistant]",
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("text_adapter", 1, "./checkpoint-11000/"),
    )
    return [output.outputs[0].text for output in outputs]


with open("responses.csv", "a", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["request", "response_1", "response_2"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    if csvfile.tell() == 0:
        writer.writeheader()

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(generate_response, query)
                   for query in query_data]
        for query, future in zip(query_data, tqdm(
            as_completed(futures), total=len(futures), desc="Generating responses"
        )):
            response_set = future.result()
            writer.writerow(
                {
                    "request": query,
                    "response_1": response_set[0],
                    "response_2": response_set[1],
                }
            )
