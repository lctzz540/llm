import csv
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("lctzz540/bunbo")

query_data = dataset["train"]["query"][:10000]

llm = LLM(model="llamappo", enable_lora=True, gpu_memory_utilization=0.8)

# Define sampling parameters
sampling_params_1 = SamplingParams(temperature=1, max_tokens=256, top_p=0.8)
sampling_params_2 = SamplingParams(temperature=0.9, max_tokens=256, top_p=0.95)

def generate_response(query):
    prompt = f"[INST] <<SYS>>\nBạn là công cụ tạo nội dung cho bài viết rao bán facebook. Hãy tạo nội dung bài viết dựa vào những thông tin sau.\nCâu hỏi: {query}\nTrả lời: [/INST]"
    
    outputs_1 = llm.generate([prompt], sampling_params_1)
    outputs_2 = llm.generate([prompt], sampling_params_2)
    
    return [outputs_1[0].outputs[0].text, outputs_2[0].outputs[0].text]

with open("responses.csv", "a", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["request", "response_1", "response_2"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    if csvfile.tell() == 0:
        writer.writeheader()

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(generate_response, query) for query in query_data]
        for query, future in zip(query_data, tqdm(as_completed(futures), total=len(futures), desc="Generating responses")):
            response_set = future.result()
            writer.writerow({
                "request": query,
                "response_1": response_set[0],
                "response_2": response_set[1],
            })
