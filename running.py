# import torch

# def calculate_num_elements(gb, dtype=torch.float32):
#     bytes_per_element = torch.tensor([], dtype=dtype).element_size()
#     num_elements = gb * 1024**3 // bytes_per_element
#     return int(num_elements)


# if __name__ == "__main__":
#     researve_size_gb = 44
#     memory_reserve = torch.empty((calculate_num_elements(researve_size_gb),), dtype=torch.float32, device='cuda')

#     # del memory_reserve




import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "16"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

model_name = "Qwen/Qwen3-32B"
# model_name = "Qwen/Qwen3.6-35B-A3B"
# model_name = "/scratch/cs/project_bil_genai/weiguo/huggingface/Qwen3-32B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)

dataset = load_dataset(
    "TIGER-Lab/MMLU-Pro",
    split="test",
)

n = 0
while True:
    print('Episode {}'.format(n))
    for sample in tqdm(dataset):
        question = sample["question"]
        options = sample["options"]

        prompt = question + "\n"

        for i, option in enumerate(options):
            prompt += f"{chr(65 + i)}. {option}\n"

        prompt += "\nAnswer the question."

        messages = [
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        model_inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # print("thinking content:", thinking_content)
        # print("content:", content)
    n += 1