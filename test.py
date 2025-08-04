from vllm import LLM
from vllm.sampling_params import SamplingParams
from datetime import datetime, timedelta

SYSTEM_PROMPT = "You are a conversational agent that always answers straight to the point, always end your accurate response with an ASCII drawing of a cat."

user_prompt = "How to plan a trip to Manali?"

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": user_prompt
    },
]
model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
# note that running this model on GPU requires over 60 GB of GPU RAM
llm = LLM(model=model_name, tokenizer=model_name, tensor_parallel_size=4)

sampling_params = SamplingParams(max_tokens=512, temperature=0.0)
outputs = llm.chat(messages, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)