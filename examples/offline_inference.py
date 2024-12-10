from vllm import LLM, SamplingParams
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

# Sample prompts.
prompts = [
    "Sometimes it happens when",
    "Sometimes it happens when",
    "Sometimes it happens when you",
    "Sometimes it happens when you are",
    "Sometimes it happens when you are in",
    "Sometimes it happens when you are in a",
    "Sometimes it happens when you are in a rush",

]
# Create a sampling params object.
sampling_params = [
    SamplingParams(temperature=0),
    SamplingParams(temperature=0, std_output_tokens=[499, 527, 304, 264, 13270], max_tokens=30),
    SamplingParams(temperature=0, std_output_tokens=[527, 304, 264, 13270], max_tokens=30),
    SamplingParams(temperature=0, std_output_tokens=[304, 264, 13270], max_tokens=30),
    SamplingParams(temperature=0, std_output_tokens=[264, 13270], max_tokens=30),
    SamplingParams(temperature=0, std_output_tokens=[13270], max_tokens=30),
    SamplingParams(temperature=0, std_output_tokens=[], max_tokens=30),
]

# Create an LLM.infer1602

llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_model_len=10000, enable_prefix_caching=True, block_size=64)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
