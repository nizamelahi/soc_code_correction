from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

time1=datetime.now()
tokenizer = AutoTokenizer.from_pretrained("HyperbeeAI/Tulpar-7b-v0")
model = AutoModelForCausalLM.from_pretrained("HyperbeeAI/Tulpar-7b-v0")
model = model.to("cuda")

input_text="What is deep learning?"
prompt = f"### User: {input_text}\n\n### Assistant:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=512)
print(tokenizer.decode(output[0]))
print(f"started at{time1}")
print(f"ended at {datetime.now()}")
print(f"time taken: {datetime.now()-time1}")
time1=datetime.now()
output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=512)
print(tokenizer.decode(output[0]))
print(f"started at{time1}")
print(f"ended at {datetime.now()}")
print(f"time taken: {datetime.now()-time1}")