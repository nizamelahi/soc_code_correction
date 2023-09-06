from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import json


with open('data.json') as json_file:
    indata = json.load(json_file)




time1=datetime.now()
tokenizer = AutoTokenizer.from_pretrained("HyperbeeAI/Tulpar-7b-v0")
model = AutoModelForCausalLM.from_pretrained("HyperbeeAI/Tulpar-7b-v0")
model = model.to("cuda")

outdata=[]
for x in range(10):
    
    input_text=f"In as few words as possible ,please provide SOC code for the job title: \"{indata[x]['job_title']}\", and put it in a json along with the SOC title "
    prompt = f"### User: {input_text}\n\n### Assistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=512)
    print(tokenizer.decode(output[0]))
    out=tokenizer.decode(output[0])
    indata[x]['new_soc_code']=out.split("{")[1].split(":")[1].split(",")[0].replace("\"","")
    indata[x]['new_soc_title']=out.split("{")[1].split(",")[1].split(":")[1].split("}")[0].replace("\"","")
    print(f"time taken: {datetime.now()-time1}")

    
    outdata.append(indata[x])
