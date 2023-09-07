from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import json


with open('data.json') as json_file:
    indata = json.load(json_file)





tokenizer = AutoTokenizer.from_pretrained("HyperbeeAI/Tulpar-7b-v0")
model = AutoModelForCausalLM.from_pretrained("HyperbeeAI/Tulpar-7b-v0")
model = model.to("cuda")

outdata=[]
for i in indata:
    time1=datetime.now()
    input_text=f"In as few words as possible ,please provide SOC_CODE for the job title: \"{i['job_title']}\", and put it in a json along with the SOC_TITLE "
    prompt = f"### User: {input_text}\n\n### Assistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=512)
    print(tokenizer.decode(output[0]))
    out=tokenizer.decode(output[0])
    dict_str=eval("{"+out.split("{")[1].split("}")[0]+"}")
    keys=list(dict_str.keys())
    i['new_soc_code']=dict_str[keys[0]]
    i['new_soc_title']=dict_str[keys[1]]
    
    print(f"time taken: {datetime.now()-time1}")

    
    outdata.append(i)

f = open('out_data.json', 'wb')
f.write(outdata)
