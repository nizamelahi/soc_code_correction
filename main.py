from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datetime import datetime
import json


with open("data.json") as json_file:
    indata = json.load(json_file)


model_name_or_path = "TheBloke/Platypus2-70B-Instruct-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    revision="gptq-4bit-32g-actorder_True",
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


# Inference can also be done using transformers' pipeline

outdata = []
for indx, i in enumerate(indata):
    if indx == 10:
        break
    time1 = datetime.now()
    prompt = f"please provide united states SOC_CODE for the job title: {i['job_title']} and put it in JSON along with the SOC_TITLE "
    prompt_template = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {prompt}

    ### Response:

    """
    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
    print(tokenizer.decode(output[0]))
    out = tokenizer.decode(output[0])
    try:
        dict_str = eval("{" + out.split("{")[1].split("}")[0] + "}")
        keys = list(dict_str.keys())
        i["new_soc_code"] = dict_str[keys[0]]
        i["new_soc_title"] = dict_str[keys[1]]
    except Exception as e:
        print(e)
        i["new_soc_code"] = "unavailable"
        i["new_soc_title"] = "unavailable"

    print(f"time taken: {datetime.now()-time1}")

    outdata.append(i)

with open(f"outdata.json", "w") as f:
    f.write(json.dumps(outdata, indent=2))
