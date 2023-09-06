import json
import pandas as pd

with open('job_data_all.json') as json_file:
    data = json.load(json_file)

mismatched_soc={}
mismatched_ind={}
count=0
for i in data:
    count+=1
    if i["match_status"] == "not matched":
        if(mismatched_soc.get(i["soc_title"])):
            mismatched_soc[i["soc_title"]]+=1
        else:
            mismatched_soc[i["soc_title"]]=1

        if(mismatched_ind.get(i["industry"])):
            mismatched_ind[i["industry"]]+=1
        else:
            mismatched_ind[i["industry"]]=1

df = pd.DataFrame(mismatched_soc.items(), columns=['soctitle', 'value'])

df=df.sort_values(by=['value'],ascending=False)

print(mismatched_ind)
print("_______________________________________________________________")

print(df.head(30))
print(f"total   {count}")