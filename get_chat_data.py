import json
import random
text = ""
with open("/aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/dia-new-v2.4.txt") as ff:
    for line in ff.readlines():
        text += line.strip()

data = json.loads(text)
new_data = []
for line in data:
    print(line)
    prompt = "请根据下面房源信息和经纪人信息，以及上下文信息判断，如何回复user的问题：\n 回复的时候按照的规则是：如果在房源信息没找到答案可以回答没找到相关信息 \n"
    line["prompt"] = prompt+ "房源信息: " + line["instruction"] + "\n" + \
                     "经纪人信息: " + line["agent_info"] + "\n" + \
                    line["input"][-1]


    line["response"] = line["output"]
    
    if len(line["input"]) == 1:
        line["history"] = []
    else:
        tmp = []
        length = len(line["input"])-1
        for j in range(0,length-1,2):
            tmp.append([line["input"][j],line["input"][j+1]])
        line["history"] = tmp
    del line["input"],line["output"],line["instruction"], line["agent_info"]
    new_data.append(line)


with open("/aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/dia-new-glm.json",'w') as ff:
    ff.writelines(json.dumps(new_data,ensure_ascii=False,indent=True))

random.shuffle(new_data)

with open("/aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/dia-new-glm.train.json",'w') as ff:
    ff.writelines(json.dumps(new_data[:4000],ensure_ascii=False,indent=True))


with open("/aistudio/workspace/system-default/envs/python3.8/software/ChatGLM-6B/dia-new-glm.val.json",'w') as ff:
    ff.writelines(json.dumps(new_data[1000:],ensure_ascii=False,indent=True))