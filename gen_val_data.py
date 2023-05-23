import json
import os
os.environ['TRANSFORMERS_CACHE']='/home/guoqiang007/software/hug-models'
os.environ['HF_HOME'] = '/home/guoqiang007/software/hug-models'
import torch
import mdtex2html
import gradio as gr
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer


CHECKPOINT_PATH = "/home/guoqiang007/code/ChatGLM-6B/checkpoint-ptuning-v3/checkpoint-3000"
model_pth = "/home/guoqiang007/software/hug-models-nocache/chatglm-6b"

config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True, pre_seq_len=128)
tokenizer = AutoTokenizer.from_pretrained(model_pth, trust_remote_code=True)

model = AutoModel.from_pretrained(model_pth, config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
# Comment out the following line if you don't use quantization
# model = model.quantize(4)
model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

d = ""
with open("./real_im_val_raw_v3.txt",encoding='utf-8') as ff:
    for line in ff.readlines():
        line = line.strip()
        d += line
data = json.loads(d)
print(len(data))
prompt = """
请根据下面房源信息和经纪人信息，以及上下文信息判断，如何回复user的问题：\n 回复的时候按照的规则是：如果在房源信息没找到答案可以回答没找到相关信息 \n
"""
for i in data:
    print("i", i)
    new_prompt = prompt + "房源信息:"+ data[i]['hous_info'] + "\n" + "经纪人信息:" + data[i]["agent_info"] + "\n"
    new_prompt.encode("utf-8")
    myhistory = []
    first = 0
    data[i]["conv"].insert(0,"你好")
    for user_quest in data[i]["conv"]:
        if first == 0:
            new_prompt += "user:"+ user_quest
        else:
            new_prompt = "user:" + user_quest
        first += 1
        response,history = model.chat(tokenizer, new_prompt, myhistory, max_length=512, top_p=0.7,
                                               temperature=0.05)
        myhistory.append(["user:"+ user_quest,response])
        print("response", json.dumps(response, ensure_ascii=False))
        print("history", json.dumps(myhistory,ensure_ascii=False))
    data[i]["conv_detail"] = myhistory
    #break        
    print(new_prompt)
with open ("real_im_val.v3.add_first.txt",'w+') as ff:
    ff.writelines(json.dumps(data))

