import os
os.environ['TRANSFORMERS_CACHE']='/home/guoqiang007/software/hug-models'
os.environ['HF_HOME'] = '/home/guoqiang007/software/hug-models'
from transformers import AutoTokenizer, AutoModel

model_pth="THUDM/chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained(model_pth, trust_remote_code=True)

model = AutoModel.from_pretrained(model_pth, trust_remote_code=True).half().cuda()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
