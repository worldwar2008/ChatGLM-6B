import os
os.environ['TRANSFORMERS_CACHE']='/home/guoqiang007/software/hug-models'
os.environ['HF_HOME'] = '/home/guoqiang007/software/hug-models'
from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html


from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

model_path = "/home/guoqiang007/software/clone-models/BELLE-7B-2M" # You can modify the path for storing the local model
model =  AutoModelForCausalLM.from_pretrained(model_path).cuda(1)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Human:")
line = """
请根据房源信息和经纪人信息,以及上下文信息来判断,如何来回复user的问题:
另外请注意，如果房源信息没有教育、交通、首付信息的话，可以不回答这方面的信息。

房源信息: 容积率:2.84;供暖:集中供暖;绿化率:35.0;面积:151.25平;物业费:1.75元;价格:960万;朝向:南北;产权年限:70年;小区:新街坊;户型:3室2厅2卫;楼层:6;城市:北京
经纪人信息: 姓名:凌昌平;品牌:链家;手机号:13911098060

user:有电梯吗
"""
inputs = 'Human: ' + line.strip() + '\n\nAssistant:'
input_ids = tokenizer(inputs, return_tensors="pt").input_ids.cuda(1)
outputs = model.generate(input_ids, max_new_tokens=200, do_sample = True, top_k = 30, top_p = 0.85, temperature = 0.35, repetition_penalty=1.2)
rets = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("Assistant:\n" + rets[0].strip().replace(inputs, ""))
print("\n------------------------------------------------\nHuman:")


# model_pth = "/home/guoqiang007/software/hug-models-nocache/chatglm-6b"

# tokenizer = AutoTokenizer.from_pretrained(model_pth, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_pth, trust_remote_code=True).half().cuda()
# model = model.eval()

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))
    for response, history in model.generate(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))       

        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(server_name='0.0.0.0',server_port=7653,share=False, inbrowser=True)
