# 8G 显存玩转书生大模型 Demo

## 关卡任务

本关任务主要包括：

1. InternLM2-Chat-1.8B 模型的部署（基础任务）
2. InternLM-XComposer2-VL-1.8B 模型的部署（进阶任务）
3. InternVL2-2B 模型的部署（进阶任务）

## 1. 使用 Cli Demo 进行 InternLM2-Chat-1.8B 模型的部署

在创建了虚拟环境并安装相关依赖后，创建目录并创建 `cli_demo.py`

```
mkdir -p /root/demo
touch /root/demo/cli_demo.py
```

然后，我们将下面的代码复制到 `cli_demo.py` 中。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)
```

运行`cli_demo.py`，结果如图所示：

![image-20240728084339743](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202407280843867.png)

## 2. InternLM2-Chat-1.8B 模型的部署

### 2.1 Streamlit Web Demo 部署 InternLM2-Chat-1.8B 模型

执行如下代码启动一个Streamlit服务

```
cd /root/demo
streamlit run /root/demo/Tutorial/tools/streamlit_demo.py --server.address 127.0.0.1 --server.port 6006
```

接下来，我们在**本地**的 PowerShell 中输入以下命令，将端口映射到本地。

```
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 40335
```

在完成端口映射后，我们便可以通过浏览器访问 `http://localhost:6006` 来启动我们的 Demo。

效果如下图所示：

![image-20240728090406785](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202407280904880.png)

### 2.2 LMDeploy 部署 InternLM-XComposer2-VL-1.8B 模型

首先，我们激活环境并安装 LMDeploy 以及其他依赖。

```
conda activate demo
pip install lmdeploy[all]==0.5.1
pip install timm==1.0.7
```

接下来，我们使用 LMDeploy 启动一个与 InternLM-XComposer2-VL-1.8B 模型交互的 Gradio 服务。

```
lmdeploy serve gradio /share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-1_8b --cache-max-entry-count 0.1
```

像上一步一样进行端口映射。

在使用 Upload Image 上传图片后，我们输入 Instruction 后按下回车，便可以看到模型的输出。

![image-20240728092809585](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202407280928707.png)

## 3. LMDeploy 部署 InternVL2-2B 模型

我们可以通过下面的命令来启动 InternVL2-2B 模型的 Gradio 服务。

```
conda activate demo
lmdeploy serve gradio /share/new_models/OpenGVLab/InternVL2-2B --cache-max-entry-count 0.1
```

在完成端口映射后，我们便可以通过浏览器访问 `http://localhost:6006` 来启动我们的 Demo。

在使用 Upload Image 上传图片后，我们输入 Instruction 后按下回车，便可以看到模型的输出。

![image-20240728093353793](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202407280933903.png)

可见，参数量增多时，模型的输出更加详细准确。