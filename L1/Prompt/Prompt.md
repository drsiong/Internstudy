# 浦语提示词工程实践

## 0. 前期准备

#### 0.1 环境配置

打开Terminal，运行如下脚本创建虚拟环境：

```cmd
# 创建虚拟环境
conda create -n langgpt python=3.10 -y
```

运行下面的命令，激活虚拟环境：

```cmd
conda activate langgpt
```

之后的操作都要在这个环境下进行。激活环境后，安装必要的Python包，依次运行下面的命令：

```
# 安装一些必要的库
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装其他依赖
pip install transformers==4.43.3

pip install streamlit==1.37.0
pip install huggingface_hub==0.24.3
pip install openai==1.37.1
pip install lmdeploy==0.5.2
```

#### 0.2 创建项目路径

运行如下命令创建并打开项目路径：

```cmd
## 创建路径
mkdir langgpt
## 进入项目路径
cd langgpt
```

#### 0.3 安装必要软件

运行下面的命令安装必要的软件：

```cmd
apt-get install tmux
```

## 1. 模型部署

这部分基于LMDeploy将开源的InternLM2-chat-1_8b模型部署为OpenAI格式的通用接口。

#### 1.1 获取模型

使用intern-studio开发机，可以直接在路径`/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b`下找到模型

#### 1.2 部署模型为OpenAI server

由于服务需要持续运行，需要将进程维持在后台，所以这里使用`tmux`软件创建新的命令窗口。运行如下命令创建窗口：

```cmd
tmux new -t langgpt
```

创建完成后，运行下面的命令进入新的命令窗口(首次创建自动进入，之后需要连接)：

```cmd
tmux a -t langgpt
```

进入命令窗口后，需要在新窗口中再次激活环境。然后，使用LMDeploy进行部署，参考如下命令：

使用LMDeploy进行部署，参考如下命令：

```cmd
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --server-port 23333 --api-keys internlm2
```

更多设置，可以参考：https://lmdeploy.readthedocs.io/en/latest/index.html

部署成功后，可以利用如下脚本调用部署的InternLM2-chat-1_8b模型并测试是否部署成功。

```
from openai import OpenAI

client = OpenAI(
    api_key = "internlm2",
    base_url = "http://0.0.0.0:23333/v1"
)

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {"role": "system", "content": "请介绍一下你自己"}
    ]
)

print(response.choices[0].message.content)
```

服务启动完成后，可以按Ctrl+B进入`tmux`的控制模式，然后按D退出窗口连接

如图，服务成功部署。

![image-20240824110440683](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408241104721.png)

#### 1.3 图形化界面调用

nternLM部署完成后，可利用提供的`chat_ui.py`创建图形化界面，在实战营项目的tools项目中。

首先，从Github获取项目，运行如下命令：

```cmd
git clone https://github.com/InternLM/Tutorial.git
```

下载完成后，运行如下命令进入项目所在的路径：

```cmd
cd Tutorial/tools
```

进入正确路径后，运行如下脚本运行项目：

```cmd
python -m streamlit run chat_ui.py
```

在本地终端中输入映射命令，进行端口映射：

```cmd
ssh -p 39646 root@ssh.intern-ai.org.cn -CNg -L 7860:127.0.0.1:8501 -o StrictHostKeyChecking=no
```

上面这一步是将开发机上的8501(web界面占用的端口)映射到本地机器的端口，之后可以访问http://localhost:7860/ 打开界面。

启动后界面如下：

![image-20240824110459737](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408241104819.png)

左侧边栏为对话的部分设置，其中最大token长度设置为0时表示不限制生成的最大token长度。API Key和Base URL是部署InternLM时的设置，必须填写。在保存设置之后，可以启动对话界面。

## 2. 基础任务

- **背景问题**：近期[相关研究](https://www.zhihu.com/search?q=相关研究&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A713006101})发现，LLM在对比[浮点数](https://www.zhihu.com/search?q=浮点数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A713006101})字时表现不佳，经验证，internlm2-chat-1.8b (internlm2-chat-7b)也存在这一问题，例如认为`13.8<13.11`。
- **任务要求**：利用LangGPT优化提示词，使LLM输出正确结果。**完成一次并提交截图即可**

#### 提示词：

```markdown
# Role: 数值比较器

## Background:
我是一个数值比较器，用于比较数值的大小，能够精确到小数点后许多位。

## Profile
- author: drsiong
- version: 1.0
- language: 中文/英文
- description: 数值比较器，能够比较两个数值的大小，并给出相应的提示词。

## Skills
- 数值比较，能够精确到小数点后许多位。
- 对于小数数字，确定它们是数值，而不是版本号。

## Goals:
- 对于小数数字，确定它们是数值，而不是版本号。
- 精确地比较两个数值，给出它们的大小关系。

## Constraints
- 保证结果准确。
- 回答简短，避免复杂的回答。
- 只需给出哪个数值更大即可。

## Workflows
1. 确定数值比较问题。
2. 计算两个数值的差值，如果小于0，则后面的数值大，如果大于0，则前面的数值大。
3. 直接给出哪个数值更大

## Init
我是一个数值比较器，请给我两个数字，我来帮你比较。
```

#### 运行结果：

![image-20240825030510816](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408250305944.png)