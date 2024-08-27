# llamaindex+Internlm2 RAG实践

## 1. 环境、模型准备

### 1.1 配置基础环境

创建虚拟环境，安装相关依赖

### 1.2 安装 Llamaindex

安装 Llamaindex和相关的包

```cmd
conda activate llamaindex
pip install llama-index==0.10.38 llama-index-llms-huggingface==0.2.0 "transformers[torch]==4.41.1" "huggingface_hub[inference]==0.23.1" huggingface_hub==0.23.1 sentence-transformers==2.7.0 sentencepiece==0.2.0
```

### 1.3 下载 Sentence Transformer 模型

源词向量模型 [Sentence Transformer](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2):（我们也可以选用别的开源词向量模型来进行 Embedding，目前选用这个模型是相对轻量、支持中文且效果较好的，同学们可以自由尝试别的开源词向量模型） 运行以下指令，新建一个python文件

```cmd
cd ~
mkdir llamaindex_demo
mkdir model
cd ~/llamaindex_demo
touch download_hf.py
```

打开`download_hf.py` 贴入以下代码

```python
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /root/model/sentence-transformer')
```

然后，在 /root/llamaindex_demo 目录下执行该脚本即可自动开始下载：

```cmd
cd /root/llamaindex_demo
conda activate llamaindex
python download_hf.py
```

### 1.4 下载 NLTK 相关资源

我们在使用开源词向量模型构建开源词向量的时候，需要用到第三方库 `nltk` 的一些资源。正常情况下，其会自动从互联网上下载，但可能由于网络原因会导致下载中断，此处我们可以从国内仓库镜像地址下载相关资源，保存到服务器上。 我们用以下命令下载 nltk 资源并解压到服务器上：

```cmd
cd /root
git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
cd nltk_data
mv packages/*  ./
cd tokenizers
unzip punkt.zip
cd ../taggers
unzip averaged_perceptron_tagger.zip
```

之后使用时服务器即会自动使用已有资源，无需再次下载

## 2. LlamaIndex HuggingFaceLLM

运行以下指令，把 `InternLM2 1.8B` 软连接出来

```cmd
cd ~/model
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b/ ./
```

运行以下指令，新建一个python文件

```cmd
cd ~/llamaindex_demo
touch llamaindex_internlm.py
```

打开llamaindex_internlm.py 贴入以下代码

```python
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage
llm = HuggingFaceLLM(
    model_name="/root/model/internlm2-chat-1_8b",
    tokenizer_name="/root/model/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)

rsp = llm.chat(messages=[ChatMessage(content="xtuner是什么？")])
print(rsp)
```

之后运行

```cmd
conda activate llamaindex
cd ~/llamaindex_demo/
python llamaindex_internlm.py
```

结果为：

![image-20240826170435155](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408261704251.png)

回答得并不好。

## 3. LlamaIndex RAG

安装 `LlamaIndex` 词嵌入向量依赖

```cmd
conda activate llamaindex
pip install llama-index-embeddings-huggingface==0.2.0 llama-index-embeddings-instructor==0.1.3
```

运行以下命令，获取知识库

```cmd
cd ~/llamaindex_demo
mkdir data
cd data
git clone https://github.com/InternLM/xtuner.git
mv xtuner/README_zh-CN.md ./
```

运行以下指令，新建一个python文件

```cmd
cd ~/llamaindex_demo
touch llamaindex_RAG.py
```

打开`llamaindex_RAG.py`贴入以下代码

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

#初始化一个HuggingFaceEmbedding对象，用于将文本转换为向量表示
embed_model = HuggingFaceEmbedding(
#指定了一个预训练的sentence-transformer模型的路径
    model_name="/root/model/sentence-transformer"
)
#将创建的嵌入模型赋值给全局设置的embed_model属性，
#这样在后续的索引构建过程中就会使用这个模型。
Settings.embed_model = embed_model

llm = HuggingFaceLLM(
    model_name="/root/model/internlm2-chat-1_8b",
    tokenizer_name="/root/model/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)
#设置全局的llm属性，这样在索引查询时会使用这个模型。
Settings.llm = llm

#从指定目录读取所有文档，并加载数据到内存中
documents = SimpleDirectoryReader("/root/llamaindex_demo/data").load_data()
#创建一个VectorStoreIndex，并使用之前加载的文档来构建索引。
# 此索引将文档转换为向量，并存储这些向量以便于快速检索。
index = VectorStoreIndex.from_documents(documents)
# 创建一个查询引擎，这个引擎可以接收查询并返回相关文档的响应。
query_engine = index.as_query_engine()
response = query_engine.query("xtuner是什么?")

print(response)
```

之后运行

```cmd
conda activate llamaindex
cd ~/llamaindex_demo/
python llamaindex_RAG.py
```

结果为：

![image-20240826174214777](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408261742902.png)

## 4. LlamaIndex web

运行之前首先安装依赖

```cmd
pip install streamlit==1.36.0
```

运行以下指令，新建一个python文件

```cmd
cd ~/llamaindex_demo
touch app.py
```

打开`app.py`贴入以下代码

```python
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

st.set_page_config(page_title="llama_index_demo", page_icon="🦜🔗")
st.title("llama_index_demo")

# 初始化模型
@st.cache_resource
def init_models():
    embed_model = HuggingFaceEmbedding(
        model_name="/root/model/sentence-transformer"
    )
    Settings.embed_model = embed_model

    llm = HuggingFaceLLM(
        model_name="/root/model/internlm2-chat-1_8b",
        tokenizer_name="/root/model/internlm2-chat-1_8b",
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True}
    )
    Settings.llm = llm

    documents = SimpleDirectoryReader("/root/llamaindex_demo/data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    return query_engine

# 检查是否需要初始化模型
if 'query_engine' not in st.session_state:
    st.session_state['query_engine'] = init_models()

def greet2(question):
    response = st.session_state['query_engine'].query(question)
    return response

      
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "你好，我是你的助手，有什么我可以帮助你的吗？"}]    

    # Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "你好，我是你的助手，有什么我可以帮助你的吗？"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama_index_response(prompt_input):
    return greet2(prompt_input)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Gegenerate_llama_index_response last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama_index_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
```

进行端口映射之后运行

```cmd
streamlit run app.py
```

![image-20240826175456220](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408261754321.png)

## 5.基础任务

- **任务要求**：基于 LlamaIndex 构建自己的 RAG 知识库，寻找一个问题 A 在使用 LlamaIndex 之前InternLM2-Chat-1.8B模型不会回答，借助 LlamaIndex 后 InternLM2-Chat-1.8B 模型具备回答 A 的能力，截图保存。

#### 设置问题

MindSearch是什么？

只需要将之前的代码中提问部分改为上述问题即可。

回答如下：

![image-20240826180419723](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408261804828.png)

回答得太笼统，不尽人意。

#### 利用LlamaIndex RAG

从链接中下载MindSearch的介绍来当作知识库

[MindSearch/README_zh-CN.md at main · InternLM/MindSearch · GitHub](https://github.com/InternLM/MindSearch/blob/main/README_zh-CN.md)

然后将它替换掉之前的知识库

![image-20240826181450320](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408261814408.png)

#### LlamaIndex Web

进行端口映射之后运行结果如下：

![image-20240826181838167](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408261818259.png)