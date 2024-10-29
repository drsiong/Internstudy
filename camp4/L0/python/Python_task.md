# Python_task

### 任务一

完成[Leetcode 383](https://leetcode.cn/problems/ransom-note/description/), 笔记中提交代码与leetcode提交通过截图

### 任务二

下面是一段调用书生浦语API实现将非结构化文本转化成结构化json的例子，其中有一个小bug会导致报错。请大家自行通过debug功能定位到报错原因并做修正。

## 任务一：Leetcode383

![image-20241029173607024](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202410291736116.png)

```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        m = list(magazine)
        r = list(ransomNote)
        for i in range(0, len(m)):
            if m[i] in r:
                r.remove(m[i])
        if len(r)== 0:
            return True
        else:
            return False
```

## 任务二：Vscode 连接InternStudio debug

初始代码为：

```python
# encoding=gbk
from openai import OpenAI
import json
def internlm_gen(prompt,client):
    '''
    LLM生成函数
    Param prompt: prompt string
    Param client: OpenAI client 
    '''
    response = client.chat.completions.create(
        model="internlm2.5-latest",
        messages=[
            {"role": "user", "content": prompt},
      ],
        stream=False
    )
    return response.choices[0].message.content

api_key = 'my api_key'
client = OpenAI(base_url="https://internlm-chat.intern-ai.org.cn/puyu/api/v1/",api_key=api_key)

content = """
书生浦语InternLM2.5是上海人工智能实验室于2024年7月推出的新一代大语言模型，提供1.8B、7B和20B三种参数版本，以适应不同需求。
该模型在复杂场景下的推理能力得到全面增强，支持1M超长上下文，能自主进行互联网搜索并整合信息。
"""
prompt = f"""
请帮我从以下``内的这段模型介绍文字中提取关于该模型的信息，要求包含模型名字、开发机构、提供参数版本、上下文长度四个内容，以json格式返回。
`{content}`
"""
res = internlm_gen(prompt,client)
res_json = json.loads(res)
print(res_json)
```

#### ssh连接后设置remote attach，并设置断点

![image-20241029184508160](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202410291845243.png)

#### 启动调试并查看变量值

![image-20241029184655619](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202410291846695.png)

#### 逐步调试后出现报错

![image-20241029185444439](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202410291854557.png)

原因是期望一个值，但是出现了nonetype，可能是空字符串或者格式不匹配，所以检查传入值res

```json
'```json\n{\n  "model_name": "书生浦语InternLM2.5",\n  "development_institution": "上海人工智能实验室",\n  "parameter_versions": [1.8B, 7B, 20B],\n  "maximum_context_length": 1000000\n}\n```'
```

可以看到，res中多了反引号 ```，多了json，1.8B等没有加双引号。

所以需要将res中的反引号和json去掉（对res操作），并且参数版本需要加上双引号（可以在prompt中实现）

修改后代码如下（已经将api_key抽象）：

```python
# encoding=gbk
from openai import OpenAI
import json
def internlm_gen(prompt,client):
    '''
    LLM生成函数
    Param prompt: prompt string
    Param client: OpenAI client 
    '''
    response = client.chat.completions.create(
        model="internlm2.5-latest",
        messages=[
            {"role": "user", "content": prompt},
       ],
        stream=False
    )
    return response.choices[0].message.content

api_key = "my api_key"
client = OpenAI(base_url="https://internlm-chat.intern-ai.org.cn/puyu/api/v1/",api_key=api_key)

content = '''
书生浦语InternLM2.5是上海人工智能实验室于2024年7月推出的新一代大语言模型，提供1.8B、7B和20B三种参数版本，以适应不同需求。
该模型在复杂场景下的推理能力得到全面增强，支持1M超长上下文，能自主进行互联网搜索并整合信息。
'''
prompt = f'''
请帮我从以下``内的这段模型介绍文字中提取关于该模型的信息，要求包含模型名字、开发机构、提供参数版本、上下文长度四个内容，以正确的json格式返回，参数版本应该有双引号。
`{content}`
'''
res = internlm_gen(prompt,client)
res_json = json.loads(res.replace('json','').replace('```',''))
print(res_json)
```

运行结果如下：

![image-20241029205516539](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202410292055597.png)

ps：两遍运行，第一遍运行先输出了res的值，第二遍运行是上述代码的结果。
