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

api_key = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1MDA3NjE1OSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTczMDE5NTQwOCwiY2xpZW50SWQiOiJlYm1ydm9kNnlvMG5semFlazF5cCIsInBob25lIjoiMTUxMzg3NDc2NDAiLCJ1dWlkIjoiY2MzMDJhN2UtZWJiNi00MWVmLTkwMzUtN2VlNDhkYWU3YjE0IiwiZW1haWwiOiIiLCJleHAiOjE3NDU3NDc0MDh9.KWfXBsph5zNKDMD7TtHa_khDtWSowcWxTq6bkg0JZT-gNT_FjsXlfBpz7FItocrh5-E1A5vZZNFWPLvVywwWaA"
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