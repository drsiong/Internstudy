# encoding=gbk
from openai import OpenAI
import json
def internlm_gen(prompt,client):
    '''
    LLM���ɺ���
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
��������InternLM2.5���Ϻ��˹�����ʵ������2024��7���Ƴ�����һ��������ģ�ͣ��ṩ1.8B��7B��20B���ֲ����汾������Ӧ��ͬ����
��ģ���ڸ��ӳ����µ����������õ�ȫ����ǿ��֧��1M���������ģ����������л�����������������Ϣ��
'''
prompt = f'''
����Ҵ�����``�ڵ����ģ�ͽ�����������ȡ���ڸ�ģ�͵���Ϣ��Ҫ�����ģ�����֡������������ṩ�����汾�������ĳ����ĸ����ݣ�����ȷ��json��ʽ���أ������汾Ӧ����˫���š�
`{content}`
'''
res = internlm_gen(prompt,client)
res_json = json.loads(res.replace('json','').replace('```',''))
print(res_json)