# Python task

### 任务一

请实现一个wordcount函数，统计英文字符串中每个单词出现的次数。返回一个字典，key为单词，value为对应单词出现的次数。

Eg:

Input:

```python
"""Hello world!  
This is an example.  
Word count is fun.  
Is it fun to count words?  
Yes, it is fun!"""
```



Output:

```python
{'hello': 1,'world!': 1,'this': 1,'is': 3,'an': 1,'example': 1,'word': 1, 
'count': 2,'fun': 1,'Is': 1,'it': 2,'to': 1,'words': 1,'Yes': 1,'fun': 1  }
```



TIPS：记得先去掉标点符号,然后把每个单词转换成小写。不需要考虑特别多的标点符号，只需要考虑实例输入中存在的就可以。

```python
text = """
Got this panda plush toy for my daughter's birthday,
who loves it and takes it everywhere. It's soft and
super cute, and its face has a friendly look. It's
a bit small for what I paid though. I think there
might be other options that are bigger for the
same price. It arrived a day earlier than expected,
so I got to play with it myself before I gave it
to her.
"""

def wordcount(text):
    pass
```

思路：先将text中的逗号、句号全部去掉，然后将text小写并split，之后遍历即可

代码：

```python
text = """
Got this panda plush toy for my daughter's birthday,
who loves it and takes it everywhere. It's soft and
super cute, and its face has a friendly look. It's
a bit small for what I paid though. I think there
might be other options that are bigger for the
same price. It arrived a day earlier than expected,
so I got to play with it myself before I gave it
to her.
"""

def wordcount(text):
    punc = [',', '.']
    for i in punc:
        text = text.replace(i, '')    
    new_text = text.lower().split()
    words = {}
    for word in new_text:
        if word in words:
            words[word] += 1
        else:
            words[word] = 1
    return words

print(wordcount(text))
```

运行结果如图：

![屏幕截图 2024-07-15 172151](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202407151727448.png)

### 任务二

请使用本地vscode连接远程开发机，将上面你写的wordcount函数在开发机上进行debug，体验debug的全流程，并完成一份debug笔记(需要截图)。

步骤：

左键单击以设置断点

![image-20240715172941502](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202407151729550.png)

点击右上角的调试按钮，选择以launch.json 进行调试

![image-20240715173219826](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202407151732898.png)

之后选择 Python Debugger，并选择调试当前文件

![image-20240715173312902](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202407151733971.png)

开始调试，左边会显示变量，局部和全局，上边会有单步调试，单步跳出等，一般选择单步调试一步一步看问题

![image-20240715173631201](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202407151736307.png)

当调试到最后，输出结果，调试结束

![image-20240715173757081](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202407151737186.png)