# HF_task

## 任务

使用Hugging Face平台、魔搭社区平台（可选）和魔乐社区平台（可选）下载文档中提到的模型（至少需要下载config.json文件、model.safetensors.index.json文件），请在必要的步骤以及结果当中截图。

## 使用Hugging Face下载模型

注册HF，使用codespaces创建环境，安装依赖

#### 下载internlm2_5-7b-chat的配置文件

![image-20241029215244573](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202410292152620.png)

![image-20241029215256037](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202410292152069.png)

如图所示，config.json文件、model.safetensors.index.json文件已经下好

#### 下载internlm2_5-chat-1_8b并打印示例输出

运行教程所给代码后，如图所示（续写 A beautiful flower)：

![image-20241029220123336](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202410292201421.png)

#### Hugging Face Spaces的使用

创建自己的space

![image-20241030151755548](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202410301517624.png)

clone到本地，将html文件修改后再push，回到space界面如下：

![image-20241030152744247](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202410301527395.png)

#### 模型上传

创建仓库，clone仓库，将刚刚下载好的config.json放进clone的仓库中，再写一个README.md文件

```markdown
# 书生浦语大模型实战营camp4
- hugging face模型上传测试
- 更多内容请访问 https://github.com/InternLM/Tutorial/tree/camp4
```

然后push，可以看到自己的model

![image-20241030154529794](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202410301545864.png)
