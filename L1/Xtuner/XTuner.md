# XTuner 微调个人小助手认知任务

## 基础任务

- 使用 XTuner 微调 InternLM2-Chat-1.8B 实现自己的小助手认知，记录复现过程并截图。

## 1. 准备工作

### 1.1 创建虚拟环境

创建虚拟环境，安装相关依赖

### 1.2 安装XTuner

虚拟环境创建完成后，就可以安装 XTuner 了。首先，从 Github 上下载源码。

```cmd
# 创建一个目录，用来存放源代码
mkdir -p /root/InternLM/code

cd /root/InternLM/code

git clone -b v0.1.21  https://github.com/InternLM/XTuner /root/InternLM/code/XTuner
```

其次，进入源码目录，执行安装。

```cmd
# 进入到源码目录
cd /root/InternLM/code/XTuner
conda activate xtuner0121

# 执行安装
pip install -e '.[deepspeed]'
```

最后，我们可以验证一下安装结果。

```cmd
xtuner version
```

![image-20240827051826595](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408270518639.png)

### 1.3 模型准备

软件安装好后，我们就可以准备要微调的模型了。

> 对于学习而言，我们可以使用 InternLM 推出的1.8B的小模型来完成此次微调演示。

对于在 InternStudio 上运行的小伙伴们，可以不用通过 HuggingFace、OpenXLab 或者 Modelscope 进行模型的下载，在开发机中已经为我们提供了模型的本地文件，直接使用就可以了。

> 我们可以通过以下代码一键通过符号链接的方式链接到模型文件，这样既节省了空间，也便于管理。

```cmd
# 创建一个目录，用来存放微调的所有资料，后续的所有操作都在该路径中进行
mkdir -p /root/InternLM/XTuner

cd /root/InternLM/XTuner

mkdir -p Shanghai_AI_Laboratory

ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b Shanghai_AI_Laboratory/internlm2-chat-1_8b
```

执行上述操作后，`Shanghai_AI_Laboratory/internlm2-chat-1_8b` 将直接成为一个符号链接，这个链接指向 `/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b` 的位置。

这意味着，当我们访问 `Shanghai_AI_Laboratory/internlm2-chat-1_8b` 时，实际上就是在访问 `/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b` 目录下的内容。通过这种方式，我们无需复制任何数据，就可以直接利用现有的模型文件进行后续的微调操作，从而节省存储空间并简化文件管理。

模型文件准备好后，我们可以使用`tree`命令来观察目录结构。

```cmd
apt-get install -y tree

tree -l
```

目录结构如下：

![image-20240827052018051](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408270520109.png)

## 2. 快速开始

这里我们用 `internlm2-chat-1_8b` 模型，通过 `QLoRA` 的方式来微调一个自己的小助手认知作为案例来进行演示。

### 2.1 微调前的模型对话

我们可以通过网页端的 Demo 来看看微调前 `internlm2-chat-1_8b` 的对话效果。

首先，我们需要准备一个Streamlit程序的脚本。

Streamlit程序的完整代码是：[tools/xtuner_streamlit_demo.py](https://github.com/InternLM/Tutorial/blob/camp3/tools/xtuner_streamlit_demo.py)。

然后，我们可以直接启动应用。

```
conda activate xtuner0121

streamlit run /root/InternLM/Tutorial/tools/xtuner_streamlit_demo.py
```

运行后，在访问前，我们还需要做的就是将端口映射到本地。

运行结果如图：

![image-20240827052414579](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408270524643.png)

### 2.2 指令跟随微调

下面我们对模型进行微调，让模型认识到自己的弟位，了解它自己是你的一个助手。

#### 2.2.1 准数据文件

为了让模型能够认清自己的身份弟位，在询问自己是谁的时候按照我们预期的结果进行回复，我们就需要通过在微调数据集中大量加入这样的数据。我们准备一个数据集文件`datas/assistant.json`，文件内容为对话数据。

```cmd
cd /root/InternLM/XTuner
mkdir -p datas
touch datas/assistant.json
```

为了简化数据文件准备，我们也可以通过脚本生成的方式来准备数据。创建一个脚本文件 `xtuner_generate_assistant.py` ：

```cmd
cd /root/InternLM/XTuner
touch xtuner_generate_assistant.py
```

输入脚本内容并保存：

```python
import json

# 设置用户的名字
name = '佳运同学'
# 设置需要重复添加的数据次数
n = 8000

# 初始化数据
data = [
    {"conversation": [{"input": "请介绍一下你自己", "output": "嘿嘿，我是{}的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦，请尽情吩咐妲己（bushi）".format(name)}]},
    {"conversation": [{"input": "你在实战营做什么", "output": "我在这里帮助{}完成XTuner微调个人小助手的任务捏".format(name)}]}
]

# 通过循环，将初始化的对话数据重复添加到data列表中
for i in range(n):
    data.append(data[0])
    data.append(data[1])

# 将data列表中的数据写入到'datas/assistant.json'文件中
with open('datas/assistant.json', 'w', encoding='utf-8') as f:
    # 使用json.dump方法将数据以JSON格式写入文件
    # ensure_ascii=False 确保中文字符正常显示
    # indent=4 使得文件内容格式化，便于阅读
    json.dump(data, f, ensure_ascii=False, indent=4)
```

准备好数据文件后，目录结构为：

![image-20240827052955925](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408270529990.png)

#### 2.2.2 准备配置文件

在准备好了模型和数据集后，我们就要根据我们选择的微调方法结合微调方案来找到与我们最匹配的配置文件了，从而减少我们对配置文件的修改量。

> 配置文件其实是一种用于定义和控制模型训练和测试过程中各个方面的参数和设置的工具。

##### 2.2.2.1 列出支持的配置文件

XTuner 提供多个开箱即用的配置文件，可以通过以下命令查看。

> `xtuner list-cfg` 命令用于列出内置的所有配置文件。参数 `-p` 或 `--pattern` 表示模式匹配，后面跟着的内容将会在所有的配置文件里进行模糊匹配搜索，然后返回最有可能得内容。比如我们这里微调的是书生·浦语的模型，我们就可以匹配搜索 `internlm2`。

```
conda activate xtuner0121

xtuner list-cfg -p internlm2
```

<details style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;">配置文件名的解释</summary><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);"></strong><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);"></strong></p><markdown-accessiblity-table data-catalyst="" style="box-sizing: border-box; display: block;"><table tabindex="0" style="box-sizing: border-box; border-spacing: 0px; border-collapse: collapse; margin-top: 0px; margin-bottom: var(--base-size-16); display: block; width: max-content; max-width: 100%; overflow: auto;"><thead style="box-sizing: border-box;"><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><th style="box-sizing: border-box; padding: 6px 13px; font-weight: var(--base-text-weight-semibold, 600); border: 1px solid var(--borderColor-default, var(--color-border-default));"></th><th style="box-sizing: border-box; padding: 6px 13px; font-weight: var(--base-text-weight-semibold, 600); border: 1px solid var(--borderColor-default, var(--color-border-default));"></th><th style="box-sizing: border-box; padding: 6px 13px; font-weight: var(--base-text-weight-semibold, 600); border: 1px solid var(--borderColor-default, var(--color-border-default));"></th></tr></thead><tbody style="box-sizing: border-box;"><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-default, var(--color-canvas-default)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr><tr style="box-sizing: border-box; background-color: var(--bgColor-muted, var(--color-canvas-subtle)); border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));"><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td><td style="box-sizing: border-box; padding: 6px 13px; border: 1px solid var(--borderColor-default, var(--color-border-default));"></td></tr></tbody></table></markdown-accessiblity-table></details>

##### 2.2.2.2 复制一个预设的配置文件

由于我们是对`internlm2-chat-1_8b`模型进行指令微调，所以与我们的需求最匹配的配置文件是 `internlm2_chat_1_8b_qlora_alpaca_e3`，这里就复制该配置文件。

> `xtuner copy-cfg` 命令用于复制一个内置的配置文件。该命令需要两个参数：`CONFIG` 代表需要复制的配置文件名称，`SAVE_PATH` 代表复制的目标路径。在我们的输入的这个命令中，我们的 `CONFIG` 对应的是上面搜索到的 `internlm2_chat_1_8b_qlora_alpaca_e3` ,而 `SAVE_PATH` 则是当前目录 `.`。

```
cd /root/InternLM/XTuner
conda activate xtuner0121

xtuner copy-cfg internlm2_chat_1_8b_qlora_alpaca_e3 .
```

复制好配置文件后，目录结构为：

![image-20240827053421807](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408270534878.png)

##### 2.2.2.3 对配置文件进行修改

在选择了一个最匹配的配置文件并准备好其他内容后，下面我们要做的事情就是根据我们自己的内容对该配置文件进行调整，使其能够满足我们实际训练的要求。

<details open="" style="box-sizing: border-box; display: block; margin-top: 0px; margin-bottom: var(--base-size-16); color: rgb(31, 35, 40); font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(255, 255, 255); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><summary style="box-sizing: border-box; display: list-item; cursor: pointer;"><b style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);">配置文件介绍</b></summary><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);">打开配置文件后，我们可以看到整体的配置文件分为五部分：</p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);">PART 1 Settings</strong>：涵盖了模型基本设置，如预训练模型的选择、数据集信息和训练过程中的一些基本参数（如批大小、学习率等）。</p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);">PART 2 Model &amp; Tokenizer</strong>：指定了用于训练的模型和分词器的具体类型及其配置，包括预训练模型的路径和是否启用特定功能（如可变长度注意力），这是模型训练的核心组成部分。</p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);">PART 3 Dataset &amp; Dataloader</strong>：描述了数据处理的细节，包括如何加载数据集、预处理步骤、批处理大小等，确保了模型能够接收到正确格式和质量的数据。</p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);">PART 4 Scheduler &amp; Optimizer</strong>：配置了优化过程中的关键参数，如学习率调度策略和优化器的选择，这些是影响模型训练效果和速度的重要因素。</p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);"><strong style="box-sizing: border-box; font-weight: var(--base-text-weight-semibold, 600);">PART 5 Runtime</strong>：定义了训练过程中的额外设置，如日志记录、模型保存策略和自定义钩子等，以支持训练流程的监控、调试和结果的保存。</p><p dir="auto" style="box-sizing: border-box; margin-top: 0px; margin-bottom: var(--base-size-16);">一般来说我们需要更改的部分其实只包括前三部分，而且修改的主要原因是我们修改了配置文件中规定的模型、数据集。后两部分都是 XTuner 官方帮我们优化好的东西，一般而言只有在魔改的情况下才需要进行修改。</p></details>

下面我们将根据项目的需求一步步的进行修改和调整吧！

在 PART 1 的部分，由于我们不再需要在 HuggingFace 上自动下载模型，因此我们先要更换模型的路径以及数据集的路径为我们本地的路径。

为了训练过程中能够实时观察到模型的变化情况，XTuner 贴心的推出了一个 `evaluation_inputs` 的参数来让我们能够设置多个问题来确保模型在训练过程中的变化是朝着我们想要的方向前进的。我们可以添加自己的输入。

在 PART 3 的部分，由于我们准备的数据集是 JSON 格式的数据，并且对话内容已经是 `input` 和 `output` 的数据对，所以不需要进行格式转换。

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- pretrained_model_name_or_path = 'internlm/internlm2-chat-1_8b'
+ pretrained_model_name_or_path = '/root/InternLM/XTuner/Shanghai_AI_Laboratory/internlm2-chat-1_8b'

- alpaca_en_path = 'tatsu-lab/alpaca'
+ alpaca_en_path = 'datas/assistant.json'

evaluation_inputs = [
-    '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
+    '请介绍一下你自己', 'Please introduce yourself'
]

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
alpaca_en = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=alpaca_en_path),
+   dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),
    tokenizer=tokenizer,
    max_length=max_length,
-   dataset_map_fn=alpaca_map_fn,
+   dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)
```

#### 2.2.3 启动微调

当我们准备好了所有内容，我们只需要将使用 `xtuner train` 命令令即可开始训练。

> `xtuner train` 命令用于启动模型微调进程。该命令需要一个参数：`CONFIG` 用于指定微调配置文件。这里我们使用修改好的配置文件 `internlm2_chat_1_8b_qlora_alpaca_e3_copy.py`。
> 训练过程中产生的所有文件，包括日志、配置文件、检查点文件、微调后的模型等，默认保存在 `work_dirs` 目录下，我们也可以通过添加 `--work-dir` 指定特定的文件保存位置。

```cmd
cd /root/InternLM/XTuner
conda activate xtuner0121

xtuner train ./internlm2_chat_1_8b_qlora_alpaca_e3_copy.py
```

在训练完后，目录结构为：

![image-20240827062027621](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408270620695.png)

#### 2.2.4 模型格式转换

模型转换的本质其实就是将原本使用 Pytorch 训练出来的模型权重文件转换为目前通用的 HuggingFace 格式文件，那么我们可以通过以下命令来实现一键转换。

我们可以使用 `xtuner convert pth_to_hf` 命令来进行模型格式转换。

> `xtuner convert pth_to_hf` 命令用于进行模型格式转换。该命令需要三个参数：`CONFIG` 表示微调的配置文件， `PATH_TO_PTH_MODEL` 表示微调的模型权重文件路径，即要转换的模型权重， `SAVE_PATH_TO_HF_MODEL` 表示转换后的 HuggingFace 格式文件的保存路径。

除此之外，我们其实还可以在转换的命令中添加几个额外的参数，包括：

| 参数名                | 解释                                         |
| --------------------- | -------------------------------------------- |
| --fp32                | 代表以fp32的精度开启，假如不输入则默认为fp16 |
| --max-shard-size {GB} | 代表每个权重文件最大的大小（默认为2GB）      |

```cmd
cd /root/InternLM/XTuner
conda activate xtuner0121

# 先获取最后保存的一个pth文件
pth_file=`ls -t ./work_dirs/internlm2_chat_1_8b_qlora_alpaca_e3_copy/*.pth | head -n 1`
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert pth_to_hf ./internlm2_chat_1_8b_qlora_alpaca_e3_copy.py ${pth_file} ./hf
```

模型格式转换完成后，我们的目录结构：

![image-20240827062324038](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408270623102.png)

转换完成后，可以看到模型被转换为 HuggingFace 中常用的 .bin 格式文件，这就代表着文件成功被转化为 HuggingFace 格式了。

此时，hf 文件夹即为我们平时所理解的所谓 “LoRA 模型文件”

> 可以简单理解：LoRA 模型文件 = Adapter

#### 2.2.5 模型合并

对于 LoRA 或者 QLoRA 微调出来的模型其实并不是一个完整的模型，而是一个额外的层（Adapter），训练完的这个层最终还是要与原模型进行合并才能被正常的使用。

> 对于全量微调的模型（full）其实是不需要进行整合这一步的，因为全量微调修改的是原模型的权重而非微调一个新的 Adapter ，因此是不需要进行模型整合的。

在 XTuner 中提供了一键合并的命令 `xtuner convert merge`，在使用前我们需要准备好三个路径，包括原模型的路径、训练好的 Adapter 层的（模型格式转换后的）路径以及最终保存的路径。

> `xtuner convert merge`命令用于合并模型。该命令需要三个参数：`LLM` 表示原模型路径，`ADAPTER` 表示 Adapter 层的路径， `SAVE_PATH` 表示合并后的模型最终的保存路径。

```cmd
cd /root/InternLM/XTuner
conda activate xtuner0121

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert merge /root/InternLM/XTuner/Shanghai_AI_Laboratory/internlm2-chat-1_8b ./hf ./merged --max-shard-size 2GB
```

合并完成后，目录结构为：

![image-20240827062732445](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408270627517.png)

### 2.3 微调后的模型对话

微调完成后，我们可以再次运行`xtuner_streamlit_demo.py`脚本来观察微调后的对话效果，不过在运行之前，我们需要将脚本中的模型路径修改为微调后的模型的路径。

```cmd
# 直接修改脚本文件第18行
- model_name_or_path = "/root/InternLM/XTuner/Shanghai_AI_Laboratory/internlm2-chat-1_8b"
+ model_name_or_path = "/root/InternLM/XTuner/merged"
```

然后，我们可以直接启动应用。（端口映射之后）

```cmd
conda activate xtuner0121

streamlit run /root/InternLM/Tutorial/tools/xtuner_streamlit_demo.py
```

运行结果如下：

![image-20240827063121392](https://typora-drsiong.oss-cn-beijing.aliyuncs.com/img/202408270631481.png)
