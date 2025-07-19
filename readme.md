# ShieldVLM: Cross-Modal Content Safety Assessment Model

[English](#english) | [中文](#中文)

<a name="english"></a>

## English

ShieldVLM is an advanced Vision-Language Model (VLM) specifically designed to evaluate the safety risks within combined text and image content (cross-modal content). It possesses a deep understanding of the correlation between text and images, enabling it to identify a wide range of potential risks.

### ✨ Features

ShieldVLM can identify the following seven major categories of safety risks:

1.  **Offensiveness**: Threat, insult, scorn, profanity, sarcasm, impoliteness, etc.
2.  **Discrimination & Stereotyping**: Social bias across various topics such as race, gender, religion, age, etc.
3.  **Physical Harm**: Actions or expressions that may influence human physical health.
4.  **Illegal Activities**: Behaviors that could cause negative societal repercussions.
5.  **Violation of Morality**: Immoral activities that do not necessarily violate the law.
6.  **Privacy & Property Damage**: Issues related to privacy and property loss.
7.  **Misinformation**: Misleading or false information.

The model supports three core assessment tasks:
* **Statement Safety Assessment**: Directly assesses whether a given text-image pair contains safety risks.
* **Prompt Safety Assessment**: Evaluates whether a question combined with an image could lead to an unsafe or harmful response.
* **Dialogue Safety Assessment**: Assesses the safety of a response within a multimodal dialogue context.

### Models and Datasets

* **Model Hub**: [thu-coai/ShieldVLM-7B-qwen](https://huggingface.co/thu-coai/ShieldVLM-7B-qwen)
* **Dataset Hub**: [thu-coai/ShieldVLM](https://huggingface.co/datasets/thu-coai/ShieldVLM)

### ⚙️ Setup and Installation

1.  Clone this repository.
2.  Install the required dependencies. It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include `transformers`, `torch`, `peft`, and `trl`.

### 🚀 Quick Start

#### Inference

You can use the `infer_shieldvlm.sh` script to quickly perform safety assessments on your data.

1.  **Prepare Input Data**:
    Prepare your data according to the format in the `examples/` directory (e.g., `examples/statement/example_input.json`). Each entry should contain an `id`, `risk_type`, `image` (path to the image), and `text`.

2.  **Run the Inference Script**:
    Modify the paths in `infer_shieldvlm.sh` or execute `infer.py` directly from the command line.

    ```bash
    # Example: Run statement safety assessment
    MODEL_PATH="thu-coai/ShieldVLM-7B-qwen" # Or your local model path

    CUDA_VISIBLE_DEVICES=0 python infer.py \
        --category "statement" \
        --model_path ${MODEL_PATH} \
        --input_path ./examples/statement/example_input.json \
        --output_path ./examples/statement/example_output.json
    ```
    * `--category`: The type of task. Options include `statement`, `prompt`, `dialog`.
    * `--model_path`: Path to your model checkpoint on Hugging Face or a local directory.
    * `--input_path`: Path to the input `json` file.
    * `--output_path`: Path to save the output `json` file with assessment results.

#### Training

ShieldVLM was trained via full supervised fine-tuning based on the `Qwen2.5-VL-7B-Instruct` model, using the **LLaMA Factory** framework.

1.  **Prepare Datasets**:
    The training set needs to be prepared according to the requirements of the **LLaMA Factory** framework. You can use the `train` folder within our [Dataset Hub](https://huggingface.co/datasets/thu-coai/ShieldVLM) as the source for the training set.

2.  **Configure Training**:
    Modify the configuration file in `train_code/` (e.g., `train_shieldvlm.yaml`) to set your model, dataset paths, and training hyperparameters.

3.  **Start Training with LLaMA Factory**:
    Execute the following script to begin training.

    ```bash
    bash train_code/run.sh
    ```
   

---

<a name="中文"></a>

## 中文

ShieldVLM 是一个先进的视觉语言模型 (Vision-Language Model, VLM)，专为评估文本与图像组合（跨模态内容）中的安全风险而设计。它能够深度理解图文之间的关联，并识别多种潜在的风险类型。

### ✨ 功能特性

ShieldVLM 能够识别以下七个主要类别的安全风险：

1.  **冒犯性 (Offensiveness)**: 涉及威胁、侮辱、嘲讽、亵渎、讽刺等不礼貌内容。
2.  **歧视与刻板印象 (Discrimination & Stereotyping)**: 涉及种族、性别、宗教、年龄等方面的社会偏见。
3.  **人身伤害 (Physical Harm)**: 关注可能影响人类身体健康的行为或表述。
4.  **违法活动 (Illegal Activities)**: 关注可能导致负面社会影响的非法行为。
5.  **违背道德 (Violation of Morality)**: 除明确违法外的其他不道德活动。
6.  **隐私与财产损害 (Privacy & Property Damage)**: 关注与隐私和财产损失相关的问题。
7.  **误导性信息 (Misinformation)**: 关注误导性或虚假信息。

模型支持三种核心的评估任务：
* **陈述安全评估 (Statement Safety)**: 直接评估给定的图文内容是否存在安全风险。
* **问题安全评估 (Prompt Safety)**: 评估一个结合了图像的问题是否会引导出不安全或有害的回答。
* **对话安全评估 (Dialogue Safety)**: 在图文对话中，评估回答的安全性。

### 模型与数据集

* **模型仓库**: [thu-coai/ShieldVLM-7B-qwen](https://huggingface.co/thu-coai/ShieldVLM-7B-qwen)
* **数据集仓库**: [thu-coai/ShieldVLM](https://huggingface.co/datasets/thu-coai/ShieldVLM)

### ⚙️ 环境安装

1.  克隆本仓库。
2.  安装所需的依赖包。建议使用虚拟环境。

    ```bash
    pip install -r requirements.txt
    ```
    主要依赖包括 `transformers`, `torch`, `peft`, 和 `trl`。

### 🚀 快速开始

#### 推理 (Inference)

您可以使用 `infer_shieldvlm.sh` 脚本快速对您的数据进行安全评估。

1.  **准备输入数据**:
    参照 `examples/` 目录下的文件格式 (例如 `examples/statement/example_input.json`) 准备您的数据。 每个条目应包含 `id`, `risk_type`, `image` (图片路径) 和 `text`。

2.  **运行推理脚本**:
    修改 `infer_shieldvlm.sh` 脚本中的路径，或直接通过命令行执行 `infer.py`。

    ```bash
    # 示例: 运行陈述安全评估
    MODEL_PATH="thu-coai/ShieldVLM-7B-qwen" # 或您的本地模型路径

    CUDA_VISIBLE_DEVICES=0 python infer.py \
        --category "statement" \
        --model_path ${MODEL_PATH} \
        --input_path ./examples/statement/example_input.json \
        --output_path ./examples/statement/example_output.json
    ```
    * `--category`: 任务类型，可选值为 `statement`, `prompt`, `dialog`。
    * `--model_path`: 您在 Hugging Face 上的模型仓库名或本地模型路径。
    * `--input_path`: 输入的 `json` 文件路径。
    * `--output_path`: 保存评估结果的 `json` 文件路径。

#### 训练 (Training)

ShieldVLM 基于 `Qwen2.5-VL-7B-Instruct` 模型，通过强大的 **LLaMA Factory** 框架进行全量监督式微调 (Supervised Fine-tuning) 训练而来。

1.  **准备数据集**:
    训练集需要根据 **LLaMA Factory** 框架的要求进行编辑。您可以使用我们 [数据集仓库](https://huggingface.co/datasets/thu-coai/ShieldVLM) 中的 `train` 文件夹作为训练集来源。

2.  **配置训练参数**:
    在 `train_code/` 目录下的配置文件 (例如 `train_shieldvlm.yaml`) 中修改您的模型、数据集路径及训练超参数。

3.  **使用 LLaMA Factory 启动训练**:
    执行以下脚本来开始训练。

    ```bash
    bash train_code/run.sh
    ```