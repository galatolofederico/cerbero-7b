# cerbero-7b Italian LLM 🚀 

> 🔥 Attention! The **new** and **more capable** version of **cerbero-7b** is now **available**!

> 📢 **cerbero-7b** is the first **100% Free** and Open Source **Italian Large Language Model** (LLM) ready to be used for **research** or **commercial applications**.

**Try an online demo [here](https://huggingface.co/spaces/galatolo/chat-with-cerbero-7b)** (quantized demo running on CPU, a lot less powerful than the original cerbero-7b)

<p align="center">
  <img width="300" height="300" src="./README.md.d/cerbero.png">
</p>

Built on top of [**mistral-7b**](https://mistral.ai/news/announcing-mistral-7b/), which outperforms Llama2 13B across all benchmarks and surpasses Llama1 34B in numerous metrics.

**cerbero-7b** is specifically crafted to fill the void in Italy's AI landscape.

A **cambrian explosion** of **Italian Language Models** is essential for building advanced AI architectures that can cater to the diverse needs of the population.

**cerbero-7b**, alongside companions like [**Camoscio**](https://github.com/teelinsan/camoscio) and [**Fauno**](https://github.com/RSTLess-research/Fauno-Italian-LLM), aims to help **kick-start** this **revolution** in Italy, ushering in an era where sophisticated **AI solutions** can seamlessly interact with and understand the intricacies of the **Italian language**, thereby empowering **innovation** across **industries** and fostering a deeper **connection** between **technology** and the **people** it serves.

**cerbero-7b** is released under the **permissive** Apache 2.0 **license**, allowing **unrestricted usage**, even **for commercial applications**.

## Model Evaluation Results 📈

The `cerbero-7b` model has been rigorously evaluated across several benchmarks to demonstrate its proficiency in understanding and generating Italian text. Below are the summarized results showcasing its performance:

### SQuAD-it Evaluation

The Stanford Question Answering Dataset (SQuAD) in Italian (SQuAD-it) is used to evaluate the model's reading comprehension and question-answering capabilities. The following table presents the F1 score and Exact Match (EM) metrics:

| Model                                        | F1 Score | Exact Match (EM) |
|----------------------------------------------|--------------|----------------------|
| **cerbero-7b**                               | **72.55%**   | **55.6%**            |
| Fauno                                        | 44.46%       | 0.00%                |
| Camoscio                                     | 37.42%       | 0.00%                |
| mistral-7b                                   | 15.55%       | 8.50%                |

### EVALITA Benchmark Results

EVALITA benchmarks assess the model's performance in tasks like toxicity detection, irony detection, and sentiment analysis. The table below shows the F1 scores for these tasks:

| Model                                        | Toxicity Detection | Irony Detection | Sentiment Analysis |
|----------------------------------------------|--------------------|-----------------|--------------------|
| **cerbero-7b**                               | **63.04%**         | **48.51%**      | **61.80%**         |
| Fauno                                        | 33.84%             | 39.17%          | 12.23%             |
| Camoscio                                     | 38.18%             | 39.65%          | 13.33%             |
| mistral-7b                                   | 34.16%             | 34.16%          | 12.14%             |


## Why Cerbero? 🤔

The name "Cerbero," inspired by the three-headed dog that guards the gates of the Underworld in Greek mythology, encapsulates the essence of our model, drawing strength from three foundational pillars:

- **Base Model: mistral-7b** 🏗️
  cerbero-7b builds upon the formidable **mistral-7b** as its base model. This choice ensures a robust foundation, leveraging the power and capabilities of a cutting-edge language model.

- **Datasets: Cerbero Dataset** 📚
  The Cerbero Dataset is a groundbreaking collection specifically curated to enhance the proficiency of cerbero-7b in understanding and generating Italian text. This dataset is a product of an innovative method combining dynamic self-chat mechanisms with advanced Large Language Model (LLM) technology. Refer to the [paper](README.md) for more details.

- **Licensing: Apache 2.0** 🕊️
  Released under the **permissive Apache 2.0 license**, cerbero-7b promotes openness and collaboration. This licensing choice empowers developers with the freedom for unrestricted usage, fostering a community-driven approach to advancing AI in Italy and beyond.

## Training Details 🚀

**cerbero-7b** is a **fully fine-tuned** LLM, distinguishing itself from LORA or QLORA fine-tunes. 
The model is trained on an expansive Italian Large Language Model (LLM) using synthetic datasets generated through dynamic self-chat on a large context window of **8192 tokens**

### Dataset Composition 📊

> 📢 Details on the **Cerbero Dataset** will be updated shortly!

### Training Setup ⚙️

**cerbero-7b** is trained on an NVIDIA DGX H100:

- **Hardware:** Utilizing 8xH100 GPUs, each with 80 GB VRAM. 🖥️
- **Parallelism:** DeepSpeed Zero stage 1 parallelism for optimal training efficiency.✨

The model has been trained for **1 epoch**, ensuring a convergence of knowledge and proficiency in handling diverse linguistic tasks.

## Getting Started 🚀

You can load **cerbero-7b** using [🤗transformers](https://huggingface.co/docs/transformers/index)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("galatolo/cerbero-7b")
tokenizer = AutoTokenizer.from_pretrained("galatolo/cerbero-7b")

prompt = """Questa è una conversazione tra un umano ed un assistente AI.
[|Umano|] Come posso distinguere un AI da un umano?
[|AI|]"""

input_ids = tokenizer(prompt, return_tensors='pt').input_ids
with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=128)

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)
```

### GGUF and llama.cpp

**cerbero-7b** is fully **compatibile** with [llama.cpp](https://github.com/ggerganov/llama.cpp)

You can find the **original** and **quantized** versions of **cerbero-7b** in the `gguf` format [here](https://huggingface.co/galatolo/cerbero-7b-gguf/tree/main)

```python
from llama_cpp import Llama
from huggingface_hub import hf_hub_download  

llm = Llama(
    model_path=hf_hub_download(
        repo_id="galatolo/cerbero-7b-gguf",
        filename="ggml-model-Q4_K.gguf",
    ),
    n_ctx=4086,
) 

llm.generate("""Questa è una conversazione tra un umano ed un assistente AI.
[|Umano|] Come posso distinguere un AI da un umano?
[|AI|]""")
```
