# cerbero-7b Italian LLM 🚀 

> 📢 **Cerbero-7b** is the first **100% Free** and Open Source **Italian Large Language Model** (LLM) ready to be used for **research** or **commercial applications**.

<p align="center">
  <img width="300" height="300" src="./README.md.d/cerbero.png">
</p>

Built on [**mistral-7b**](https://mistral.ai/news/announcing-mistral-7b/), which outperforms Llama2 13B across all benchmarks and surpasses Llama1 34B in numerous metrics.

**cerbero-7b** is specifically crafted to fill the void in Italy's AI landscape.

A **cambrian explosion** of **Italian Language Models** is essential for building advanced AI architectures that can cater to the diverse needs of the population.

**cerbero-7b**, alongside companions like [**Camoscio**](https://github.com/teelinsan/camoscio) and [**Fauno**](https://github.com/RSTLess-research/Fauno-Italian-LLM), aims to help **kick-start** this **revolution** in Italy, ushering in an era where sophisticated **AI solutions** can seamlessly interact with and understand the intricacies of the **Italian language**, thereby empowering **innovation** across **industries** and fostering a deeper **connection** between **technology** and the **people** it serves.

**cerbero-7b** is released under the **permissive** Apache 2.0 **license**, allowing **unrestricted usage**, even **for commercial applications**.

## Why Cerbero? 🤔

The name "Cerbero," inspired by the three-headed dog that guards the gates of the Underworld in Greek mythology, encapsulates the essence of our model, drawing strength from three foundational pillars:

- **Base Model: mistral-7b** 🏗️
  cerbero-7b builds upon the formidable **mistral-7b** as its base model. This choice ensures a robust foundation, leveraging the power and capabilities of a cutting-edge language model.

- **Datasets: Fauno Dataset** 📚
  Utilizing the comprehensive **fauno dataset**, cerbero-7b gains a diverse and rich understanding of the Italian language. The incorporation of varied data sources contributes to its versatility in handling a wide array of tasks.

- **Licensing: Apache 2.0** 🕊️
  Released under the **permissive Apache 2.0 license**, cerbero-7b promotes openness and collaboration. This licensing choice empowers developers with the freedom for unrestricted usage, fostering a community-driven approach to advancing AI in Italy and beyond.

## Training Details 🚀

cerbero-7b is **fully fine-tuned**, distinguishing itself from LORA or QLORA fine-tunes. 
The model is trained on an expansive Italian Large Language Model (LLM) using synthetic datasets generated through dynamic self-chat on a large context window of **8192 tokens**

### Dataset Composition 📊

We employed the [Fauno training dataset](https://github.com/RSTLess-research/Fauno-Italian-LLM). The training data covers a broad spectrum, incorporating:

- **Medical Data:** Capturing nuances in medical language. 🩺
- **Technical Content:** Extracted from Stack Overflow to enhance the model's understanding of technical discourse. 💻
- **Quora Discussions:** Providing valuable insights into common queries and language usage. ❓
- **Alpaca Data Translation:** Italian-translated content from Alpaca contributes to the model's language richness and contextual understanding. 🦙

### Training Setup ⚙️

cerbero-7b is trained on an NVIDIA DGX H100:

- **Hardware:** Utilizing 8xH100 GPUs, each with 80 GB VRAM. 🖥️
- **Parallelism:** DeepSpeed Zero stage 1 parallelism for optimal training efficiency.✨

The model has been trained for **3 epochs**, ensuring a convergence of knowledge and proficiency in handling diverse linguistic tasks.

## Getting Started 🚀

You can load cerbero-7b using [🤗transformers](https://huggingface.co/docs/transformers/index)


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