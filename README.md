<p align="center">
  <a href="https://huggingface.co/DAMO-NLP-MT/polylm-1.7b">PolyLM-1.7B<a> ｜ <a href="https://huggingface.co/DAMO-NLP-MT/polylm-13b">PolyLM-13B<a> ｜ <a href="https://huggingface.co/DAMO-NLP-MT/polylm-chat-13b">PolyLM-13B-Chat<a> ｜ <a href="https://modelscope.cn/studios/damo/demo-polylm-multialpaca-13b/summary">Demo</a> ｜ &nbsp<a href="https://arxiv.org/pdf/2307.06018.pdf">Paper</a>
</p>

**PolyLM** is a polyglot large language model, which is aimed to address the following blanks and limitations in current LLM research, offering a comprehensive and innovative solution to advance this field.

1. **Covering 18 of the most commonly spoken languages**. PolyLM is proficient in the major non-English languages spoken worldwide, such as Spanish, Russian, Arabic, Japanese, Korean, Thai, Indonesian, and Chinese etc. It is a perfect complement to the existing open-source models, including: (1) LLaMA, in which English is predominant among the whole dataset. (2) BLOOM, fails to address languages spoken by significant populations, such as Japanese, Korean and Thai.
2. **Better multingual instruction-following capability**. We propose MULTIALPACA to complement ALPACA and CHINESEALPACA, making LLMs better follow multilingual instructions, particularly those coming from non-native English speakers.
3. **Strong performance**. In comparison with popular multilingual LLMs of the similar model size, PolyLM demonstrates remarkable performance on various tasks, including QA, understanding, and generation.

We opensource PolyLM on both **Modelscope** and **Huggingface** involving two scales (i.e., 1.7B and 13B).

| Model             | Modelscope                                                                                                             | Huggingface                                                                    | 
| :-------------------------- | -----------------------------------------------------------------------------------------------------------: | -----------------------------------------------------------------------------: |
| **PolyLM-1.7B**             | <a href="https://modelscope.cn/models/damo/nlp_polylm_1b7_text_generation/summary">[link]<a>                 | <a href="https://huggingface.co/DAMO-NLP-MT/polylm-1.7b">[link]<a>             |
| **PolyLM-13B**              | <a href="https://modelscope.cn/models/damo/nlp_polylm_13b_text_generation/summary">[link]<a>                 | <a href="https://huggingface.co/DAMO-NLP-MT/polylm-13b">[link]<a>              | 
| **PolyLM-Multialpaca-13B**  | <a href="https://modelscope.cn/models/damo/nlp_polylm_multialpaca_13b_text_generation/summary">[link]<a>     | <a href="https://huggingface.co/DAMO-NLP-MT/polylm-multialpaca-13b">[link]<a>  |
| **PolyLM-Chat-13B**         | <a href="https://www.modelscope.cn/models/damo/nlp_polylm_assistant_13b_text_generation/summary">[link]<a>   | <a href="https://huggingface.co/DAMO-NLP-MT/polylm-chat-13b">[link]<a>         |

## Usage

Find below some example scripts on how to use the model in Modelscope and Transformers.

### Requirements

```bash
git clone https://github.com/modelscope/modelscope
cd modelscope
pip install .
```

```bash
pip install transformers==4.31.0 accelerate einops
```

### Transformers
There are example scripts using PolyLM-13B, PolyLM-Multialpaca-13B, and PolyLM-Chat-13B for inference.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# PolyLM-1.7B
# model_path = "DAMO-NLP-MT/polylm-1.7b"
# PolyLM-13B
model_path = "DAMO-NLP-MT/polylm-13b"
# PolyLM-Multialpaca-13B
# model_path = "DAMO-NLP-MT/polylm-multialpaca-13b"
# PolyLM-Chat-13B
# model_path = "DAMO-NLP-MT/polylm-chat-13b"

tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()

# PolyLM-13B/PolyLM-1.7B
input_doc = f"Beijing is the capital of China.\nTranslate this sentence from English to Chinese."
# PolyLM-Multialpaca-13B
# input_doc = f"Beijing is the capital of China.\nTranslate this sentence from English to Chinese.\n\n"
# PolyLM-Chat-13B
# input_doc = f"Beijing is the capital of China.\nTranslate this sentence from English to Chinese."
# input_doc = "<|user|>\n" + f"{input_doc}\n" + "<|assistant|>\n"

inputs = tokenizer(input_doc, return_tensors="pt")

generate_ids = model.generate(
  inputs.input_ids,
  attention_mask=inputs.attention_mask,
  do_sample=False,
  num_beams=4,
  max_length=128,
  early_stopping=True
)

decoded = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f">>> {decoded}")
### results
### Beijing is the capital of China.\nTranslate this sentence from English to Chinese.\\n北京是中华人民共和国的首都。\n ...
```

### Modelscope
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download

# PolyLM-13B
model_id = 'damo/nlp_polylm_13b_text_generation'
revision = 'v1.0.3'

# PolyLM-Multialpaca-13B
# model_id = 'damo/nlp_polylm_multialpaca_13b_text_generation'
# revision = 'v1.0.0'

# PolyLM-Chat-13B
# model_id = 'damo/nlp_polylm_assistant_13b_text_generation'
# revision = 'v1.0.0'

model_dir = snapshot_download(model_id, revision)

# PolyLM-13B/PolyLM-1.7B
input_text = f"Beijing is the capital of China.\nTranslate this sentence from English to Chinese."
# PolyLM-Multialpaca-13B
# input_text = f"Beijing is the capital of China.\nTranslate this sentence from English to Chinese.\n\n"
# PolyLM-Chat-13B
# input_text = f"Beijing is the capital of China.\nTranslate this sentence from English to Chinese."
# input_text = "<|user|>\n" + f"{input_text}\n" + "<|assistant|>\n"

kwargs = {"do_sample": False, "num_beams": 4, "max_new_tokens": 128, "early_stopping": True, "eos_token_id": 2}
pipeline_ins = pipeline(Tasks.text_generation, model=model_dir)

result = pipeline_ins(input_text, **kwargs)
print(result['text'])
```

## Demo
We have provided a web [demo](https://modelscope.cn/studios/damo/demo-polylm-multialpaca-13b/summary) based on gradio for you to experience. Similarly, you can run [polylm_web_demo_gradio.py](https://huggingface.co/DAMO-NLP-MT/polylm-chat-13b/blob/main/polylm_web_demo_gradio.py) to deploy a web demo locally (it requires 48GB of GPU memory, i.e., 2 V100 GPUs or a single A100 GPU).

## Bibtex
``` bibtex
@misc{wei2023polylm,
      title={PolyLM: An Open Source Polyglot Large Language Model}, 
      author={Xiangpeng Wei and Haoran Wei and Huan Lin and Tianhao Li and Pei Zhang and Xingzhang Ren and Mei Li and Yu Wan and Zhiwei Cao and Binbin Xie and Tianxiang Hu and Shangjie Li and Binyuan Hui and Bowen Yu and Dayiheng Liu and Baosong Yang and Fei Huang and Jun Xie},
      year={2023},
      eprint={2307.06018},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
