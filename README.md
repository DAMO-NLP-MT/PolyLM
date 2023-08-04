<p align="center">
  <a href="https://huggingface.co/DAMO-NLP-MT/polylm-1.7b">PolyLM-1.7B<a> ｜ <a href="https://huggingface.co/DAMO-NLP-MT/polylm-13b">PolyLM-13B<a> ｜ <a href="https://huggingface.co/DAMO-NLP-MT/polylm-chat-13b">PolyLM-13B-Chat<a> ｜ <a href="https://modelscope.cn/studios/damo/demo-polylm-multialpaca-13b/summary">Demo</a> ｜ &nbsp<a href="https://arxiv.org/pdf/2307.06018.pdf">Paper</a>
</p>

**PolyLM** is a polyglot large language model, which is aimed to address the following blanks and limitations in current LLM research, offering a comprehensive and innovative solution to advance this field.

1. **Covering 18 of the most commonly spoken languages**. PolyLM is proficient in the major non-English languages spoken worldwide, such as Spanish, Russian, Arabic, Japanese, Korean, Thai, Indonesian, and Chinese etc. It is a perfect complement to the existing open-source models, including: (1) LLaMA, in which English is predominant among the whole dataset. (2) BLOOM, fails to address languages spoken by significant populations, such as Japanese, Korean and Thai.
2. **Better multingual instruction-following capability**. We propose MULTIALPACA to complement ALPACA and CHINESEALPACA, making LLMs better follow multilingual instructions, particularly those coming from non-native English speakers.
3. **Strong performance**. In comparison with popular multilingual LLMs of the similar model size, PolyLM demonstrates remarkable performance on various tasks, including QA, understanding, and generation.

We opensource PolyLM on both **Modelscope** and **Huggingface** involving two scales (i.e., 1.7B and 13B).

| Model             | Modelscope                                                                                                   | Huggingface                                                                    | 
| :-------------------------- | -----------------------------------------------------------------------------------------------------------: | -----------------------------------------------------------------------------: |
| **PolyLM-1.7B**             | <a href="https://modelscope.cn/models/damo/nlp_polylm_1b7_text_generation/summary">[link]<a>                 | <a href="https://huggingface.co/DAMO-NLP-MT/polylm-1.7b">[link]<a>             |
| **PolyLM-13B**              | <a href="https://modelscope.cn/models/damo/nlp_polylm_13b_text_generation/summary">[link]<a>                 | <a href="https://huggingface.co/DAMO-NLP-MT/polylm-13b">[link]<a>              | 
| **PolyLM-Multialpaca-13B**  | <a href="https://modelscope.cn/models/damo/nlp_polylm_multialpaca_13b_text_generation/summary">[link]<a>     | <a href="https://huggingface.co/DAMO-NLP-MT/polylm-multialpaca-13b">[link]<a>  |
| **PolyLM-Chat-13B**         | <a href="https://www.modelscope.cn/models/damo/nlp_polylm_assistant_13b_text_generation/summary">[link]<a>   | <a href="https://huggingface.co/DAMO-NLP-MT/polylm-chat-13b">[link]<a>         |

## Usage

Find below some example scripts on how to use the model in Modelscope and Transformers.

### Requirements

