---
title: DeepSeek Text Generation Demo
emoji: 📚
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

# DeepSeek Text Generation Demo

This is a simple text generation demo using the DeepSeek SmolLM2 language model with a Gradio interface. It uses less than 1B parameters in total

## Description

This application provides a web interface for text generation using the SmolLM2 language model. Users can input a prompt and adjust various generation parameters to control the output.

## Features

- Interactive web interface built with Gradio
- Adjustable generation parameters:
  - Maximum new tokens (1-150)
  - Temperature (0.1-2.0)
  - Top-K sampling (1-100)
- Real-time text generation

## Usage

1. Enter your prompt in the text input field
2. Adjust the generation parameters (optional):
   - **Max New Tokens**: Controls the length of the generated text
   - **Temperature**: Controls randomness (higher = more creative, lower = more focused)
   - **Top-K**: Controls diversity of word choices
3. Click submit to generate text

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
 ## Run the application:
   ```bash
   python app.py
   ```
   The interface will be available at `http://localhost:7860`


## Train the model:
```bash
python train.py
```


# Model details
SmolLM2 is a language model designed for [add your model's specific details here]. The model uses the [specify tokenizer] tokenizer from Hugging Face's transformers library.

## Llama 2 Architecture

![Llama 2 Architecture](./static/DeepSeekModel.jpg)
Read for more details.

# Model Summary
 
```bash 
Model: DeepSeekTransfomerModel(
  (embedding): Embedding(49152, 768)
  (layers): ModuleList(
    (0-29): 30 x DeepSeekBlock(
      (attn_norm): RMSNorm((768,), eps=None, elementwise_affine=True)
      (attn): MultiHeadLatentAttention(
        (kv_proj_d): Linear(in_features=768, out_features=192, bias=True)
        (q_proj_d): Linear(in_features=768, out_features=192, bias=True)
        (k_proj_u): Linear(in_features=192, out_features=384, bias=True)
        (q_proj_u): Linear(in_features=192, out_features=384, bias=True)
        (v_proj_u): Linear(in_features=192, out_features=768, bias=True)
        (rope_k): Linear(in_features=768, out_features=384, bias=True)
        (rope_q): Linear(in_features=192, out_features=384, bias=True)
        (rotatry_embedding): CustomLlamaRotaryEmbedding()
        (o_proj): Linear(in_features=768, out_features=768, bias=True)
      )
      (ffn_norm): RMSNorm((768,), eps=None, elementwise_affine=True)
      (ffn): DeepSeekFFN(
        (shared_experts): ModuleList(
          (0): DeepSeekExpert(
            (gate): Linear(in_features=768, out_features=1536, bias=False)
            (up): Linear(in_features=768, out_features=1536, bias=False)
            (down): Linear(in_features=1536, out_features=768, bias=False)
            (act_fn): SiLU()
          )
        )
        (routed_experts): ModuleList(
          (0-6): 7 x DeepSeekExpert(
            (gate): Linear(in_features=768, out_features=1536, bias=False)
            (up): Linear(in_features=768, out_features=1536, bias=False)
            (down): Linear(in_features=1536, out_features=768, bias=False)
            (act_fn): SiLU()
          )
        )
        (routing): Linear(in_features=768, out_features=7, bias=True)
      )
    )
  )
  (norm): RMSNorm((768,), eps=None, elementwise_affine=True)
  (head): Linear(in_features=768, out_features=49152, bias=True)
)
===============================================================================================================================================================================
Layer (type:depth-idx)                             Input Shape               Output Shape              Param #                   Mult-Adds                 Param %
===============================================================================================================================================================================
DeepSeekTransfomerModel                            [8, 64]                   [8, 64, 49152]            --                        --                         -3.88%
├─Embedding: 1-1                                   [8, 64]                   [8, 64, 768]              37,748,736                301,989,888                 3.88%
├─ModuleList: 1-2                                  --                        --                        --                        --                             --
│    └─DeepSeekBlock: 2-1                          [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-1                           [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-2          [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-3                           [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-4                       [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-2                          [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-5                           [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-6          [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-7                           [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-8                       [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-3                          [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-9                           [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-10         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-11                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-12                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-4                          [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-13                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-14         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-15                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-16                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-5                          [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-17                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-18         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-19                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-20                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-6                          [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-21                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-22         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-23                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-24                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-7                          [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-25                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-26         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-27                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-28                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-8                          [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-29                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-30         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-31                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-32                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-9                          [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-33                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-34         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-35                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-36                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-10                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-37                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-38         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-39                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-40                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-11                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-41                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-42         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-43                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-44                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-12                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-45                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-46         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-47                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-48                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-13                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-49                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-50         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-51                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-52                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-14                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-53                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-54         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-55                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-56                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-15                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-57                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-58         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-59                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-60                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-16                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-61                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-62         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-63                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-64                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-17                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-65                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-66         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-67                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-68                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-18                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-69                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-70         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-71                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-72                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-19                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-73                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-74         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-75                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-76                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-20                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-77                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-78         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-79                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-80                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-21                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-81                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-82         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-83                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-84                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-22                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-85                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-86         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-87                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-88                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-23                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-89                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-90         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-91                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-92                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-24                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-93                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-94         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-95                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-96                      [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-25                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-97                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-98         [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-99                          [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-100                     [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-26                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-101                         [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-102        [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-103                         [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-104                     [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-27                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-105                         [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-106        [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-107                         [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-108                     [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-28                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-109                         [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-110        [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-111                         [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-112                     [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-29                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-113                         [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-114        [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-115                         [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-116                     [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
│    └─DeepSeekBlock: 2-30                         [8, 64, 768]              [8, 64, 768]              --                        --                             --
│    │    └─RMSNorm: 3-117                         [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─MultiHeadLatentAttention: 3-118        [8, 64, 768]              [8, 64, 768]              1,551,744                 12,413,952                  0.16%
│    │    └─RMSNorm: 3-119                         [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
│    │    └─DeepSeekFFN: 3-120                     [8, 64, 768]              [8, 64, 768]              28,316,942                3,652,233,272               2.91%
├─RMSNorm: 1-3                                     [8, 64, 768]              [8, 64, 768]              768                       6,144                       0.00%
├─Linear: 1-4                                      [8, 64, 768]              [8, 64, 49152]            37,797,888                302,383,104                 3.89%
===============================================================================================================================================================================
Total params: 971,654,052
Trainable params: 971,654,052
Non-trainable params: 0
Total mult-adds (G): 110.54
===============================================================================================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 2237.47
Params size (MB): 3886.62
Estimated Total Size (MB): 6124.09
===============================================================================================================================================================================

```

# Training Logs
## Training with 5000 steps (Starting from step 0)
```bash
(venv) gitesh.grover@Giteshs-MacBook-Pro ai-era-assignment13 % python train.py
Device: mps
Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 341.22it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 346746.91it/s]
gradient_accumalate_steps: 8
Updating MLP bias
outputs: torch.Size([8, 64, 49152]), loss: 11.384061813354492
Epoch: 0, Step: 0, Batch(micro): 0, Batch (considering grad accum): 0,  Loss: 11.3841, Time: 5.09s, Token/s: 100.68
Saved checkpoint at step 0
What is Gravity?iates movie socializing Britannicalinux compliment Dol glance Pepper tears index wasted Renewablett aromaosc revenue checkout Rob utilitycriptesting BCEhoodtranslatematch nicknameLetter blurredbeck
outputs: torch.Size([8, 64, 49152]), loss: 11.48910140991211
Epoch: 0, Step: 1, Batch(micro): 1, Batch (considering grad accum): 0,  Loss: 11.4891, Time: 4.80s, Token/s: 106.75
outputs: torch.Size([8, 64, 49152]), loss: 11.449575424194336
Epoch: 0, Step: 2, Batch(micro): 2, Batch (considering grad accum): 0,  Loss: 11.4496, Time: 5.21s, Token/s: 98.35
outputs: torch.Size([8, 64, 49152]), loss: 11.459564208984375
Epoch: 0, Step: 3, Batch(micro): 3, Batch (considering grad accum): 0,  Loss: 11.4596, Time: 4.93s, Token/s: 103.94
outputs: torch.Size([8, 64, 49152]), loss: 11.449928283691406
Epoch: 0, Step: 4, Batch(micro): 4, Batch (considering grad accum): 0,  Loss: 11.4499, Time: 3.52s, Token/s: 145.28
outputs: torch.Size([8, 64, 49152]), loss: 11.487056732177734
Epoch: 0, Step: 5, Batch(micro): 5, Batch (considering grad accum): 0,  Loss: 11.4871, Time: 3.24s, Token/s: 158.13
outputs: torch.Size([8, 64, 49152]), loss: 11.492469787597656
Epoch: 0, Step: 6, Batch(micro): 6, Batch (considering grad accum): 0,  Loss: 11.4925, Time: 3.30s, Token/s: 155.35
outputs: torch.Size([8, 64, 49152]), loss: 11.384774208068848
Epoch: 0, Step: 7, Batch(micro): 7, Batch (considering grad accum): 0,  Loss: 11.3848, Time: 8.66s, Token/s: 59.11
outputs: torch.Size([8, 64, 49152]), loss: 10.7886381149292
Epoch: 0, Step: 8, Batch(micro): 8, Batch (considering grad accum): 1,  Loss: 10.7886, Time: 7.02s, Token/s: 72.96
:
:
outputs: torch.Size([8, 64, 49152]), loss: 7.737437725067139
Epoch: 0, Step: 3499, Batch(micro): 3499, Batch (considering grad accum): 437,  Loss: 7.7374, Time: 2.27s, Token/s: 225.76
Updating MLP bias
outputs: torch.Size([8, 64, 49152]), loss: 7.447840213775635
Epoch: 0, Step: 3500, Batch(micro): 3500, Batch (considering grad accum): 437,  Loss: 7.4478, Time: 2.59s, Token/s: 197.81
Saved checkpoint at step 3500
What is Gravity?,. by! in
, can.,.
 and,
, of of your.,, in., in you of with.
outputs: torch.Size([8, 64, 49152]), loss: 7.890702724456787

::
::

Epoch: 0, Step: 4497, Batch(micro): 4497, Batch (considering grad accum): 562,  Loss: 8.4338, Time: 2.78s, Token/s: 184.00
outputs: torch.Size([8, 64, 49152]), loss: 7.812904357910156
Epoch: 0, Step: 4498, Batch(micro): 4498, Batch (considering grad accum): 562,  Loss: 7.8129, Time: 2.55s, Token/s: 201.02
outputs: torch.Size([8, 64, 49152]), loss: 7.534916877746582
Epoch: 0, Step: 4499, Batch(micro): 4499, Batch (considering grad accum): 562,  Loss: 7.5349, Time: 2.58s, Token/s: 198.14
Updating MLP bias
outputs: torch.Size([8, 64, 49152]), loss: 7.738061428070068
Epoch: 0, Step: 4500, Batch(micro): 4500, Batch (considering grad accum): 562,  Loss: 7.7381, Time: 2.14s, Token/s: 239.67
Saved checkpoint at step 4500
What is Gravity?
 it a. into that the " how. by- like a.. such
., to ( and ( the: this
 these like
:
::

```


