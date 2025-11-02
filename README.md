Gemma-3-1B Medical Reasoning (GRPO Fine-tune)
---
base_model: unsloth/gemma-3-1b-it
tags:
- text-generation
- instruction-following
- medical-reasoning
- grpo
- transformers
- unsloth
- gemma-3
- generated_from_trainer
language:
- en
datasets:
- FreedomIntelligence/medical-o1-reasoning-SFT
license: apache-2.0
---

# Gemma-3-1B Medical Reasoning (GRPO Fine-tune)

This model is a fine-tuned version of Google's `unsloth/gemma-3-1b-it`, specifically adapted for medical reasoning tasks. It has been trained using Unsloth's efficient training library with Group Relative Policy Optimization (GRPO) to enhance its ability to provide step-by-step reasoning for medical problems.

View the model on Hugging Face - https://huggingface.co/naazimsnh02/gemma-3-finetune

## Model Details

- **Developed by:** naazimsnh02
- **Base Model:** [unsloth/gemma-3-1b-it](https://huggingface.co/unsloth/gemma-3-1b-it)
- **Fine-tuning Method:** Group Relative Policy Optimization (GRPO) with LoRA.
- **Training Frameworks:** Unsloth, TRL, Transformers
- **Language:** English
- **License:** apache-2.0

## Training Data

The model was fine-tuned on the `en` split of the [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset. This dataset is designed to improve the medical reasoning capabilities of large language models and contains complex medical questions with detailed "Complex_CoT" (Chain of Thought) and "Response" fields.

### Data Preprocessing

The dataset was formatted to follow a specific conversational structure with a system prompt that instructs the model to act as a medical reasoning assistant. Special tokens were introduced to delineate the reasoning and final answer sections:

-   `<start_reasoning>` and `<end_reasoning>` for the step-by-step analysis.
-   `<SOLUTION>` and `</SOLUTION>` for the final conclusion.

## Training Procedure

The model was trained for 300 steps using the Unsloth library, which enables faster training and reduced memory usage. The training process utilized the following key components:

-   **`FastModel` from Unsloth:** This allowed for efficient loading and preparation of the base model.
-   **LoRA (Low-Rank Adaptation):** To make the fine-tuning process more efficient, LoRA adapters were added to the model. This involves training only a small number of parameters, significantly reducing the computational cost. The following LoRA parameters were used:
    -   `r`: 8
    -   `lora_alpha`: 8
    -   `lora_dropout`: 0
-   **GRPO (Group Relative Policy Optimization):** This reinforcement learning technique was used to refine the model's reasoning abilities. GRPO evaluates multiple generated responses and uses their relative ranking to update the model's policy, encouraging it to produce more accurate and well-formatted outputs.

### Reward Functions

The GRPO training was guided by a set of custom reward functions designed to enforce a specific output structure and correctness:

1.  **`match_format_exactly`:** Rewarded the model for perfectly adhering to the predefined reasoning and solution format.
2.  **`match_format_approximately`:** Provided partial rewards for including the special tokens, even if the overall format was not perfect.
3.  **`check_answer`:** Assessed the correctness of the final answer through exact and approximate string matching of key medical terms.

### Training Hyperparameters

The training was configured with the following `GRPOConfig` settings:

-   **`learning_rate`:** 3e-6
-   **`per_device_train_batch_size`:** 6
-   **`gradient_accumulation_steps`:** 4
-   **`num_generations`:** 6
-   **`max_prompt_length`:** 768
-   **`max_completion_length`:** 256
-   **`optimizer`:** `adamw_torch_fused`

### Hardware

The model was trained on a **free Tesla T4 Kaggle instance**.

## How to Use

To use this model for inference, you can load it using the `unsloth` library.

python
from unsloth import FastModel
import torch

model, tokenizer = FastModel.from_pretrained(
    model_name = "naazimsnh02/gemma-3-finetune", # Your Hugging Face model repo
    max_seq_length = 1024,
    load_in_4bit = True, # Use 4bit quantization for faster inference
)

# Define the system prompt with the special tokens
system_prompt = \
"""You are a medical reasoning assistant.
Analyze the medical problem carefully and provide your step-by-step reasoning.
Place your reasoning between <start_reasoning> and <end_reasoning>.
Then, provide your final medical conclusion between <SOLUTION> and </SOLUTION>."""

# Format the prompt using the chat template
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": "A 33-year-old woman is brought to the emergency department 15 minutes after being stabbed in the chest with a screwdriver. Given her vital signs of pulse 110/min, respirations 22/min, and blood pressure 90/65 mm Hg, along with the presence of a 5-cm deep stab wound at the upper border of the 8th rib in the left midaxillary line, which anatomical structure in her chest is most likely to be injured?"},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
)

# Generate the response
from transformers import TextStreamer
inputs = tokenizer(text, return_tensors="pt").to("cuda")
_ = model.generate(
    **inputs,
    max_new_tokens = 1024,
    temperature = 1.0,
    top_p = 0.95,
    top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
    pad_token_id = tokenizer.eos_token_id # Set pad token for open-ended generation
)

## Intended Use and Limitations
This model is intended for research and educational purposes to explore the application of large language models in medical reasoning. It can be used to generate step-by-step analyses of medical problems.
This is not a medical device and should not be used for any of the following:
Medical Advice: It is not a substitute for professional medical advice, diagnosis, or treatment.
Clinical Use: It is not intended for use in any clinical decision-making process.
Patient Care: It should not be used to guide patient care in any way.
The model's knowledge is limited to the information present in its training data and it may produce inaccurate or outdated information. Always consult with a qualified healthcare professional for any medical concerns.

- **Developed by:** naazimsnh02
- **License:** apache-2.0
- **Finetuned from model :** unsloth/gemma-3-1b-it

This gemma3_text model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
