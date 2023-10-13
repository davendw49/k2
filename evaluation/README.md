# Evaluation for Geoscience LLM

This folder contains the evaluation scripts for language model taking geoscience exams.

```
.
├── multiple_choice_samples_wa.txt # prompt (with 'the answer is') for 5-shot eval
├── multiple_choice_samples.txt # naive prompt for 5-shot eval
└── post_process.py # benchmark preprocessing scripts
└── memtra # Memorizing transformers
└── memtra # Memorizing transformers
```

**We will release end to end version at the end of October, along with Geo-Eval**

## Usage

-> Here is an example:

- For k2 series model with lora
```bash
python run_eval.py --model_name k2_ni --base_model /home/daven/llm/qokori/llama-2023-05-07-15-10/checkpoint/ --lora_weights /home/daven/llm/qokori/qokori-sft/outputs/geo_llama/
```

- For k2 series model without lora
```bash
python run_eval.py --model_name geollama --base_model /home/daven/llm/qokori/llama-2023-05-07-15-10/checkpoint/
```

- For non-k2 series model
```bash 
python run_eval.py --model_name gpt2_xl --base_model gpt2-xl
```