<div style="text-align:center">
<img src="https://big-cheng.com/k2/k2.png" alt="k2-logo" width="200"/>
<h2>🏔️ Large Language Model for Geoscience</h2>
</div>

Code and data for paper ***"Learning A Foundation Language Model for Geoscience Knowledge Understanding and Utilization"***

## Introduction

We introduce **K2** (7B), an open-source language model trained by firstly further pretraining LLaMA on collected and cleaned geoscience literatures including geoscience open-access papers and wikipedia pages and secondly fine-tuning with knowledge-intensive instruction tuning data (GeoSignal). Preliminary evaluation using GeoBenchmark (consisting of NPEE and AP Test on Geology, Geography and Environmental science) as benchmark. Compared with several baseline models with similar number of parameters, K2 outperforms the baselines on both the objective tasks and the subjective tasks. 
In this repository, we will share the following code and data

- We release K2 weights in two parts (one can add our delta to the original LLaMA weights, and use `peft_model` with `transformers` to obtain the entire K2 model.)
    - Delta weights after further pretraining with geoscience text corous to comply with the LLaMA model license. 
    - Adapter model weights trained by PEFT (LoRA).
- We release the cora data of GeoSignal under the contraint of DDE, if you want the full version of GeoSignal, you can [email](mailto:davendw@sjtu.edu.cn) the author for further cooperation.
- We release the GeoBenchmark, the first-ever benchmark for the evaluation of the capability of LLMs on geoscience.
- We release the code of further pretrain and instruction tuning of K2.

***The following is the overview of training K2:***
![overview](https://big-cheng.com/k2/overview.png)

## Data

### Further pretrain

Our text corpus for further pretraining on LLaMA-7B consists of 3.9 billion tokens from geoscience papers published in selected highquality journals in earth science and mainly collected by [GAKG](https://gakg.acemap.info/).

**Delta Model on [Huggingface](https://huggingface.co/): [daven3/k2_fp_delta](https://huggingface.co/daven3/k2_fp_delta)**

### Instruction Tuning: GeoSignal

Scientific domain adaptation has two main steps during the instruction tuning. 
- Instruction tuning with general instruction-tuning data, here we use Alpaca-GPT4. 
- Instruction tuning with restructured domain knowledge, which we call expertise instruction tuning. For K2, we use knowledge-intensive instruction data, GeoSignal.

***The following is the illustration of training domain specific language model recipe:***
![recipe](https://big-cheng.com/k2/recipe.png)

- **Adapter Model on [Huggingface](https://huggingface.co/): [daven3/k2_it_adapter](https://huggingface.co/daven3/k2_fp_delta)**
- **Dataset on [Huggingface](https://huggingface.co/): [geosignal](https://huggingface.co/datasets/daven3/geosignal)**

### Benchmark: GeoBenchmark

In GeoBenchmark, we collect 183 multiple-choice questions in NPEE,
and 1,395 in total in AP Test, for objective tasks. Meanwhile, we gather all 939 subjective questions in NPEE to be the subjective tasks set and use 50 to measure the baselines with human evaluation. 

- **Dataset on [Huggingface](https://huggingface.co/): [geobenchmark](https://huggingface.co/datasets/daven3/geobenchmark)**

## Code

### Further Pretrain

The training script is **`run_clm.py`**

```bash
deepspeed --num_gpus=4 run_clm.py --deepspeed ds_config_zero3.json >log 2>&1 &
```

### Instruction tuning

The training script is **`finetune.py`**

- For first step: alignment with human
```bash
python finetune.py --base_model /path/to/checkpoint-30140 --data_path /path/to/alpaca.json --output_dir /path/to/stage/one/model/ --cuda_id 2 --lora_target_modules "q_proj" "k_proj" "v_proj"
```

- For second step: alignment with expert
```bash
python finetune.py --base_model /path/to/checkpoint-30140 --data_path /path/to/geosignal.json --output_dir /path/to/stage/two/model/ --cuda_id 2 --lora_target_modules "q_proj" "k_proj" "v_proj" --resume_from_checkpoint /path/to/stage/one/model/
```

## Why named K2 ?

K2 is original from the name of the second highest mountain in the world, which we believe in the future larger and more powerful geoscience language models will be created. What's more, to train a model to shift to a disciplain with a large domain barrier, we have encountered many difficulties *(collecting corpus, clean academic data, computing power, ...)*, which shares with the fact that climbing K2 is no less difficult than Mount Everest🏔️.

## Contributors

This project was founded by the Acemap at Shanghai Jiao Tong University, the including [Cheng Deng](https://github.com/davendw49), [Tianhang Zhang](https://github.com/zthang), [Zhongmou He](https://github.com/twelfth-star), [Qiyuan Chen](), [Yuanyuan Shi](), [Le Zhou](), supervised by Weinan Zhang, Luoyi Fu, Zhouhan Lin and Junxian He, Xinbing Wang. The whole project is under the support from Chenghu Zhou and Institute of Geographical Science, Natural Resources Research, Chinese Academy of Sciences and [Deep-time Digital Earth Big Science Project](https://www.iugs.org/dde). 


## Acknowledgements

K2 has referred the following open-source projects. We would like to express our gratitude and respect to the researchers of the projects.

- Facebook LLaMA: https://github.com/facebookresearch/llama
- Stanford Alpaca: https://github.com/tatsu-lab/stanford_alpaca
- alpaca-lora by @tloen: https://github.com/tloen/alpaca-lora

K2 is under the support of [Deep-time Digital Earth Big Science Project](https://www.iugs.org/dde). 

## TO-DO
- [ ] Release the full version of GeoSignal.
- [ ] Release the evaluation code over GeoBenchmark.
- [ ] Series of applications with K2.

## License
k2 is a research preview intended for non-commercial use only, subject to the model License of LLaMA, Terms of Use of the data generated by OpenAI. Please contact us if you find any potential violation. The code is released under the Apache License 2.0.

## Citation
If you use the code or data of **k2**, please declare the reference with:

```
@misc{deng2023k2,
      title={Learning A Foundation Language Model for Geoscience Knowledge Understanding and Utilization}, 
      author={Cheng Deng, Tianhang Zhang, Zhongmou He, Qiyuan Chen, Yuanyuan Shi, Le Zhou, Luoyi Fu, Weinan Zhang, Xinbing Wang, Chenghu Zhou, Zhouhan Lin and Junxian He},
      year={2023}
}
```