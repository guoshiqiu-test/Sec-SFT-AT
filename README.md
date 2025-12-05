# [EMNLP2025] AdaptThink: LLM Can Learn When to Think

<p align="center">
    🤗 <a href="https://huggingface.co/collections/THU-KEG/adaptthink-682a1059aa9f5102c4fa0470" target="_blank">HF Repo</a> • 📃 <a href="https://arxiv.org/abs/2505.13417" target="_blank">Paper</a>
</p>

## 🔍 Table of Contents
- [🤖️ AdaptThink](#adapt_think)
- [⚙️ Released Models](#model)
- [🔥 Training](#training)
- [📊 Evaluation](#evaluation)
- [🧐 Cases](#cases)
- [📝 Citation](#citation)

<a name="adapt_think"></a>
## 🤖️ AdaptThink
We present **AdapThink**, a novel reinforcement learning (RL) algorithm that enables reasoning models to adaptively choose between **Thinking** and **NoThinking** modes according to the difficulty of each input problem, thereby achieving automatic hybrid reasoning. Specifically, the model engages in thinking only when the problem is determined to be challenging; for other simple questions, it will bypass the thinking process and directly produce a concise final solution. This approach substantially reduces inference costs while further improving overall performance.
<img width="1327" alt="image" src="https://github.com/user-attachments/assets/35f62f31-3210-4f11-98cb-06d73e0231e8" />



<a name="model"></a>
## ⚙️ Released Models

### All Available Datasets and Models
We apply the AdaptThink algorithm on DeepSeek-R1-Distill-Qwen-1.5B with $\delta$ from 0 to 0.1, and DeepSeek-R1-Distill-Qwen-7B with $\delta=0.05$. A larger $\large$ results in a higher proportion of NoThinking responses, which reduces more inference costs but also diminishes the resultant improvement in accuracy.

All the trained models are available on HuggingFace. 


| Name | HF Repo |
|---|---|
| AdaptThink-1.5B-delta0 | [🤗 HF Repo](https://huggingface.co/THU-KEG/AdaptThink-1.5B-delta0) |
| AdaptThink-1.5B-delta0.01 | [🤗 HF Repo](https://huggingface.co/THU-KEG/AdaptThink-1.5B-delta0.01) |
| AdaptThink-1.5B-delta0.02 | [🤗 HF Repo](https://huggingface.co/THU-KEG/AdaptThink-1.5B-delta0.02) |
| AdaptThink-1.5B-delta0.05 | [🤗 HF Repo](https://huggingface.co/THU-KEG/AdaptThink-1.5B-delta0.05) |
| AdaptThink-1.5B-delta0.075 | [🤗 HF Repo](https://huggingface.co/THU-KEG/AdaptThink-1.5B-delta0.075) |
| AdaptThink-1.5B-delta0.1 | [🤗 HF Repo](https://huggingface.co/THU-KEG/AdaptThink-1.5B-delta0.1) |
| AdaptThink-7B-delta0.05 | [🤗 HF Repo](https://huggingface.co/THU-KEG/AdaptThink-7B-delta0.05) |

<a name="training"></a>
## 🔥 Training

Our training code is based on [VeRL](https://github.com/volcengine/verl) framework.

### 1. Creating Environment
We use [vLLM](https://github.com/vllm-project/vllm) 0.8.2, which supports [flash-attention](https://github.com/Dao-AILab/flash-attention). 
```
conda create -n adapt_think python=3.10
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### 2. Check the chat template in HF models
After you download DeepSeek models, you should check `chat_template` in `tokenizer_config.json` to ensure the template ends with `<｜Assistant｜><think>\\n`, otherwise there will be bugs when running our code.

### 3. Pre-sampling from reference models
First, we need to pre-sample multiple responses from the reference model for each training problem to evaluate its instance-level accuracy. The sampling process will take several hours. For convenience, we have released our post-processed results in `./data/train/ref_results`, which can be directly used for training.
```
# Initialize VLLM server. You can start multiple servers to accelerate pre-sampling.
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --served_model_name DeepSeek-R1-Distill-Qwen-1.5B --tensor_parallel_size 4

# Sampling 16 responses for each training problem. 
python src/presampling_ref_responses.py --K 16 --dataset_path ./data/train/deepscaler.json --model_name DeepSeek-R1-Distill-Qwen-1.5B --max_tokens 16384

# Postprocess to get instance-level accuracy
python src/postprocess_ref_results.py --input_path ./data/train/ref_presampling/DeepSeek-R1-Distill-Qwen-1.5B_deepscaler_n0_K16_len16384.json --output_path ./data/train/ref_results/DeepSeek-R1-Distill-Qwen-1.5B_deepscaler_K16_len16384.json
```

### 4. Preprocess training and test Datasets
```
bash scripts/preprocess_dataset.sh
```

### 5. Training
The training context size, batch size, and the learning rate are set to 16K, 128, and 2e-6, respectively. We train the models for 1 epoch, which is 314 steps in total. For the 1.5B model, we use one 8\*H800 node and cost about 32 hours. For the 7B model, we use four 8\*H800 nodes and cost about 28 hours. Finally, we select the checkpoints on 300 and 150 steps for the 1.5B and 7B models, respectively, where the models' accuracy and response lengths achieve a good balance.

To facilitate the training process, you can set a larger learning rate, such as 5e-5. However, it may make the training more unstable.
```
# 1.5b, single-node
bash scripts/run_adapt_think_1.5b_deepscaler_16k_delta0.05_btz128_lr2e-6.sh

# 7b, single-node
bash scripts/run_adapt_think_7b_deepscaler_16k_delta0.05_btz128_lr2e-6.sh

# 7b, multi-node
bash submit_mpi.sh scripts/run_adapt_think_7b_deepscaler_16k_delta0.05_btz128_lr2e-6_multinode.sh
```


<a name="evaluation"></a>
## 📊 Evaluation
During training, VeRL will automatically evaluate on you selected test sets for every `trainer.test_freq` step.

We also provide additional scripts for evaluation.

```
# convert checkpoint to HF model
bash scripts/convert_to_hf.sh

# eval
bash scripts/run_eval_verl_hf.sh
```

You can also evaluate downloaded HF models by running:
```
bash scripts/run_eval_hf.sh
```

We list our evaluation results as follows:
#### 1.  Comparison with existing methods for efficient reasoning on mathematics datasets
<img width="1447" alt="image" src="https://github.com/user-attachments/assets/53592ec3-17d9-4c4b-99ee-1868b5c82238" />

#### 2. Nothinking responses ratio and accuracy across different difficulty levels on MATH500
<img width="1462" alt="image" src="https://github.com/user-attachments/assets/cc2de266-b67a-47ab-835d-9bce922b13fc" />

#### 3. Comparison of different $\delta$ values
<img width="1444" alt="image" src="https://github.com/user-attachments/assets/41c86f73-68f8-4d71-ac75-2033c43b964b" />

#### 4. Evaluation results on MMLU
<img width="500" alt="image" src="https://github.com/user-attachments/assets/fdd20adc-b879-4105-8420-0851944c507f" />

## 🧐 Cases
### Simple problem
![image](https://github.com/user-attachments/assets/1f6aaa1c-a1c8-4d49-92c5-2e1b219a643a)
![image](https://github.com/user-attachments/assets/1c3d2dbc-5a98-4066-a8a8-90afff0fc7a3)


### Difficult problem
![image](https://github.com/user-attachments/assets/500a0377-3be4-48a2-b5a0-a98c7d228a30)


<a name="citation"></a>
## 📝 Citation

If you find our work useful, please consider citing AdaptThink:

```
@article{zhang2025adapt_think,
  title = {AdaptThink: LLM Can Learn When to Think} 
  author={Jiajie Zhang and Nianyi Lin and Lei Hou and Ling Feng and Juanzi Li},
  journal={arXiv preprint arXiv: 2505.13417},
  url={https://arxiv.org/abs/2505.13417}
  year={2025}
}
```
