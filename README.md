
# Seqthetic 

This tool generates synthetic sequence data to help test various ideas in pretraining sequence models with synthetic data. It is used in meta-language repo (Coming Soon).

Features:
1. **Diversity**: Supports generating data following various patterns including fractional Brownian Motion(fbm), [LIME](https://arxiv.org/pdf/2101.06223)(TODO), [TILT](https://arxiv.org/abs/2004.14601)(TODO) and [synthetic pretraining tasks](https://arxiv.org/abs/2206.10139) etc.
2. **Spec-Driven**: Everything about the dataset is described by a spec, which helps with documenting each ablation and high-level manipulation. 
## Installation 

```
pip install -e .

```
## Usage
The `syn_data.py` script allows you to generate synthetic datasets with unique domain configurations, including masking and domain promoter sequences. The domain promoter sequence signals the start of specific domain tasks within the dataset and is particularly useful for multi-domain or task-specific pretraining. The script takes a `.json` configuration file to specify dataset parameters.

#### Example Command

Run the following command to generate synthetic data with `syn_data.py`:

```bash
python tests/syn_data.py --num_token 5M --data_config configs/example_config.json

```

Example Configuration JSON
Below is an example .json configuration file for generating synthetic data with the identity domain:
```
{
    "name": "identity",
    "vocab_size": 10000,
    "sequence_length_min": 256,
    "sequence_length_max": 1024,
}
```
## Concepts 

### Domain

Domains are different kinds of data generation methods. For example, the `Identity` domain simply copies sequence two times, while the `Duplicate` domain requires models to repeat every token in the source sequence multiple times. 

Domains are specified with classes inherited from `DomainSpec`. They are under the `seqthetic.domains` directory. You can call `make_sequences` on a domain spec to generate sequences with designated token count and seed. A spec is fully determined by its parameters and the seed you pass into the `make_sequence` method.

Some domains require a `Vocabulary` class. This class specifies the vocabulary that the sequence tokens are sampled from. The main parameters of `Vocabulary` are `num_vocab` and `prob`, where `prob` signifies the weights of different tokens. Setting to a string `"uniform"` means each token has equal opportunity to be sampled, while `Zipf()` means a long tailed distribution, with some tokens claiming higher weights than others.
 
Many domains are of a mapping nature: we want models to learn the transformation of sequences, not sequence itself. These domains have use `delimiter_tokens` to separate input and output. 


# Pretrain

## Install Deepspeed (For reference)

```bash
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
conda env create -n deepspeed -f environment.yml --force
conda activate deepspeed
DS_BUILD_OPS=1 DS_BUILD_CUTLASS_OPS=0 DS_BUILD_RAGGED_DEVICE_OPS=0 DS_BUILD_SPARSE_ATTN=0 DS_BUILD_EVOFORMER_ATTN=0 pip install .
```
#### Example command for pretrain pythia-14m with only nature language
```bash
model_name_or_path="EleutherAI/pythia-14m"
data_paths="/data/openwebtext"
nl_eval_data_path="/data/wikitext-2"
data_ratios="1"
run_name="pythia-14m-openwebtext"
checkpoint_dir="/model/pretrain/${run_name}"
random_initialize="True"

# global_batch_size=1024
num_meta_lang_tokens="0"
max_length="1024"
batch_size="128" # 8 gpus
gradient_accumulation_steps="1"
max_steps="10000"
eval_steps="1000"
save_steps="1000"
resume_training="False"
lr="5e-4"

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 25031 train/pretrain.py \
  --model_name_or_path ${model_name_or_path} \
  --num_meta_lang_tokens ${num_meta_lang_tokens} \
  --random_initialize ${random_initialize} \
  --resume_training ${resume_training} \
  --deepspeed src/configs/ds_z2_config.json \
  --architecture causal \
  --output_dir ${checkpoint_dir} \
  --save_strategy steps \
  --eval_on_start \
  --save_steps ${save_steps} \
  --eval_strategy steps \
  --eval_steps ${eval_steps} \
  --gather_weights True \
  --learning_rate ${lr} \
  --data_paths ${data_paths} \
  --nl_eval_data_path ${nl_eval_data_path} \
  --data_ratios ${data_ratios} \
  --per_device_train_batch_size ${batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --max_length ${max_length} \
  --max_steps ${max_steps} \
  --gradient_checkpointing False \
  --bf16 True \
  --logging_steps 10 \
  --report_to wandb \
  --run_name ${run_name}

```
