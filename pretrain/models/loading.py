from utils.distributed import is_main_process
import transformers
import torch


def load_tokenizer(
    model_name_or_path,
    cache_dir=None,
    num_meta_lang_tokens=5000,
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
    )
    if num_meta_lang_tokens > 0:
        tokenizer.add_tokens([f"<meta_lang_{i}>" for i in range(num_meta_lang_tokens)], special_tokens=False)
        if is_main_process():
            print(f"Added {num_meta_lang_tokens} meta language tokens")

    return tokenizer


def load_model(
    model_name_or_path, 
    architecture, 
    tokenizer=None,
    flash_attention=False,
    cache_dir=None,
    num_meta_lang_tokens=5000,
):
    if architecture == 'causal':
        # Check hf_home
        # rank = get_rank()
        # print(f"Rank {rank}: {os.environ['HF_HOME']}")
        # print(f"Rank {rank}: cache dir: {args.cache_dir}")
        if flash_attention:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16,
                attn_implementation='flash_attention_2'
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16,
            )
    elif architecture == 'seq2seq':
        if flash_attention:
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                attn_implementation='flash_attention_2'
            )
        else:
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
            )
    else:
        raise ValueError(f"Architecture {architecture} not supported")
    
    if num_meta_lang_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        if is_main_process():
            print(f"Resized token embeddings to {len(tokenizer)}")
    
    return model