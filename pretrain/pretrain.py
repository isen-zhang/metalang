from itertools import chain
import sys
sys.path.append("./")
from data.dataset import IterableCollator, PretrainStreamingDataset, PretrainStreamingIdsDataset, PretrainStreamingMixDataset
from data.loading import load_datasets
from models.loading import load_model, load_tokenizer
from utils.setting import set_project, set_system, set_args, set_distributed_logging
from dataclasses import field, dataclass
from typing import Dict, Optional, Any
from data.loading import load_datasets
import transformers
from transformers import Trainer, AutoConfig, AutoModelForCausalLM
from typing import List
from prompts.templates import null_template
from utils.logging import get_logger
from utils.distributed import get_rank, is_main_process
from datasets import load_dataset


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: str = field(default="google/flan-t5-base")
    architecture: str = field(default='causal')
    flash_attention: bool = False
    data_paths: List[str] = field(default_factory=list) # List of data paths
    data_ratios: List[float] = field(default_factory=list) # List of data ratios
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    resume_training: bool = False
    per_device_train_batch_size = 8
    max_length: int = 2048
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    gather_weights: bool = True
    # datasets: List[str] = field(default_factory=list)
    # dataset_nums: List[int] = field(default_factory=int)
    template: str = field(default="llama-3")
    random_initialize: bool = True
    # For iterable dataset with variable sizes
    dispatch_batches: bool = False
    split_batches: bool = True
    # dataloader_drop_last: bool = True
    # meta-lang setting
    num_meta_lang_tokens: int = 0


def train():
    # set_system("src/configs/project_config.json")
    # set_distributed_logging(strict=True)
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    set_args(args)
    set_project(args)

    # Get rank
    rank = get_rank()
    Logger = get_logger("logs", level="INFO", rank=rank)

    tokenizer = load_tokenizer(
        args.model_name_or_path, 
        cache_dir=args.cache_dir,
        num_meta_lang_tokens=args.num_meta_lang_tokens,
    )

    Logger.info("---- Loading Datasets ----")
    # dataset, collator = load_datasets(
    #     chat=args.chat,
    #     architecture=args.architecture,
    #     datasets=args.datasets,
    #     dataset_nums=args.dataset_nums,
    #     tokenizer=tokenizer,
    #     max_length=args.max_length,
    #     template=args.template,
    # )
    # dataset = PretrainStreamingDataset(
    #     tokenizer=tokenizer,
    #     data_name_or_path=args.data_path,
    #     cutoff_len=args.max_length,
    # )
    # dataset = PretrainStreamingIdsDataset(
    #     tokenizer=tokenizer,
    #     data_name_or_path=args.data_path,
    #     cutoff_len=args.max_length,
    #     concatenate=True,
    # )
    dataset = PretrainStreamingMixDataset(
        tokenizer=tokenizer,
        data_name_or_paths=args.data_paths,
        data_ratios=args.data_ratios,
        cutoff_len=args.max_length,
    )
    collator = IterableCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_max_length=True,
    )


    Logger.info(f"Data path: {args.data_paths}")

    Logger.info("---- Loading Model ----")

    model = load_model(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        architecture=args.architecture,
        flash_attention=args.flash_attention,
        cache_dir=args.cache_dir,
        num_meta_lang_tokens=args.num_meta_lang_tokens,
    )
    # if args.random_initialize:
    #     config = AutoConfig.from_pretrained(args.model_name_or_path)
    #     model = AutoModelForCausalLM.from_config(config)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         args.model_name_or_path,
    #         cache_dir=args.cache_dir,
    #     )

    trainer = Trainer(
        model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
    )

    trainer.train(resume_from_checkpoint=args.resume_training)
    if is_main_process():
        tokenizer.save_pretrained(args.output_dir)

    # Whether to gather weights before saving
    # This is prefered for small models
    if args.gather_weights:
        trainer.save_model(args.output_dir)
    else:
        trainer.deepspeed.save_checkpoint(args.output_dir)
    # trainer.deepspeed.save_16bit_model(args.output_dir)
    # lean_state_dict = deepspeed.checkpoint.utils.de[epspeed.checkpoint.utils.clone_tensors_for_torch_save(trainer.deepspeed.module.state_dict())
    # trainer.deepspeed.module.save_pretrained("lean_after", state_dict=lean_state_dict)


if __name__ == "__main__":
    train()