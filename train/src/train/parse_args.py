from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import os
import sys
import torch

from transformers import TrainingArguments


@dataclass
class TrainConfig:
    sm_channel_training: str
    sm_channel_testing: str
    training_arguments: TrainingArguments


def args_to_config() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "--sm_channel_train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN") or "/opt/ml/input/data/training",
        help="Input directory for the training data in the container",
    )

    arg_parser.add_argument(
        "--sm_channel_test",
        type=str,
        default=os.environ.get("SM_CHANNEL_TEST") or "/opt/ml/input/data/testing",
        help="Input directory for the testing data in the container",
    )
    arg_parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/opt/ml/input/data/weights",
        help="Model name in the Huggingface hub, or path to local weights",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR") or "/opt/ml/model",
        help="Directory into which to write the model artifacts after training",
    )
    arg_parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )
    arg_parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
    )
    arg_parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
    )
    arg_parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    arg_parser.add_argument("--weight_decay", type=float, default=0.01)

    # TrainingArguments options
    arg_parser.add_argument("--bf16", action="store_false")
    arg_parser.add_argument("--save_strategy", type=str, default="epoch")
    arg_parser.add_argument("--eval_strategy", type=str, default="epoch")
    arg_parser.add_argument("--report_to", type=str, default="tensorboard")
    arg_parser.add_argument("--use_cache", action="store_true")

    # bitsandbytes options
    arg_parser.add_argument("--load_in_4bit", action="store_false")
    arg_parser.add_argument("--bnb_4bit_use_double_quant", action="store_false")
    arg_parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    arg_parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=torch.dtype,
        default=torch.bfloat16
    )

    # additional model options
    arg_parser.add_argument("--trust_remote_code", action="store_false")
    arg_parser.add_argument("--device_map", type=str, default="auto")

    # LORA options
    arg_parser.add_argument("--lora_rank", type=int, default=8)
    arg_parser.add_argument("--lora_alpha", type=int, default=32)
    arg_parser.add_argument("--lora_dropout", type=float, default=0.05)
    arg_parser.add_argument("--lora_bias", type=str, default="none")
    arg_parser.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")
    arg_parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    )

    # DataCollator options
    arg_parser.add_argument("--mlm", action="store_true")

    return arg_parser.parse_args(sys.argv[1:])
