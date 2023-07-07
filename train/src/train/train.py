import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from argparse import Namespace


def train(args: Namespace) -> None:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Falcon requires you to allow remote code execution. This is because the model 
    # uses a new architecture that is not part of transformers yet.
    # The code is provided by the model authors in the repo.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
        ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    trainer = Trainer(
    model=model,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_test_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir=log_bucket,
        logging_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        save_strategy = "no",
        output_dir="outputs",
        report_to="tensorboard",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # Start training
    trainer.train()
