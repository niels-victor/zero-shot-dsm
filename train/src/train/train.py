from transformers import BitsAndBytesConfig, AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from argparse import Namespace
from datasets import DatasetDict, Dataset


def train(args: Namespace) -> None:
    dataset_dict = load_data(args)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
    )

    # Falcon requires you to allow remote code execution. This is because the model 
    # uses a new architecture that is not part of transformers yet.
    # The code is provided by the model authors in the repo.
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        quantization_config=bnb_config,
        device_map=args.device_map
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path
    )
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=args.collator_mlm
    )

    target_modules = vars(args)["lora_target_modules"]
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=args.lora_task_type
    )

    model = get_peft_model(model, config)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.eval_strategy,
        output_dir=args.output_dir,
        report_to=args.report_to,
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        args=training_args,
        data_collator=data_collator,
    )

    model.config.use_cache = args.use_cache  # silence the warnings. Please re-enable for inference!

    # Start training
    trainer.train()

    trainer.save_model(output_dir=args.output_dir)

    trainer.evaluate()


def load_data(args: Namespace) -> DatasetDict:
    train_data = Dataset.load_from_disk(args.sm_channel_train)
    test_data = Dataset.load_from_disk(args.sm_channel_test)
    return DatasetDict({"train": train_data, "test": test_data})
