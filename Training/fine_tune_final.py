import os
import torch
import gc
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import evaluate
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util  # For SBERTScore

torch.autograd.set_detect_anomaly(True)

# Initialize evaluation metrics
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
sbert_model = SentenceTransformer("all-mpnet-base-v2")  # Lightweight SBERT model for SBERTScore
epoch_count = 0

train_losses = []
num_loss_to_stop = 50

class EarlyStoppingOnLossCallback(TrainerCallback):
    def __init__(self, threshold_loss):
        super().__init__()
        self.threshold_loss = threshold_loss

    def on_log(self, args, state, control, logs=None, **kwargs):
        global train_losses
        # Check if training loss is available in the logs
        if logs is not None and "loss" in logs:
            current_loss = logs["loss"]
            train_losses.append(current_loss)
            prev_avg = sum(train_losses[-num_loss_to_stop:])/num_loss_to_stop
            # Stop training if the loss is below the threshold
            # if current_loss <= self.threshold_loss:
            if current_loss > prev_avg:
                # print(f"Training loss ({current_loss}) reached the threshold ({self.threshold_loss}). Stopping training.")
                print(f"Training loss ({current_loss}) reached the threshold ({prev_avg}). Stopping training.")
                control.should_training_stop = True  # Stop training
                control.should_train = False


def initialize_wandb(model_name, task_type):
    wandb.finish()
    # wandb.init(project=f"{task_type}_fine_tuning1", name=f"run_{model_name}", reinit=True)
    wandb.init(project=f"final_fine_tuning_{task_type}", name=f"run_{model_name}", reinit=True)

def load_parquet_dataset(parquet_path):
    df = pd.read_parquet(parquet_path)
    return Dataset.from_pandas(df)

def split_dataset(dataset, test_size=0.1):
    return dataset.train_test_split(test_size=test_size)

def preprocess_function(examples, tokenizer, task_type, max_input_length=512, max_target_length=256):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.bos_token
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token  # Set pad token if missing
    if task_type == "summarization":
        inputs = [f"Summarize this part of the research paper to less than {len(inp.split())//10} words:\n{inp}" for inp in examples["input"]]
        targets = examples["summary1"]
    elif task_type == "story_generation":
        inputs = [f'''You are a master storyteller, known for crafting immersive and emotionally engaging stories that captivate readers of all ages.
Below is a "summary" of a research paper. Your task is to transform this summary into a "fully developed short story" with a natural flow, engaging characters, and a compelling plot.
## Guidelines for the Story:
- **Creative & Narrative-Driven**: Do not sound like a research paper. The story should feel "organic, engaging, and immersive".
- **Well-Developed Characters**: Introduce "relatable, human-like" characters with clear motivations.
- **Flow & Pacing**: The story should "unfold naturally" with a clear "beginning, middle, and end".
- **Easily Understandable**: Use "simple, conversational, yet elegant language" that anyone can enjoy.
- **Show, Don't Tell**: Use "vivid descriptions" and "natural dialogue" instead of just explaining ideas.
- **Engaging Conflict & Resolution**: The story should have "a central conflict" that gets resolved meaningfully.
---
### **Summary:**
{inp}
---
### Now, weave this into a captivating short story.
Ensure it feels like a "real, immersive narrative", not an AI-generated text. Make it flow like a professionally written short story. End the story naturally with a satisfying conclusion.
(Stop generating as soon as the story is complete.)''' for inp in examples["summary1"]]
        targets = examples["story"]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds, tokenizer, task_type):
    global epoch_count
    predictions, labels = eval_preds
    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)  # Clip negatives to zero
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    metrics = {}
    if task_type == "summarization":
        # Summarization metrics
        rouge_scores = rouge.compute(predictions=predictions, references=labels)
        bert_scores = bertscore.compute(predictions=predictions, references=labels, lang="en")
        metrics.update({
            "rouge1": rouge_scores["rouge1"] * 100,
            "rouge2": rouge_scores["rouge2"] * 100,
            "rougeL": rouge_scores["rougeL"] * 100,
            "bert_f1": np.mean(bert_scores["f1"]) * 100,
            "epoch": epoch_count,
        })
    elif task_type == "story_generation":
        # Story generation metrics
        bert_scores = bertscore.compute(predictions=predictions, references=labels, lang="en")
        sbert_scores = np.mean([util.pytorch_cos_sim(sbert_model.encode(pred), sbert_model.encode(ref)).item() for pred, ref in zip(predictions, labels)])
        # sbert_scores = util.cos_sim(sbert_model.encode(predictions), sbert_model.encode(labels)).diag().mean().item()
        metrics.update({
            "bert_f1": np.mean(bert_scores["f1"]) * 100,
            "sbert_score": sbert_scores * 100,
            "epoch": epoch_count,
        })
    epoch_count += 1

    wandb.log(metrics)
    return metrics

def fine_tune_model(model_info, train_dataset, val_dataset, output_dir, task_type, epochs=10):
    global epoch_count
    epoch_count = 1
    model_name = model_info["name"]
    model_id = model_info["model_id"]
    # if os.path.exists(os.path.join(output_dir, task_type, model_name)):
    #     shutil.rmtree()
    max_input_length = model_info.get("max_input_length", 512)
    max_target_length = model_info.get("max_target_length", 128)
    batch_size = model_info.get("batch_size", 4)
    gradient_accumulation_steps = model_info.get("gradient_accumulation_steps", 1)
    learning_rate = model_info.get("learning_rate", 3e-5)
    weight_decay = model_info.get("weight_decay", 0.01)
    warmup_steps = model_info.get("warmup_steps", 1000)
    generation_max_length = model_info.get("generation_max_length", max_target_length)
    save_total_limit = model_info.get("save_total_limit", 3)
    threshold_loss = model_info.get("threshold", 0.2)

    # Initialize Weights & Biases for logging
    initialize_wandb(model_name, task_type)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # Load model with appropriate class
    if "mistral" in model_id.lower() or "zephyr" in model_id.lower() or "phi" in model_id.lower() or "tinyllama" in model_id.lower():
        # Causal models
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    else:
        if "prophetnet" in model_id.lower():
            # Seq2Seq models
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                quantization_config=quantization_config
            )
            # Move the model to the GPU manually
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
        elif "bart" in model_id.lower():
            # Seq2Seq models
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            model.to(device)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
            )
        data_collator = DataCollatorForSeq2Seq(tokenizer)
    if "bart" not in model_id.lower(): # and "prophetnet" not in model_id.lower():
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=model_info['r'],
            lora_alpha=model_info['lora_alpha'],
            target_modules=model_info['target_modules'],
            lora_dropout=0.1,
            bias="none",
            task_type=model_info['task_type'],
        )
        model = get_peft_model(model, lora_config)
        model.train()
        # for name, _ in model.named_parameters():
        # for name, param in model.named_parameters():
        #     if "lora" in name:
        #         print(f"{name}: requires_grad = {param.requires_grad}")
        # print(model.training)
        # model.print_trainable_parameters()
        # print("LoRA Layers Require Grad:", any(p.requires_grad for p in model.parameters()))

    # Tokenization
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer, task_type, max_input_length, max_target_length),batched=True,remove_columns=train_dataset.column_names,)
    tokenized_val = val_dataset.map(lambda x: preprocess_function(x, tokenizer, task_type, max_input_length, max_target_length),batched=True,remove_columns=val_dataset.column_names,)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(output_dir, task_type, model_name),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,  # FP16 for older GPUs
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,  # BF16 for newer GPUs
        # fp16=torch.cuda.is_available() and "t5-base" not in model_id.lower(),  # Enable mixed precision if GPU available
        # fp16=False,
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        logging_dir=os.path.join(output_dir, task_type, model_name, "logs"),
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        logging_steps=10,
        report_to="wandb",
        run_name=f"{task_type}_{model_name}",
    )

    # Initialize the custom callback
    early_stopping_callback = EarlyStoppingOnLossCallback(threshold_loss=threshold_loss)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, task_type),
        callbacks=[early_stopping_callback],
    )

    print(f"$$$$$$$$$$\tStarting {task_type} fine-tuning for {model_name}...")
    # trainer.train(resume_from_checkpoint="fine_tuned_models/summarization/ProphetNet-Summarization/checkpoint-2356/")
    trainer.train()
    task_checkpoint = os.path.join(output_dir, task_type, model_name)
    model.save_pretrained(task_checkpoint)
    print(f"$$$$$$$$$$\tFinished {task_type} fine-tuning for {model_name} and saved in {task_checkpoint}...")
    tokenizer.save_pretrained(task_checkpoint)

    # Cleanup
    wandb.finish()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return task_checkpoint

def main(parquet_path, output_dir="fine_tuned_models", epochs=20):
    dataset = load_parquet_dataset(parquet_path)
    split_ds = split_dataset(dataset)
    train_ds, val_ds = split_ds["train"], split_ds["test"]

    # Summarization parameters
    summarization_params = [
        {"name": "Bart-Base-Summarization","model_id": "facebook/bart-base","max_input_length": 1024,"max_target_length": 128,"batch_size": 8,"gradient_accumulation_steps": 1,"learning_rate": 5e-5,"warmup_steps": 1000,"weight_decay": 0.01,"fp16": True, "threshold":0.2, "r": 16, "lora_alpha": 32, "target_modules": ["q_proj","v_proj"], "task_type": TaskType.SEQ_2_SEQ_LM},
        {"name": "T5-Base-Summarization","model_id": "t5-base","max_input_length": 1024,"max_target_length": 128,"batch_size": 4,"gradient_accumulation_steps": 4,"learning_rate": 3e-5,"warmup_steps": 500,"weight_decay": 0.01,"fp16": True, "threshold":0.2, "r": 16, "lora_alpha": 32, "target_modules": ["q",  "v"], "task_type": TaskType.SEQ_2_SEQ_LM},
        {"name": "ProphetNet-Summarization","model_id": "microsoft/prophetnet-large-uncased","max_input_length": 2048,"max_target_length": 256,"batch_size": 2,"gradient_accumulation_steps": 4,"learning_rate": 3e-5,"warmup_steps": 1500,"weight_decay": 0.01,"fp16": True, "threshold":3.4, "r": 16, "lora_alpha": 64, "target_modules": ["query_proj","value_proj"], "task_type": TaskType.SEQ_2_SEQ_LM},
        {"name": "TinyLlama-Summarization","model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0","max_input_length": 2048,"max_target_length": 256,"batch_size": 2,"gradient_accumulation_steps": 4,"learning_rate": 5e-5,"warmup_steps": 500,"weight_decay": 0.01,"fp16": True, "threshold":0.15, "r": 8, "lora_alpha": 32, "target_modules": ["q_proj","v_proj"], "task_type": TaskType.CAUSAL_LM},
        {"name": "Phi-3-Mini-Summarization","model_id": "microsoft/Phi-3-mini-4k-instruct","max_input_length": 4096,"max_target_length": 256,"batch_size": 2,"gradient_accumulation_steps": 4,"learning_rate": 3e-5,"warmup_steps": 500,"weight_decay": 0.01,"fp16": True, "threshold":0.15, "r": 8, "lora_alpha": 32, "target_modules": ["qkv_proj"], "task_type": TaskType.CAUSAL_LM},
        {"name": "Zephyr-7B-Summarization","model_id": "HuggingFaceH4/zephyr-7b-beta","max_input_length": 8192,"max_target_length": 256,"batch_size": 2,"gradient_accumulation_steps": 4,"learning_rate": 2e-5,"warmup_steps": 1000,"weight_decay": 0.01,"fp16": True, "threshold":0.1, "r": 16, "lora_alpha": 64, "target_modules": ["q_proj","v_proj"], "task_type": TaskType.CAUSAL_LM},
        {"name": "Mistral-7B-Summarization","model_id": "mistralai/Mistral-7B-Instruct-v0.3","max_input_length": 8192,"max_target_length": 256,"batch_size": 2,"gradient_accumulation_steps": 4,"learning_rate": 2e-5,"warmup_steps": 1000,"weight_decay": 0.01,"fp16": True, "threshold":0.08, "r": 16, "lora_alpha": 64, "target_modules": ["q_proj","v_proj"], "task_type": TaskType.CAUSAL_LM},
    ]

    # Story generation parameters
    story_generation_params = [
        {"name": "Bart-Base-Story-Generation","model_id": "fine_tuned_models/summarization/Bart-Base-Summarization","max_input_length": 512,"max_target_length": 256,"batch_size": 8,"gradient_accumulation_steps": 1,"learning_rate": 1e-4,"warmup_steps": 500,"weight_decay": 0.01,"fp16": True, "threshold":0.15, "r": 16, "lora_alpha": 32, "target_modules": ["q_proj","v_proj"], "task_type": TaskType.SEQ_2_SEQ_LM},
        {"name": "T5-Base-Story-Generation","model_id": "fine_tuned_models/summarization/T5-Base-Summarization","max_input_length": 512,"max_target_length": 256,"batch_size": 4,"gradient_accumulation_steps": 4,"learning_rate": 1e-4,"warmup_steps": 500,"weight_decay": 0.01,"fp16": True, "threshold":0.15, "r": 16, "lora_alpha": 32, "target_modules": ["q",  "v"], "task_type": TaskType.SEQ_2_SEQ_LM},
        {"name": "ProphetNet-Story-Generation","model_id": "fine_tuned_models/summarization/ProphetNet-Summarization","max_input_length": 1024,"max_target_length": 512,"batch_size": 2,"gradient_accumulation_steps": 4,"learning_rate": 5e-5,"warmup_steps": 1000,"weight_decay": 0.01,"fp16": True, "threshold":0.1, "r": 16, "lora_alpha": 64, "target_modules": ["query_proj","value_proj"], "task_type": TaskType.SEQ_2_SEQ_LM},
        {"name": "TinyLlama-Story-Generation","model_id": "fine_tuned_models/summarization/TinyLlama-Summarization","max_input_length": 1024,"max_target_length": 256,"batch_size": 2,"gradient_accumulation_steps": 4,"learning_rate": 1e-4,"warmup_steps": 500,"weight_decay": 0.01,"fp16": True, "threshold":0.1, "r": 8, "lora_alpha": 32, "target_modules": ["q_proj","v_proj"], "task_type": TaskType.CAUSAL_LM},
        {"name": "Phi-3-Mini-Story-Generation","model_id": "fine_tuned_models/summarization/Phi-3-Mini-Summarization","max_input_length": 1024,"max_target_length": 512,"batch_size": 2,"gradient_accumulation_steps": 4,"learning_rate": 5e-5,"warmup_steps": 500,"weight_decay": 0.01,"fp16": True, "threshold":0.1, "r": 8, "lora_alpha": 32, "target_modules": ["qkv_proj"], "task_type": TaskType.CAUSAL_LM},
        {"name": "Zephyr-7B-Story-Generation","model_id": "fine_tuned_models/summarization/Zephyr-7B-Summarization","max_input_length": 1024,"max_target_length": 512,"batch_size": 2,"gradient_accumulation_steps": 4,"learning_rate": 5e-5,"warmup_steps": 500,"weight_decay": 0.01,"fp16": True, "threshold":0.1, "r": 16, "lora_alpha": 64, "target_modules": ["q_proj","v_proj"], "task_type": TaskType.CAUSAL_LM},
        {"name": "Mistral-7B-Story-Generation","model_id": "fine_tuned_models/summarization/Mistral-7B-Summarization","max_input_length": 1024,"max_target_length": 512,"batch_size": 2,"gradient_accumulation_steps": 4,"learning_rate": 5e-5,"warmup_steps": 500,"weight_decay": 0.01,"fp16": True, "threshold":0.08, "r": 16, "lora_alpha": 64, "target_modules": ["q_proj","v_proj"], "task_type": TaskType.CAUSAL_LM},
    ]

    # Fine-tune for summarization
    for model_info1, model_info2 in zip(summarization_params, story_generation_params):
        # print(model_info1, model_info2, sep="\n", end="\n\n")
        model_info2['model_id'] = fine_tune_model(model_info1, train_ds, val_ds, output_dir, task_type="summarization", epochs=epochs)
        final_checkpoint = fine_tune_model(model_info2, train_ds, val_ds, output_dir, task_type="story_generation", epochs=epochs)
        print(f"Fine-tuning model is saved in {final_checkpoint}")

if __name__ == "__main__":
    main("research_papers_final.parquet", output_dir="fine_tuned_models", epochs=50)
