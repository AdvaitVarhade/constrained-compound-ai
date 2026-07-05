import os
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-3B"
OUTPUT_DIR = "./lora_outputs"
# Hard limit of 4096 to prevent KV cache explosion per [RULE 1]
MAX_SEQ_LENGTH = 4096 

def formatting_prompts_func(example):
    """
    Formats a single example using the tokenizer's chat template.
    Newer TRL versions call this with batched=False, so we return a single string.
    """
    return global_tokenizer.apply_chat_template(
        example['messages'], 
        tokenize=False, 
        add_generation_prompt=False
    )

def train_adapter(adapter_name: str, dataset_path: str):
    """
    Trains a specialized LoRA adapter on the given dataset.
    Implements multiple techniques to avoid Out-Of-Memory (OOM) on constrained GPUs.
    """
    print(f"\n{'='*50}\nStarting QLoRA training for adapter: {adapter_name}\n{'='*50}")
    
    # 1. Configure QLoRA (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 2. Load the base model in 4-bit
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 3. Prepare model memory efficiency
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    # 4. Configure LoRA target modules
    # We focus on linear layers (specifically the attention projections) 
    # with a rank of r=16 for a good balance of adaptability and small adapter size.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Attention projection linear layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # The SFTTrainer will apply the LoRA configuration to the base model for us.
    
    # 5. Load the specific dataset for this adapter
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")
    
    # 6. Setup Training Arguments
    # These settings are aggressively tuned to prevent OOM on a 4GB GPU.
    adapter_output_dir = os.path.join(OUTPUT_DIR, adapter_name)
    training_args = SFTConfig(
        output_dir=adapter_output_dir,
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=1,      # Absolutely crucial: batch size of 1 to minimize VRAM usage
        gradient_accumulation_steps=8,      # Accumulate gradients to simulate a larger batch size of 8
        optim="paged_adamw_32bit",          # Paged optimizer moves states to CPU RAM if GPU VRAM spikes
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        max_grad_norm=0.3,                  # Gradient clipping to prevent exploding gradients
        num_train_epochs=3,                  # Multiple passes over small dataset for thorough learning
        # max_steps=500,                      # Replaced by num_train_epochs for better coverage
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,                          # Enabled bf16 for Ampere GPU
        gradient_checkpointing=True,        # Explicitly enable gradient checkpointing in the trainer
        report_to="none"                    # Disable wandb/tensorboard for clean local execution
    )
    
    # 7. Initialize the SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=global_tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )
    
    # 8. Execute Training
    print("Beginning training loop...")
    trainer.train()
    
    # 9. Save the adapter securely
    # The saved weights can later be merged or converted to GGUF using llama.cpp scripts
    final_save_path = os.path.join(adapter_output_dir, "final")
    trainer.model.save_pretrained(final_save_path)
    global_tokenizer.save_pretrained(final_save_path)
    print(f"[OK] Adapter '{adapter_name}' trained and saved to: {final_save_path}")

def main():
    print("Initializing Phase 6: LoRA Fine-Tuning Pipeline...")
    
    # Load the global tokenizer used by the formatting function
    global global_tokenizer
    global_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Fix for standard causal language modeling padding issues
    global_tokenizer.pad_token = global_tokenizer.eos_token
    global_tokenizer.padding_side = "right" 
    
    # Verify datasets exist before attempting to train
    orchestrator_dataset = "data/orchestrator_dataset.jsonl"
    coder_dataset = "data/coder_dataset.jsonl"
    
    if not os.path.exists(orchestrator_dataset) or not os.path.exists(coder_dataset):
        print("❌ Error: Datasets not found. Please run prepare_datasets.py first to generate the synthetic data.")
        return
        
    # Sequence 1: Train the Orchestrator adapter
    # (Already completed and saved)
    # train_adapter("orchestrator", orchestrator_dataset)
    
    # Sequence 2: Train the Coder adapter
    # Includes both unified diff (Delta-Code) and scaffold (full file) examples
    train_adapter("coder", coder_dataset)
    
    print("\n🎉 Pipeline Complete! Both adapters have been trained successfully.")
    print("Next Actions: Use the llama.cpp 'convert-lora-to-gguf.py' script to convert the adapters for InferenceGateway usage.")

if __name__ == "__main__":
    main()
