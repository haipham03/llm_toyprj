import torch
import transformers
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

USER_HEADER = "### HUMAN:\n"
RESPONSE_HEADER = "### RESPONSE:\n"

class CustomDataProcessor():
    def __init__(self, tokenizer, data_path):
        self.tokenizer = tokenizer
        self.data_path = data_path

    def get_data(self) -> DatasetDict:
        data = load_dataset(self.data_path, split='train[:10%]')
        data = data.map(lambda data_point: self.tokenizer(
            self.gen_prompt(data_point["conversations"]),
            max_length=2048,
            truncation=True,
        ))
        return data

    def gen_prompt(self, data) -> str:
        gen_text = ""
        for turn in data:
            entity = turn["from"]
            value = turn["value"]

            if entity == "human":
                gen_text = gen_text + USER_HEADER + value + "\n\n"
            elif entity == "gpt":
                gen_text = gen_text + RESPONSE_HEADER + value + self.tokenizer.eos_token + "\n\n"
                
        return gen_text


bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
)

class QloraTrainer:
    def __init__(self, base_model_path, lora, data_path):
        self.adapter_model = None
        self.merged_model = None
        self.data_processor = None
        self.data_path = data_path
        
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
        self.model = LlamaForCausalLM.from_pretrained(base_model_path, quantization_config=bnb_config, device_map={"":0})
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        
        if lora is not None:
            self.adapter_model = PeftModel.from_pretrained(self.model, lora, is_trainable = True)
        else:
            config = LoraConfig(
                r=4,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.adapter_model = get_peft_model(self.model, config)   

    def train(self):

        print("Data preprocessing ...")
        self.data_processor = CustomDataProcessor(self.tokenizer, self.data_path)
        data = self.data_processor.get_data()

        print("Training ...")
        trainer = transformers.Trainer(
            model=self.adapter_model,
            train_dataset=data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                num_train_epochs=1,
                learning_rate=0.0002,
                fp16=True,
                logging_steps=10,
                output_dir="trainer_outputs/",
                report_to="tensorboard",
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        self.adapter_model.config.use_cache = False 
        trainer.train()

        model_save_path = "models/adapter_model"
        trainer.save_model(model_save_path)
        
