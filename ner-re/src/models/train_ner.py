 # [训练] NER 启动脚本

import os
import config.config as config
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from data.data_loader import DiaKG_NER_Dataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def main():
    print(f"Loading NER Model: {config.NER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.NER_MODEL_NAME)

    # 标签映射
    label2id = {"O": 0}
    for i, ent in enumerate(config.ENTITY_TYPES):
        label2id[f"B-{ent}"] = 2 * i + 1
        label2id[f"I-{ent}"] = 2 * i + 2
    id2label = {v: k for k, v in label2id.items()}

    # 数据集
    print("Processing Data...")
    dataset = DiaKG_NER_Dataset(config.DATA_DIR, tokenizer, label2id, config.NER_MAX_LEN)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 模型
    model = AutoModelForTokenClassification.from_pretrained(
        config.NER_MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=os.path.join(config.OUTPUT_DIR, "ner_model"),
        eval_strategy="epoch", save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        save_total_limit=1,
        logging_steps=50
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )

    print("Starting Training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print("✅ NER Training Completed.")

if __name__ == "__main__":
    main()
