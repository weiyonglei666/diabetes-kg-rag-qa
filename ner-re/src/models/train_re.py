# [训练] RE 启动脚本

import os
import config.config as config
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from data.data_loader import DiaKG_RE_Dataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def main():
    print(f"Loading RE Model: {config.RE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.RE_MODEL_NAME)

    label2id = {rel: i for i, rel in enumerate(config.RELATION_TYPES)}
    id2label = {v: k for k, v in label2id.items()}

    print("Processing Data (Generating Dual Samples + Negative Samples)...")
    dataset = DiaKG_RE_Dataset(config.DATA_DIR, tokenizer, label2id, config.RE_MAX_LEN)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Total Samples: {len(dataset)} (Train: {train_size}, Val: {val_size})")

    model = AutoModelForSequenceClassification.from_pretrained(
        config.RE_MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=os.path.join(config.OUTPUT_DIR, "re_model"),
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
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    print("Starting Training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print("✅ RE Training Completed.")


if __name__ == "__main__":
    main()
