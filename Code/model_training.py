from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import numpy
import evaluate
import pandas
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang = 'en', tgt_lang = 'zh')
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

#load the source dataset
en_data = pandas.read_json("en-US.jsonl", lines=True)
zh_data = pandas.read_json("zh-CN.jsonl", lines=True)

data = pandas.concat([en_data["utt"], zh_data["utt"]], axis=1, keys=["en", "zh"])
train_data, test_data = train_test_split(data)

dataset = DatasetDict({
                        "train": Dataset.from_pandas(train_data),
                        "test": Dataset.from_pandas(test_data)
                    })
dataset = dataset.remove_columns(["__index_level_0__"])

def tokenize_function(data):
    padding = "max_length"
    max_length = 200
    sources = [d for d in data["en"]]
    targets = [d for d in data["zh"]]
    inputs = tokenizer(sources, max_length=max_length, padding=padding, truncation=True)
    label = tokenizer(targets, max_length=max_length, padding=padding, truncation=True)
    inputs["labels"] = label["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = numpy.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model)

training_args = TrainingArguments(
                    output_dir="Trained model", 
                    evaluation_strategy="epoch",
                    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()