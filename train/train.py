import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments,
    EvalPrediction, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configurações
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
BATCH_SIZE = 4
EPOCHS = 3
MAX_LENGTH = 256  # Tamanho máximo permitido pelo modelo
STRIDE = 128  # Quantidade de tokens sobrepostos

# Função para aplicar Sliding Window
def split_text_sliding_window(text, tokenizer, max_length=MAX_LENGTH, stride=STRIDE):
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    chunks = []
    
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk))
        
        if i + max_length >= len(tokens):  # Último pedaço
            break
    
    return chunks

# Dataset Customizado com Sliding Window
class DifficultyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH, stride=STRIDE):
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        for text, label in zip(texts, labels):
            chunks = split_text_sliding_window(text, tokenizer, max_length, stride)
            self.texts.extend(chunks)
            self.labels.extend([label] * len(chunks))  # Cada pedaço herda o mesmo label

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Função de métricas de avaliação
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }

# Carregar datasets processados
with open("train_dataset.json", "r") as f:
    train_data = json.load(f)
with open("test_dataset.json", "r") as f:
    test_data = json.load(f)

# Processamento dos dados para o modelo
train_texts, train_labels = [], []
test_texts, test_labels = [], []

for item in train_data:
    train_texts.append(item["Questao"])
    train_labels.append(item["difficulty"])  # Já numérico

for item in test_data:
    test_texts.append(item["Questao"])
    test_labels.append(item["difficulty"])  # Já numérico

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Criando datasets com Sliding Window
train_dataset = DifficultyDataset(train_texts, train_labels, tokenizer)
test_dataset = DifficultyDataset(test_texts, test_labels, tokenizer)

# Modelo
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)  # 3 classes (0, 1, 2)

# Configuração do Treinamento
training_args = TrainingArguments(
    output_dir="./results_distilbert",
    evaluation_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_dir="./logs_distilbert",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    learning_rate=1e-5,
    weight_decay=0.01,
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

# Treinamento
trainer.train()
eval_results = trainer.evaluate()

# 📊 Visualização
metrics = ["eval_loss", "eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]
titles = ["Loss", "Accuracy", "F1-score", "Precision", "Recall"]
colors = "blue"

fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))

for i, metric in enumerate(metrics):
    value = eval_results[metric]
    axes[i].bar(["DistilBERT"], [value], color=colors)
    axes[i].set_title(titles[i])
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.savefig("distilbert_metrics.png")
plt.show()
