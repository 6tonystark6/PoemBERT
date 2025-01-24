import csv
import os

import pandas as pd
import random
import torch
import transformers
from lion_pytorch import Lion
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch import nn
from tqdm import tqdm
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def read_file(file_name):
    data = pd.read_csv(file_name)
    data = shuffle(data)
    return data


comments_data = read_file('./aug_poems.csv')

num_classes = 5
num_epochs = 200
lr = 0.0001
batch_size = 64

tokenizer = BertTokenizer.from_pretrained('../pretrained/bert-base-chinese')

train_data, test_data = train_test_split(comments_data, test_size=0.3, random_state=42,
                                         stratify=comments_data['Sentiment'])

train_inputs = tokenizer(train_data['Text'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=8)
test_inputs = tokenizer(test_data['Text'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=8)

train_labels = torch.tensor(train_data['Sentiment'].tolist())
test_labels = torch.tensor(test_data['Sentiment'].tolist())

train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('../pretrained/bert-base-chinese',
                                                                  num_labels=num_classes)
        self.classifier = nn.Linear(768, 5)
        self.dropout = nn.Dropout(0.4)  # 添加 Dropout

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        logits = self.classifier(pooler_output)
        return last_hidden_state,pooler_output,logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 5
model = BertClassifier(num_classes)


def train_bert_classifier(model, train_loader, test_loader, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_train_optimization_steps = len(train_loader) * num_epochs
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             int(num_train_optimization_steps * 0.2),
                                                             num_train_optimization_steps)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        correct_train = 0
        total_train = 0
        for batch in tqdm(train_loader, desc="Epoch {}".format(epoch + 1)):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            last_hidden_state,pooler_output,logits = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(logits, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        accuracy_train = correct_train / total_train
        print("Epoch {}: Training Accuracy: {:.2%}".format(epoch + 1, accuracy_train))

        # 在验证集上进行评估
        model.eval()
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation"):
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                last_hidden_state,pooler_output,logits = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(logits, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

        accuracy_valid = correct_valid / total_valid
        print("Epoch {}: Validation Accuracy: {:.2%}".format(epoch + 1, accuracy_valid))

        if epoch == (num_epochs - 1):
            torch.save(model.state_dict(), 'sentiment_bert.pt')


bert_classifier = BertClassifier(num_classes)
train_bert_classifier(bert_classifier, train_loader, test_loader, num_epochs, lr)

