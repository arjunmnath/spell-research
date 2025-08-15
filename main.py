import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
import copy
from transformers import BertModel, RobertaModel, BertTokenizer, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, random_split, DataLoader, IterableDataset, ConcatDataset
import sklearn
from torch.optim import AdamW
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score 
from tqdm import tqdm
import demoji 
import random
demoji.download_codes() 
import preprocessor as p
from indictrans import Transliterator
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED)
plt.rcParams['figure.figsize'] = [15, 8]
plt.rcParams.update({'font.size': 8})
RANDOM_SEED = 42
model_path = 'ai4bharat/indic-bert'
model_path = 'xlm-roberta-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value)  
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
random_seed(RANDOM_SEED, True)


class Dataset_OLID():
    def __init__(self, train_data, batch_size = 32):
        self.train_data = train_data
        self.batch_size = batch_size

        self.label_dict = {'Not_offensive': 0,
                            'Offensive_Targeted_Insult_Group': 3,
                            'Offensive_Targeted_Insult_Individual': 2,
                            'Offensive_Targeted_Insult_Other': 4,
                            'Offensive_Untargetede': 1}
                                    
        self.count_dic = {}
        self.train_dataset = self.process_data(self.train_data)

    def tokenize(self, sentences, padding = True, max_len = 256):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        input_ids, attention_masks = [], []
        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(sent,
                                                    add_special_tokens=True,
                                                    max_length=max_len, 
                                                    padding='max_length', 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt', 
                                                    truncation = True)
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return {'input_ids': (input_ids), 'attention_masks': (attention_masks)}
    
    def process_data(self, data):
        sentences, labels = [], []
        print(len(data))
        for id,line in enumerate(data):
            if id==0: continue
            sentence = line.strip().split('\t')
            label = sentence[2:]

            if label[0] == 'NOT': labels.append(0)
            elif label[1] == 'UNT': labels.append(1)
            elif label[2] == 'IND': labels.append(2)
            elif label[2] == 'GRP': labels.append(3)
            else: labels.append(4)

            sentence = sentence[1].replace('#','').lower()
            emoji_dict = demoji.findall(sentence)
            if len(emoji_dict): 
                for emoji, text in emoji_dict.items():
                    sentence = sentence.replace(emoji, ' '+text+' ')
                    sentence = ' '.join(sentence.split())
            sentences.append(sentence)
            self.count_dic[labels[-1]] = self.count_dic.get(labels[-1], 0) + 1
        inputs = self.tokenize(sentences)
        return TensorDataset(inputs['input_ids'], inputs['attention_masks'], torch.Tensor(labels))
    
    def get_dataloader(self, inputs, labels, train = True):
        data = TensorDataset(inputs['input_ids'], inputs['attention_masks'], labels)
        if train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)

with open('olid/olid-training-v1.0.tsv', 'r') as f:
    train_data = f.readlines()
olid_data = Dataset_OLID(train_data)

tr1 = Transliterator(source='tam', target='eng', build_lookup=True)
tr2 = Transliterator(source='mal', target='eng', build_lookup=True)
tr3 = Transliterator(source='kan', target='eng', build_lookup=True)

class Dataset():
    def __init__(self, train_data, val_data, batch_size = 32):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size

        self.label_dict = {'Not_offensive': 0,
                            'Offensive_Targeted_Insult_Group': 3,
                            'Offensive_Targeted_Insult_Individual': 2,
                            'Offensive_Targeted_Insult_Other': 4,
                            'Offensive_Untargetede': 1}
        self.count_dic = {}
        self.train_dataset = self.process_data(self.train_data)
        self.val_dataset = self.process_data(self.val_data)

        
        # self.train_dataloader = self.get_dataloader(self.train_inputs, self.train_labels)
        # self.val_dataloader = self.get_dataloader(self.val_inputs, self. val_labels, train = False)

    def tokenize(self, sentences, padding = True, max_len = 256):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        input_ids, attention_masks = [], []
        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(sent,
                                                    add_special_tokens=True,
                                                    max_length=max_len, 
                                                    padding='max_length', 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt', 
                                                    truncation = True)
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return {'input_ids': input_ids, 'attention_masks': attention_masks}
    
    def process_data(self, data):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        sentences, labels = [], []
        for line in data:
            sentence = line.strip().split('\t')
            label = sentence.pop()
            if label not in self.label_dict: continue
            # print('label found')
            sentence = ((' '+tokenizer.sep_token+' ').join(sentence)).replace('#','').lower()
            sentence = tr3.transform(tr2.transform(tr1.transform(sentence)))
            # sentence = p.clean(' '.join(sentence)).replace('#','')
            emoji_dict = demoji.findall(sentence)
            if len(emoji_dict): 
                for emoji, text in emoji_dict.items():
                    sentence = sentence.replace(emoji, ' '+text+' ')
                    sentence = ' '.join(sentence.split())
            sentences.append(sentence)
            # if label =='Not_offensive': labels.append(0)
            # else:
            labels.append(self.label_dict[label])
            self.count_dic[labels[-1]] = self.count_dic.get(labels[-1], 0) + 1
        inputs = self.tokenize(sentences)

        return TensorDataset(inputs['input_ids'], inputs['attention_masks'], torch.Tensor(labels))
    
    def get_dataloader(self, inputs, labels, train = True):
        data = TensorDataset(inputs['input_ids'], inputs['attention_masks'], labels)
        if train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)


with open('./FIRE-2025/Kannada/kannada_offensive_train.csv', 'r') as f:
    train_data = f.readlines()
with open('./FIRE-2025/Kannada/kannada_offensive_dev.csv', 'r') as f:
    val_data = f.readlines()
kan_data = Dataset(train_data, val_data)

with open('./FIRE-2025/Malayalam/mal_full_offensive_train.csv', 'r') as f:
    train_data = f.readlines()
with open('./FIRE-2025/Malayalam/mal_full_offensive_dev.csv', 'r') as f:
    val_data = f.readlines()
mal_data = Dataset(train_data, val_data)

with open('./FIRE-2025/Tamil/tamil_offensive_full_train.csv', 'r') as f:
    train_data = f.readlines()
with open('./FIRE-2025/Tamil/tamil_offensive_full_dev.csv', 'r') as f:
    val_data = f.readlines()
tam_data = Dataset(train_data, val_data)


# Save and Load Functions
def save_metrics(save_path, epochs, model, optimizer, F1):

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epochs': epochs+1,
                  'F1': F1}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, model, optimizer):
    try: 
        state_dict = torch.load(load_path, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    except: 
        state_dict = {}

    print(f'Model loaded from <== {load_path}')
    
    return state_dict.get('epochs', 0), state_dict.get('F1', 0)


class Embedding(torch.nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.embeddings = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.3)
        self.output_vector_size = self.embeddings.config.hidden_size * 2

    def forward(self, input_ids, mask):
        outputs = self.embeddings(input_ids, mask)
        out = outputs.last_hidden_state # -> (batch_size, num_words, 768)
        mean_pooling = torch.mean(out, 1)
        max_pooling, _ = torch.max(out, 1)
        embed = torch.cat((mean_pooling, max_pooling), 1) # -> (batch_size, 768 * 2)
        y_pred = self.dropout(embed)
        return y_pred


class HANFE(nn.Module):
    def __init__(self, input_vector_size, hidden_size=128, dropout_prob=0.3, num_heads=4):
        super(HANFE, self).__init__()
        self.word_rnn = nn.LSTM(input_vector_size, hidden_size, batch_first=True)
        self.word_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

        self.sentence_rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.sentence_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(1)
        word_out, _ = self.word_rnn(x)
        word_out = word_out.permute(1, 0, 2)
        word_attended, _ = self.word_attention(word_out, word_out, word_out)  
        word_attended = word_attended.permute(1, 0, 2)  
        word_attended = word_attended.mean(dim=1)  
        
        sentence_out, _ = self.sentence_rnn(word_attended.unsqueeze(1)) 
        sentence_out = sentence_out.permute(1, 0, 2) 
        sentence_attended, _ = self.sentence_attention(sentence_out, sentence_out, sentence_out)  
        sentence_attended = sentence_attended.permute(1, 0, 2)  
        sentence_attended = sentence_attended.mean(dim=1)  

        return sentence_attended

class Classifier(nn.Module):
    def __init__(self, hidden_size=128, num_classes=5, dropout_prob=0.3):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc2(x)
        return x
        # x = self.dropout(x)
        # x = self.relu(x)
        # logits = self.fc2(x)
        # return logits


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embed = Embedding()
        self.fe = HANFE(self.embed.output_vector_size)
        self.classifier = Classifier()

    def forward(self, input_ids, mask):
        x = self.embed(input_ids, mask)
        # x = self.fe(x)
        logits = self.classifier(x)
        return logits


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
 
def get_predicted(preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    return pred_flat
 
def evaluate(test_dataloader, model):
    model.eval()
    y_preds, y_test = np.array([]), np.array([])

    for batch in test_dataloader:
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device).long()
        with torch.no_grad():        
            ypred = model(b_input_ids, b_input_mask)
        ypred = ypred.cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        y_preds = np.hstack((y_preds, get_predicted(ypred)))
        y_test = np.hstack((y_test, label_ids))

    weighted_f1 = f1_score(y_test, y_preds, average='weighted')
    return weighted_f1, y_preds, y_test
 
def train(training_dataloader, validation_dataloader, model, filepath, weights = None, learning_rate = 2e-5, epochs = 4, print_every = 10):
    total_steps = len(training_dataloader) * epochs
    torch.cuda.empty_cache()
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    current_epoch, best_weighted_f1 = load_metrics(filepath, model, optimizer)
    if weights == None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
    for epoch_i in range(current_epoch, epochs):
        model.train()
        for batch in tqdm(training_dataloader):
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device).long()
            
            outputs = model(b_input_ids, b_input_mask)
            loss = criterion(outputs, b_labels)
 
            # if step%print_every == 0:
            #     print(loss.item())
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
 
        print('### Validation Set Stats')
        weighted_f1, ypred, ytest = evaluate(validation_dataloader, model)
        print(f"  Weighted F1 {epoch_i}: {weighted_f1:.4f}")
        if weighted_f1 > best_weighted_f1:
            best_weighted_f1 = weighted_f1
            save_metrics(filepath, epoch_i, model.module.embed.embeddings, optimizer, weighted_f1)

# train_dataset = ConcatDataset([olid_data.train_dataset, kan_data.train_dataset, mal_data.train_dataset, tam_data.train_dataset])
# train_dataset = ConcatDataset([mal_data.train_dataset, tam_data.train_dataset])
train_dataset = ConcatDataset([mal_data.train_dataset, olid_data.train_dataset])
val_dataset = mal_data.val_dataset


count_dic = {}
for data in train_dataset:
    label = int(data[2])
    count_dic[label] = count_dic.get(label, 0)+1
weights = torch.Tensor([1+np.log(len(train_dataset)/count_dic[i]) for i in range(5)]).to(device)

print(weights)
print(len(train_dataset))


train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=64)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=128)


model = Model().to(device)
model = nn.DataParallel(model)
model = torch.compile(model)
optimizer = AdamW(model.parameters(), lr=3e-5, eps = 1e-8)
# load_metrics('olid_kannada_mbert.pt', model, optimizer)

train(train_dataloader, val_dataloader, model, 'olid_xlmr_base_embed_new.pt', weights=weights, epochs=4)

_, ypred, ytest = evaluate(val_dataloader , model)
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, classification_report
array = confusion_matrix(ytest, ypred)

precision, recall, f1, _ = precision_recall_fscore_support(
                ytest, ypred, average='weighted', zero_division=0
            )
accuracy = accuracy_score(ytest, ypred)
fpr, tpr, thresholds = roc_curve(ytest, ypred)
roc_auc = auc(fpr, tpr)

print(f"Accuracy: {accuracy:.4f}")
print(f"precision: {precision: .4f}")
print(f"recall: {recall: .4f}")
print(f"f1: {f1: .4f}")
print(f"AUC: {roc_auc:.4f}")
print(classification_report(ytest, ypred))












