from datasets import load_dataset
from transformers import RobertaTokenizerFast
from transformers import RobertaForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

#load
dataset= load_dataset('MartinThoma/wili_2018')
print(dataset)
label_index= dataset['train'][0]['label']
lang_code= dataset['train'].features['label'].names[label_index]

#tokenization
tokenizer= RobertaTokenizerFast.from_pretrained('roberta-base')
def tokenize(batch):
    return tokenizer(batch['sentence'],padding='max_length',truncation=True,max_length=128)

#tokenize the datasets
tokenized_train=dataset['train'].map(tokenize,batched=True)
tokenized_test=dataset['train'].map(tokenize,batched=True)

#rename label column to labels and set format for PyTorch
tokenized_train= tokenized_train.rename_column('label','labels')
tokenized_test= tokenized_test.rename_column('label','labels')
tokenized_train.set_format('torch',columns=['input_ids','attention_mask','labels'])
tokenized_test.set_format('torch',columns=['input_ids','attention_mask','labels'])
print(tokenized_train[0])
print(tokenized_train[0]['labels'])

#load the roberta model
num_labels= len(dataset['train'].features['label'].names)
model= RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=num_labels)
print(model)

#define metrics
def compute_metrics(pred):
    labels= pred.label_ids
    preds= pred.predictions.argmax(-1)
    precision, recall, f1,_=precision_recall_fscore_support(labels, preds,average='weighted')
    acc= accuracy_score(labels,preds)
    return{
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
#define training arguments
training_args= TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    logging_dir='./logs',
    logging_steps=50,
    report_to=["none"] #disable wandb logging for simplicity
)
print(training_args)

#train the model
trainer= Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics= compute_metrics
)
trainer.train()
results=trainer.evaluate(tokenized_test)
print(results)

def predict_language(text):
    inputs=tokenizer(text,return_tensors='pt',padding='max_length',truncation=True, max_length=128)
    inputs={k: v.to(model.device) for k, v in inputs.items()} #move inputs to teh same device as model
    outputs= model(**inputs)
    pred_label= outputs.logits.argmax(dim=1).item()
    lang_code=dataset['train'].features['label'].names[pred_label]
    return lang_code


    predicted_language= predict_language(text_result)
    print(f"Predicted language code:{predicted_language}")
   
