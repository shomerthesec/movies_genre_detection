import pandas as pd 
import torch
import torch.nn as nn
import requests
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import os
import re

# ============================================================================
# CONFIG
# ============================================================================
MODEL_NAME = 'prajjwal1/bert-tiny'
MAX_LENGTH = 512
BATCH_SIZE = 64
LR = 2e-3
EPOCHS = 20
WARMUP_STEPS = 500
OUTPUT_DIR = '/media/shomer/Windows/Users/shomer/data/New folder/genre_project/tiny_bert_training/models/qwen_summerized_bert_finetuned'
CACHE_DIR = '/media/shomer/Windows/Users/shomer/data/New folder/genre_project/tiny_bert_training/models/cache'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# DATASET
# ============================================================================
class GenreDataset(Dataset):
    def __init__(self, plots, labels, tokenizer, max_length):
        self.plots = plots
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.plots)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.plots[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ============================================================================
# MODEL
# ============================================================================
class GenreClassifier(nn.Module):
    def __init__(self, model_name, num_labels, cache_dir=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        hidden_size = self.bert.config.hidden_size
        
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.pooler_output
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return loss, logits

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        loss, _ = model(input_ids, attention_mask, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, logits = model(input_ids, attention_mask, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': acc,
        'micro_f1': f1_micro,
        'macro_f1': f1_macro,
        'micro_precision': p_micro,
        'micro_recall': r_micro,
        'macro_precision': p_macro,
        'macro_recall': r_macro
    }

def train_model(model, train_loader, val_loader, epochs, device):
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_micro_f1': [], 'val_macro_f1': []}
    best_f1 = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        metrics = evaluate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(metrics['loss'])
        history['val_accuracy'].append(metrics['accuracy'])
        history['val_micro_f1'].append(metrics['micro_f1'])
        history['val_macro_f1'].append(metrics['macro_f1'])
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f} | Micro F1: {metrics['micro_f1']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
        
        if metrics['micro_f1'] > best_f1:
            best_f1 = metrics['micro_f1']
            save_model(model, f'{OUTPUT_DIR}/best_model.pt')
            print(f"✓ New best model saved! (F1: {best_f1:.4f})")
    
    return history

# ============================================================================
# SAVE/LOAD FUNCTIONS
# ============================================================================
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

# ============================================================================
# DATA PROCESSING
# ============================================================================
def summarize_plot(plot_text):
    url = "http://localhost:11434/api/generate"
    prompt = f"Please provide a concise, one-sentence summary of the following movie plot:\n\n{plot_text}"
    
    try:
        response = requests.post(url, json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False})
        return response.json().get("response", "").strip()
    except:
        return plot_text

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def prepare_data(df, top_genres):
    # Keep only top genres
    df['Genre'] = df['Genre'].apply(lambda x: x if x in top_genres else 'other')
    
    # Filter genres with enough samples
    genre_counts = df['Genre'].value_counts()
    valid_genres = genre_counts[genre_counts > 50].index
    df = df[df['Genre'].isin(valid_genres)]
    
    # Clean data
    df = df.dropna(subset=['Title', 'Release Year', 'Plot'])
    df = df.drop_duplicates(subset=['Title', 'Release Year', 'Plot'])
    
    return df

# ============================================================================
# MAIN
# ============================================================================
def main():
    # Load data
    print("Loading data...")
    try:
        df = pd.read_csv("/media/shomer/Windows/Users/shomer/data/New folder/genre_project/tiny_bert_training/qwen_summerized_plots.csv")
    except:
        df = pd.read_csv("/media/shomer/Windows/Users/shomer/data/New folder/genre_project/data/wiki_movie_plots_deduped.csv")
        top_genres = df['Genre'].value_counts().index[:25]
        df = prepare_data(df, top_genres)
        df['Plot_Clean'] = df['Plot'].apply(summarize_plot)
        df.to_csv("/media/shomer/Windows/Users/shomer/data/New folder/genre_project/tiny_bert_training/qwen_summerized_plots.csv")
    
    # Prepare labels
    print("Preparing labels...")
    genre_labels = df['Genre'].values
    unique_genres = sorted(df['Genre'].unique())
    genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
    labels = np.array([genre_to_idx[g] for g in genre_labels])
    num_labels = len(unique_genres)
    
    # Split data
    plots = df['Plot_Clean'].tolist()
    X_train, X_temp, y_train, y_temp = train_test_split(plots, labels, test_size=0.2, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Create datasets
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    
    train_dataset = GenreDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = GenreDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = GenreDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Train model
    print("Training model...")
    model = GenreClassifier(MODEL_NAME, num_labels, CACHE_DIR).to(device)
    history = train_model(model, train_loader, val_loader, EPOCHS, device)
    
    # Test
    print("\nTesting best model...")
    model = load_model(model, f'{OUTPUT_DIR}/best_model.pt', device)
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Micro F1: {test_metrics['micro_f1']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    
    # Save artifacts
    tokenizer.save_pretrained(f'{OUTPUT_DIR}/tokenizer')
    joblib.dump(genre_to_idx, f'{OUTPUT_DIR}/label_mapping.pkl')
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    axes[0, 1].plot(history['val_accuracy'])
    axes[0, 1].set_title('Validation Accuracy')
    
    axes[1, 0].plot(history['val_micro_f1'], label='Micro F1')
    axes[1, 0].plot(history['val_macro_f1'], label='Macro F1')
    axes[1, 0].set_title('F1 Scores')
    axes[1, 0].legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_results.png', dpi=300)
    print(f"\nDone! Model saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()