import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import json
import os
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
MODEL_NAME = 'prajjwal1/bert-tiny'
MAX_LENGTH = 512
BATCH_SIZE = 64
LR = 2e-3
EPOCHS = 20
WARMUP_STEPS = 500
OUTPUT_DIR = '/media/shomer/Windows/Users/shomer/data/New folder/genre_project/tiny_bert_training/models/bert_finetuned_optimized'
CACHE_DIR = '/media/shomer/Windows/Users/shomer/data/New folder/genre_project/tiny_bert_training/models/cache'

torch.manual_seed(42)
np.random.seed(42)

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
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Init weights
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.pooler_output
        x = self.dropout(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return loss, logits

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, device, use_amp=True):
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                loss, _ = model(input_ids, attention_mask, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return {
        'val_loss': total_loss / len(dataloader),
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
    
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_accuracy': [], 
        'val_micro_f1': [], 
        'val_macro_f1': []
    }
    best_f1 = 0.0
    
    for epoch in range(epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*80}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        metrics = evaluate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(metrics['val_loss'])
        history['val_accuracy'].append(metrics['accuracy'])
        history['val_micro_f1'].append(metrics['micro_f1'])
        history['val_macro_f1'].append(metrics['macro_f1'])
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f} | Micro F1: {metrics['micro_f1']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
        
        if metrics['micro_f1'] > best_f1:
            best_f1 = metrics['micro_f1']
            save_model(model, optimizer, scheduler, best_f1, history, f'{OUTPUT_DIR}/best_model.pt')
            print(f"✓ New best model saved! (F1: {best_f1:.4f})")
        
        save_model(model, optimizer, scheduler, best_f1, history, f'{OUTPUT_DIR}/checkpoint_epoch_{epoch+1}.pt')
    
    return history, best_f1

# ============================================================================
# SAVE/LOAD
# ============================================================================
def save_model(model, optimizer, scheduler, best_f1, history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_f1': best_f1,
        'history': history
    }, path)

def load_model(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

# ============================================================================
# DATA PROCESSING
# ============================================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_most_common_genre(combined_genre, top_genres):
    for g in top_genres:
        if g in combined_genre:
            return g
    return combined_genre

def clean_dataset(df, top_genres):
    # Map to most common genre
    df['Genre'] = df['Genre'].apply(lambda x: get_most_common_genre(x, top_genres) if x not in top_genres else x)
    
    # Filter genres with enough samples
    genre_counts = df['Genre'].value_counts()
    valid_genres = genre_counts[genre_counts > 50].index
    df = df[df['Genre'].isin(valid_genres)]
    
    # Handle missing values
    df['Director'] = df['Director'].fillna('Unknown')
    df['Cast'] = df['Cast'].fillna('Unknown')
    df['Genre'] = df['Genre'].fillna('unknown')
    df['Plot'] = df['Plot'].fillna('')
    
    # Remove missing critical fields
    df = df.dropna(subset=['Title', 'Release Year', 'Origin/Ethnicity'])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Title', 'Release Year', 'Plot'], keep='first')
    
    # Clean text
    df['Plot_Clean'] = df['Plot'].apply(clean_text)
    df['Title_Clean'] = df['Title'].apply(clean_text)
    
    # Process genres
    df['Genre_List'] = df['Genre'].apply(lambda x: [g.strip().lower() for g in str(x).split(',') if g.strip()])
    
    # Filter rare genres
    all_genres = [g for genres in df['Genre_List'] for g in genres]
    genre_counts = Counter(all_genres)
    common_genres = {g for g, count in genre_counts.items() if count >= 10}
    
    df['Genre_List_Filtered'] = df['Genre_List'].apply(lambda genres: [g for g in genres if g in common_genres])
    df = df[df['Genre_List_Filtered'].apply(len) > 0]
    
    print(f"Dataset after cleaning: {df.shape[0]} rows")
    return df

def plot_results(history, test_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', marker='o', color='coral')
    axes[0, 1].set_title('Validation Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Scores
    axes[1, 0].plot(history['val_micro_f1'], label='Micro F1', marker='o', color='green')
    axes[1, 0].plot(history['val_macro_f1'], label='Macro F1', marker='s', color='blue')
    axes[1, 0].set_title('Validation F1 Scores', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Test metrics
    metrics_data = {
        'Micro F1': test_metrics['micro_f1'],
        'Macro F1': test_metrics['macro_f1'],
        'Accuracy': test_metrics['accuracy']
    }
    axes[1, 1].bar(metrics_data.keys(), metrics_data.values(), color=['green', 'blue', 'coral'])
    axes[1, 1].set_title('Test Set Metrics', fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/media/shomer/Windows/Users/shomer/data/New folder/genre_project/tiny_bert_training/models/bert_finetuned_optimized/bert_finetuning_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*80)
    print("BERT GENRE CLASSIFICATION")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv("/media/shomer/Windows/Users/shomer/data/New folder/genre_project/data/wiki_movie_plots_deduped.csv")
    top_genres = df['Genre'].value_counts().index[:25]
    
    # Clean data
    print("\nCleaning data...")
    df_clean = clean_dataset(df, top_genres)
    
    # Encode genres
    print("\nEncoding genres...")
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df_clean['Genre_List_Filtered'])
    labels = np.argmax(genre_encoded, axis=1)
    num_labels = genre_encoded.shape[1]
    
    print(f"Number of genres: {num_labels}")
    print(f"Number of movies: {len(df_clean)}")
    
    # Split data
    plots = df_clean['Plot_Clean'].tolist()
    X_temp, X_test, y_temp, y_test = train_test_split(plots, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
    
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Create datasets
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    
    train_dataset = GenreDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = GenreDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = GenreDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)
    
    # Train model
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    model = GenreClassifier(MODEL_NAME, num_labels, CACHE_DIR).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    history, best_f1 = train_model(model, train_loader, val_loader, EPOCHS, device)
    
    # Test
    print("\n" + "="*80)
    print("TESTING BEST MODEL")
    print("="*80)
    model, _ = load_model(model, f'{OUTPUT_DIR}/best_model.pt', device)
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Micro F1: {test_metrics['micro_f1']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    
    # Save artifacts
    tokenizer.save_pretrained(f'{OUTPUT_DIR}/tokenizer')
    joblib.dump(mlb, f'{OUTPUT_DIR}/label_encoder.pkl')
    
    config_dict = {
        'model_name': MODEL_NAME,
        'max_length': MAX_LENGTH,
        'num_labels': num_labels,
        'best_val_f1': best_f1,
        'test_metrics': test_metrics
    }
    with open(f'{OUTPUT_DIR}/config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Plot
    plot_results(history, test_metrics)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

if __name__ == '__main__':
    main()