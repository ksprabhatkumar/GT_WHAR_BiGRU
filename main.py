import os
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
from dataset.dsads_loader import get_loso_dataloaders
from models.gt_whar import GT_WHAR
from utils.metrics import calculate_metrics

# ==========================================
# ⚙️ PAPER HYPERPARAMETERS (TABLE I)
# ==========================================
HYPERPARAMS = {
    "dataset": "DSADS",
    "epochs": 50,              # Increased to 50 to allow LR decay to work
    "batch_size": 64,          # Paper Table I
    "lr": 1e-3,                # Paper Table I
    "decay_period": 10,        # Paper Table I
    "decay_rate": 0.9,         # Paper Table I
    "hidden_size": 64,         # Paper Table I
    "dropout": 0.3,            # Regularization for LOSO
    "weight_decay": 1e-4       # L2 Regularization
}

# --- LOGGING FUNCTIONS ---
def log_results(subject_id, acc, f1, filepath="results.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (f"[{timestamp}] Subj: {subject_id} | Acc: {acc:.2f}% | F1: {f1:.2f}% | "
                 f"Params: BS={HYPERPARAMS['batch_size']}, LR={HYPERPARAMS['lr']}, "
                 f"Decay={HYPERPARAMS['decay_rate']}@Ep{HYPERPARAMS['decay_period']}, "
                 f"Hidden={HYPERPARAMS['hidden_size']}\n")
    with open(filepath, "a") as f:
        f.write(log_entry)

def log_summary(avg_acc, avg_f1, filepath="results.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (f"[{timestamp}] >>> FINAL 8-FOLD AVG (BiGRU) | Acc: {avg_acc:.2f}% | F1: {avg_f1:.2f}% "
                 f"[Trained for {HYPERPARAMS['epochs']} Epochs]\n\n")
    with open(filepath, "a") as f:
        f.write(log_entry)
# -------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc, macro_f1 = calculate_metrics(all_labels, all_preds)
    return acc, macro_f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running BiGRU Baseline on DEVICE: {device} ---")
    
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/latest_checkpoint.pth"
    
    start_subject = 1
    start_epoch = 1
    loso_acc_dict = {}
    loso_f1_dict = {}
    
    # --- CRASH RECOVERY ---
    if os.path.exists(checkpoint_path):
        print(f"\n[!] Resume checkpoint found! Loading from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        
        start_subject = checkpoint['subject_id']
        start_epoch = checkpoint['epoch'] + 1
        loso_acc_dict = checkpoint.get('loso_acc_dict', {})
        loso_f1_dict = checkpoint.get('loso_f1_dict', {})
        
        if start_epoch > HYPERPARAMS["epochs"]:
            start_subject += 1
            start_epoch = 1
            
        print(f"[!] Resuming training from Subject {start_subject}, Epoch {start_epoch}")
    
    # --- LOSO LOOP ---
    for subject_id in range(start_subject, 9):
        print(f"\n======================================")
        print(f" FOLD {subject_id}/8: Testing on Subject {subject_id}")
        print(f"======================================")
        
        # NOTE: Using Batch Size 64 as per the paper!
        train_loader, test_loader = get_loso_dataloaders(subject_id, batch_size=HYPERPARAMS["batch_size"])
        
        model = GT_WHAR(hidden_dim=HYPERPARAMS["hidden_size"]).to(device)
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS["lr"], weight_decay=HYPERPARAMS["weight_decay"])
        
        # NEW: Learning Rate Scheduler (From Table I)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=HYPERPARAMS["decay_period"], 
            gamma=HYPERPARAMS["decay_rate"]
        )
        
        best_acc = loso_acc_dict.get(subject_id, 0.0)
        best_f1 = loso_f1_dict.get(subject_id, 0.0)
        
        # Restore state if resuming mid-fold
        if start_epoch > 1 and subject_id == start_subject:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"-> Restored model, optimizer, and scheduler weights.")

        for epoch in range(start_epoch, HYPERPARAMS["epochs"] + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
            
            # Step the learning rate scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            is_best = False
            if test_acc > best_acc:
                best_acc = test_acc
                best_f1 = test_f1
                is_best = True
                
            print(f"Epoch {epoch:02d} | LR: {current_lr:.6f} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
            
            if is_best:
                torch.save(model.state_dict(), f"checkpoints/best_model_fold_{subject_id}.pth")
                
            loso_acc_dict[subject_id] = best_acc
            loso_f1_dict[subject_id] = best_f1
            
            # Save Checkpoint
            torch.save({
                'subject_id': subject_id,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loso_acc_dict': loso_acc_dict,
                'loso_f1_dict': loso_f1_dict,
            }, checkpoint_path)
            
        print(f"-> Best for Subject {subject_id}: Acc = {best_acc:.4f}, F1 = {best_f1:.4f}")
        
        log_results(subject_id, best_acc * 100, best_f1 * 100)
        
        start_epoch = 1

    print("\n======================================")
    print("FINAL 8-FOLD LOSO RESULTS (Matches Paper Benchmarks):")
    avg_acc = np.mean(list(loso_acc_dict.values())) * 100
    avg_f1 = np.mean(list(loso_f1_dict.values())) * 100
    print(f"Average Accuracy: {avg_acc:.2f}%")
    print(f"Average Macro-F1: {avg_f1:.2f}%")
    print("======================================")
    
    log_summary(avg_acc, avg_f1)
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)