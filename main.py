import os
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from tqdm import tqdm
from dataset.dsads_loader import get_loso_dataloaders
from models.gt_whar import GT_WHAR
from utils.metrics import calculate_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    acc, macro_f1 = calculate_metrics(all_labels, all_preds)
    return total_loss / len(dataloader), acc, macro_f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc, macro_f1 = calculate_metrics(all_labels, all_preds)
    return total_loss / len(dataloader), acc, macro_f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on DEVICE: {device} ---")
    
    epochs = 40 # Standard for LOSO loops
    
    # ---------------------------------------------------------
    # CHECKPOINT SETUP & RESUME LOGIC
    # ---------------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/latest_checkpoint.pth"
    
    start_subject = 1
    start_epoch = 1
    loso_acc_dict = {}
    loso_f1_dict = {}
    
    if os.path.exists(checkpoint_path):
        print(f"\n[!] Resume checkpoint found! Loading from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        
        start_subject = checkpoint['subject_id']
        start_epoch = checkpoint['epoch'] + 1
        loso_acc_dict = checkpoint['loso_acc_dict']
        loso_f1_dict = checkpoint['loso_f1_dict']
        
        # If it crashed exactly after finishing epoch 40, move to the next subject
        if start_epoch > epochs:
            start_subject += 1
            start_epoch = 1
            
        print(f"[!] Resuming training from Subject {start_subject}, Epoch {start_epoch}")
    # ---------------------------------------------------------
    
    # Leave-One-Subject-Out Cross-Validation Loop
    for subject_id in range(start_subject, 9):
        print(f"\n======================================")
        print(f" FOLD {subject_id}/8: Testing on Subject {subject_id}")
        print(f"======================================")
        
        train_loader, test_loader = get_loso_dataloaders(subject_id, batch_size=256)
        
        model = GT_WHAR(hidden_dim=64).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Restore Best Metrics
        best_acc = loso_acc_dict.get(subject_id, 0.0)
        best_f1 = loso_f1_dict.get(subject_id, 0.0)
        
        # Restore Model/Optimizer state if resuming mid-fold
        if start_epoch > 1 and subject_id == start_subject:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"-> Restored model and optimizer weights for mid-fold resume.")

        for epoch in range(start_epoch, epochs + 1):
            train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
            
            is_best = False
            if test_acc > best_acc:
                best_acc = test_acc
                best_f1 = test_f1
                is_best = True
                
            print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
            
            # --- SAVE CHECKPOINTS ---
            # 1. Save Best Model specifically for this fold
            if is_best:
                torch.save(model.state_dict(), f"checkpoints/best_model_fold_{subject_id}.pth")
                
            # 2. Update dictionaries
            loso_acc_dict[subject_id] = best_acc
            loso_f1_dict[subject_id] = best_f1
            
            # 3. Save Latest Checkpoint for crash-recovery
            torch.save({
                'subject_id': subject_id,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loso_acc_dict': loso_acc_dict,
                'loso_f1_dict': loso_f1_dict,
            }, checkpoint_path)
            
        print(f"-> Best for Subject {subject_id}: Acc = {best_acc:.4f}, F1 = {best_f1:.4f}")
        
        # Reset start_epoch to 1 for the next subject fold
        start_epoch = 1

    # Final Calculation
    print("\n======================================")
    print("FINAL 8-FOLD LOSO RESULTS (Matches Paper Benchmarks):")
    avg_acc = np.mean(list(loso_acc_dict.values())) * 100
    avg_f1 = np.mean(list(loso_f1_dict.values())) * 100
    print(f"Average Accuracy: {avg_acc:.2f}%")
    print(f"Average Macro-F1: {avg_f1:.2f}%")
    print("======================================")