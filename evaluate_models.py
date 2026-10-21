import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, roc_auc_score

# Import architectures and dataset from your training script
from train_brain_gat import OasisNiftiDataset, VanillaBrainGCN, VanillaBrainGAT, FIXED_TIMEPOINTS

def run_evaluation(model_name, model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x_batch, adj_batch, y_batch in test_loader:
            out = model(x_batch, adj_batch)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # Probability of being Class 1
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    print("\n" + "="*60)
    print(f"         EVALUATION REPORT: {model_name.upper()}         ")
    print("="*60)
    
    # Calculate standard classification metrics
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'], zero_division=0))
    
    # Calculate ROC-AUC Score safely
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
        print(f"ROC-AUC Score: {auc_score:.4f}")
    except ValueError:
        print("ROC-AUC Score: Unable to calculate (Only one class present or predicted in split)")
        
    print("="*60)

def main():
    DATA_PATH = "./Dataset"
    GCN_WEIGHTS = "best_gcn_model.pt"
    GAT_WEIGHTS = "best_gat_model.pt"
    
    # Re-initialize dataset with the exact same random seed split used during training
    dataset = OasisNiftiDataset(root_dir=DATA_PATH)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    _, in_features = dataset[0][0].shape
    
    # 1. Evaluate GCN
    if os.path.exists(GCN_WEIGHTS):
        gcn_model = VanillaBrainGCN(in_channels=in_features, hidden_channels=32)
        gcn_model.load_state_dict(torch.load(GCN_WEIGHTS))
        run_evaluation("GCN Baseline", gcn_model, test_loader)
    else:
        print(f"Skipping GCN: '{GCN_WEIGHTS}' not found.")

    # 2. Evaluate GAT
    if os.path.exists(GAT_WEIGHTS):
        gat_model = VanillaBrainGAT(in_channels=in_features, hidden_channels=32)
        gat_model.load_state_dict(torch.load(GAT_WEIGHTS))
        run_evaluation("GAT Attention Network", gat_model, test_loader)
    else:
        print(f"Skipping GAT: '{GAT_WEIGHTS}' not found. Wait for the training script to finish!")

if __name__ == "__main__":
    main()