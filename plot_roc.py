import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_curve, auc

from train_brain_gat import OasisNiftiDataset, VanillaBrainGCN, VanillaBrainGAT

def get_predictions(model, test_loader):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for x_batch, adj_batch, y_batch in test_loader:
            out = model(x_batch, adj_batch)
            probs = F.softmax(out, dim=1)
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

def main():
    DATA_PATH = "./Dataset"
    GCN_WEIGHTS = "best_gcn_model.pt"
    GAT_WEIGHTS = "best_gat_model.pt"
    
    dataset = OasisNiftiDataset(root_dir=DATA_PATH)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    _, in_features = dataset[0][0].shape
    
    plt.figure(figsize=(8, 6))
    
    if os.path.exists(GCN_WEIGHTS):
        gcn_model = VanillaBrainGCN(in_channels=in_features, hidden_channels=32)
        gcn_model.load_state_dict(torch.load(GCN_WEIGHTS))
        labels, probs = get_predictions(gcn_model, test_loader)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='blue', lw=2, label='GCN Baseline (AUC = ' + str(round(roc_auc, 4)) + ')')
    
    if os.path.exists(GAT_WEIGHTS):
        gat_model = VanillaBrainGAT(in_channels=in_features, hidden_channels=32)
        gat_model.load_state_dict(torch.load(GAT_WEIGHTS))
        labels, probs = get_predictions(gat_model, test_loader)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='GAT Attention (AUC = ' + str(round(roc_auc, 4)) + ')')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_image = "roc_curve.png"
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print("\n[Success] ROC curve chart compiled and saved to: " + str(os.path.abspath(output_image)))

if __name__ == "__main__":
    main()
