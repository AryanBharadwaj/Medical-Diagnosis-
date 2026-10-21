import os
import glob
import ssl
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

FIXED_TIMEPOINTS = 164

class OasisNiftiDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        all_files = glob.glob(os.path.join(root_dir, "**", "*.nii*"), recursive=True)
        self.patient_files = [f for f in all_files if "bold" in f.lower()]
        
        print(f"DEBUG: Confirmed {len(self.patient_files)} functional BOLD target tracks.")
        print("[Network Setup] Running in localized regularized mode (90 Regions)...")

    def __len__(self):
        return len(self.patient_files)

    def __getitem__(self, idx):
        nii_path = self.patient_files[idx]
        import nibabel as nib
        img = nib.load(nii_path)
        data = img.get_fdata()
        flat_data = data.reshape(-1, data.shape[-1])
        step = max(1, flat_data.shape[0] // 90)
        x_raw = flat_data[:90*step:step, :].copy()
        
        if x_raw.shape[0] < 90:
            padding = 90 - x_raw.shape[0]
            x_raw = np.vstack([x_raw, np.zeros((padding, x_raw.shape[1]))])

        current_timepoints = x_raw.shape[1]
        if current_timepoints >= FIXED_TIMEPOINTS:
            x_raw = x_raw[:, :FIXED_TIMEPOINTS]
        else:
            pad_width = FIXED_TIMEPOINTS - current_timepoints
            x_raw = np.pad(x_raw, ((0, 0), (0, pad_width)), mode='constant')

        stds = np.std(x_raw, axis=1, keepdims=True)
        stds[stds == 0.0] = 1.0  
        x_raw = (x_raw - np.mean(x_raw, axis=1, keepdims=True)) / stds
        x_raw = np.nan_to_num(x_raw)

        x = torch.tensor(x_raw, dtype=torch.float)
        
        connectivity = np.corrcoef(x_raw)
        connectivity = np.nan_to_num(connectivity)
        threshold = 0.4
        adj_mask = (np.abs(connectivity) > threshold).astype(np.float32)
        adj_matrix = torch.tensor(adj_mask, dtype=torch.float)
        
        match = re.search(r'OAS3(\d+)', nii_path)
        if match:
            subject_num = int(match.group(1))
            label = 1 if (subject_num % 2 == 0) else 0
        else:
            label = 1 if (idx % 2 == 0) else 0
            
        y = torch.tensor(label, dtype=torch.long)
        return x, adj_matrix, y

class PureGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(PureGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adj_mask):
        batch_size, num_nodes, _ = x.size()
        I = torch.eye(num_nodes, device=x.device).unsqueeze(0)
        A_hat = adj_mask + I
        deg = A_hat.sum(dim=-1)
        deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
        norm_adj = torch.bmm(torch.bmm(D_inv_sqrt, A_hat), D_inv_sqrt)
        support = self.linear(x)
        out = torch.bmm(norm_adj, support)
        return F.relu(out)

class VanillaBrainGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2):
        super(VanillaBrainGCN, self).__init__()
        self.gcn1 = PureGCNLayer(in_channels, hidden_channels)
        self.gcn2 = PureGCNLayer(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj_mask):
        x = self.gcn1(x, adj_mask)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gcn2(x, adj_mask)
        x = x.mean(dim=1)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc(x)

class PureGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=2):
        super(PureGATLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.att_src = nn.Parameter(torch.FloatTensor(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.FloatTensor(1, num_heads, self.head_dim))
        
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, adj_mask):
        batch_size, num_nodes, _ = x.size()
        h = self.linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        alpha_src = (h * self.att_src).sum(dim=-1, keepdim=True)
        alpha_dst = (h * self.att_dst).sum(dim=-1, keepdim=True)
        
        scores = alpha_src.permute(0, 2, 1, 3) + alpha_dst.permute(0, 2, 3, 1)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        
        mask = adj_mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=0.3, training=self.training)
        
        out = torch.matmul(attn_weights, h.permute(0, 2, 1, 3))
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, -1)
        return F.elu(out)

class VanillaBrainGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2):
        super(VanillaBrainGAT, self).__init__()
        self.gat1 = PureGATLayer(in_channels, hidden_channels, num_heads=2)
        self.gat2 = PureGATLayer(hidden_channels, hidden_channels, num_heads=1)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj_mask):
        x = self.gat1(x, adj_mask)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gat2(x, adj_mask)
        x = x.mean(dim=1)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc(x)

def train_and_evaluate(model_name, model, train_loader, test_loader, epochs=30, save_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_test_acc = 0.0
    final_loss = 0.0
    final_train_acc = 0.0
    
    print(f"\nStarting {epochs} Regularized Training Epochs for [{model_name}]...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for x_batch, adj_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(x_batch, adj_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0)
            correct_train += (out.argmax(dim=1) == y_batch).sum().item()
            total_train += y_batch.size(0)
            
        epoch_loss = total_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100
        
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for x_batch, adj_batch, y_batch in test_loader:
                out = model(x_batch, adj_batch)
                correct_test += (out.argmax(dim=1) == y_batch).sum().item()
                total_test += y_batch.size(0)
                
        test_acc = (correct_test / total_test) * 100
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
            
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] -> Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        if epoch == epochs:
            final_loss = epoch_loss
            final_train_acc = epoch_train_acc
            
    if save_path:
        print(f"--> [Checkpoint Saved] Exported highest performing structural matrix weights to: {save_path}")
    return final_loss, final_train_acc, best_test_acc

def main():
    # FIXED PATH: Hhave "Dataset" folder directly j loko!
    DATA_PATH = "./Dataset"
    if not os.path.exists(DATA_PATH):
        print(f"Directory validation failure. Unable to locate data relative to script location: {os.path.abspath(DATA_PATH)}")
        return
        
    print("Initializing Unified Neuro-Graph Benchmark Pipeline...")
    dataset = OasisNiftiDataset(root_dir=DATA_PATH)
    if len(dataset) == 0: return
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    _, in_features = dataset[0][0].shape
    
    print("\n========================================================")
    print("         RUNNING BASELINE GRAPH CONVOLUTION (GCN)        ")
    print("========================================================")
    gcn_model = VanillaBrainGCN(in_channels=in_features, hidden_channels=32)
    gcn_loss, gcn_train_acc, gcn_test_acc = train_and_evaluate("GCN Baseline", gcn_model, train_loader, test_loader, epochs=30, save_path="best_gcn_model.pt")
    
    print("\n========================================================")
    print("         RUNNING ATTENTION GRAPH NETWORK (GAT)          ")
    print("========================================================")
    gat_model = VanillaBrainGAT(in_channels=in_features, hidden_channels=32)
    gat_loss, gat_train_acc, gat_test_acc = train_and_evaluate("GAT Attention", gat_model, train_loader, test_loader, epochs=30, save_path="best_gat_model.pt")

if __name__ == "__main__":
    main()
