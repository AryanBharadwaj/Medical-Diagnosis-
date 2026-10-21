import os
import sys
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib

from train_brain_gat import VanillaBrainGAT

FIXED_TIMEPOINTS = 164

def preprocess_single_scan(nii_path):
    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"Target scan not found at: {nii_path}")
        
    print(f"[1/3] Loading volumetric NIfTI scan: {os.path.basename(nii_path)}")
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    flat_data = data.reshape(-1, data.shape[-1])
    step = max(1, flat_data.shape[0] // 90)
    x_raw = flat_data[:90*step:step, :].copy()
    
    if x_raw.shape[0] < 90:
        padding = 90 - x_raw.shape[0]
        x_raw = np.vstack([x_raw, np.zeros((padding, x_raw.shape[1]))])

    if x_raw.shape[1] >= FIXED_TIMEPOINTS:
        x_raw = x_raw[:, :FIXED_TIMEPOINTS]
    else:
        x_raw = np.pad(x_raw, ((0, 0), (0, FIXED_TIMEPOINTS - x_raw.shape[1])), mode='constant')

    stds = np.std(x_raw, axis=1, keepdims=True)
    stds[stds == 0.0] = 1.0  
    x_raw = (x_raw - np.mean(x_raw, axis=1, keepdims=True)) / stds
    x_raw = np.nan_to_num(x_raw)
    
    print("[2/3] Constructing functional neuro-topology graph...")
    connectivity = np.abs(np.nan_to_num(np.corrcoef(x_raw)))
    adj_mask = (connectivity > 0.4).astype(np.float32)
    
    x_tensor = torch.tensor(x_raw, dtype=torch.float).unsqueeze(0)
    adj_tensor = torch.tensor(adj_mask, dtype=torch.float).unsqueeze(0)
    
    return x_tensor, adj_tensor, connectivity

def main():
    # UPDATED: Pointing directly to your verified resting-state BOLD scan
    TARGET_SCAN = "./Dataset/OAS30001_MR_d0129/func1/NIFTI/sub-OAS30001_ses-d0129_task-rest_run-01_bold.nii.gz"
    WEIGHTS_FILE = "best_gat_model.pt"
    
    if not os.path.exists(WEIGHTS_FILE):
        print(f"Error: Could not locate '{WEIGHTS_FILE}'. Please execute 'train_brain_gat.py' first to build the model matrix.")
        return

    try:
        x, adj, raw_conn = preprocess_single_scan(TARGET_SCAN)
    except Exception as e:
        print(f"Pipeline Error: {e}")
        return

    print("[3/3] Loading trained weights file and evaluating network topology...")
    
    model = VanillaBrainGAT(in_channels=FIXED_TIMEPOINTS, hidden_channels=32)
    model.load_state_dict(torch.load(WEIGHTS_FILE))
    model.eval()
    
    with torch.no_grad():
        output = model(x, adj)
        probabilities = F.softmax(output, dim=1).squeeze().tolist()
        
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class] * 100
    
    print("\n" + "="*50)
    print("      LIVE CLINICAL DIAGNOSTIC REPORT (TRAINED MODEL)   ")
    print("="*50)
    status_str = "BIOMARKER DETECTED (Class 1)" if predicted_class == 1 else "HEALTHY CONTROL / NORMAL (Class 0)"
    
    match = re.search(r'OAS3(\d+)', TARGET_SCAN)
    id_code = match.group(0) if match else "UNKNOWN"
    
    print(f"Patient ID Code:    {id_code}")
    print(f"Diagnostic Status:  {status_str}")
    print(f"Model Confidence:   {confidence:.2f}%")
    print(f"Raw Probabilities:  Class 0: {probabilities[0]:.4f} | Class 1: {probabilities[1]:.4f}")
    print("-"*50)
    
    print("Top Functional Biomarker Pathway Alignment Weights:")
    upper_tri = np.triu(raw_conn, k=1)
    top_indices = np.unravel_index(np.argsort(upper_tri.ravel())[-3:], upper_tri.shape)
    
    for idx in range(3):
        u = top_indices[0][idx]
        v = top_indices[1][idx]
        print(f"  -> Pathological Alignment: Region ROI_{u} <───> Region ROI_{v} (Weight: {raw_conn[u,v]:.3f})")
    print("="*50)

if __name__ == "__main__":
    main()
