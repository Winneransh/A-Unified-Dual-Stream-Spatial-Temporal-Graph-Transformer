import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import pandas as pd
import math
import os
import glob
import re
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LOAD LABELS FROM ALL SIX SOURCES
# ============================================================================

def load_all_labels(csv_path, schizo_pickle_path, parkinson_pickle_folder):
    """
    Load labels from all six datasets.

    Label mapping:
    - Class 0: Healthy/Control (Group A + Schizo Healthy + Parkinson HC)
    - Class 1: Cognitive Impairment (Group C)
    - Class 2: Frontotemporal Dementia (Group F)
    - Class 3: Schizophrenia
    - Class 4: Parkinson's OFF Medication
    """
    labels_dict = {}

    print(f"\n{'='*80}")
    print("LOADING LABELS FROM ALL DATASETS")
    print(f"{'='*80}\n")

    # === 1. Load CSV labels (A/C/F) ===
    print("1. Loading A/C/F labels from CSV...")
    df = pd.read_csv(csv_path)
    csv_label_mapping = {'A': 0, 'C': 1, 'F': 2}

    csv_counts = {0: 0, 1: 0, 2: 0}
    for _, row in df.iterrows():
        participant_id = row['participant_id']
        group_label = row['label']
        labels_dict[participant_id] = csv_label_mapping[group_label]
        csv_counts[csv_label_mapping[group_label]] += 1

    print(f"   Group A (Healthy): {csv_counts[0]} subjects")
    print(f"   Group C (Cognitive): {csv_counts[1]} subjects")
    print(f"   Group F (Dementia): {csv_counts[2]} subjects")

    # === 2. Load schizophrenia labels from pickle ===
    print("\n2. Loading Schizophrenia labels from pickle...")
    with open(schizo_pickle_path, 'rb') as f:
        schizo_data = pickle.load(f)

    schizo_healthy = 0
    schizo_patient = 0
    for file_data in schizo_data['data']:
        subject_id = file_data.get('subject_id', file_data.get('filename', ''))
        label = file_data['label']
        if label == 0:
            labels_dict[subject_id] = 0
            schizo_healthy += 1
        elif label == 1:
            labels_dict[subject_id] = 3
            schizo_patient += 1

    print(f"   Schizo Healthy: {schizo_healthy} subjects -> mapped to Class 0")
    print(f"   Schizophrenia: {schizo_patient} subjects -> mapped to Class 3")

    # === 3. Load Parkinson's labels from pickle folder ===
    print("\n3. Loading Parkinson's labels from pickle folder...")
    pickle_folder = Path(parkinson_pickle_folder)
    pkl_files = sorted(glob.glob(os.path.join(pickle_folder, "*_fractal_8s.pkl")))

    parkinson_healthy = 0
    parkinson_patient = 0
    for pkl_path in pkl_files:
        basename = os.path.basename(pkl_path)
        if re.match(r'^h_', basename) or 'hc' in basename.lower():
            labels_dict[basename] = 0
            parkinson_healthy += 1
        elif re.match(r'^p_off_', basename) or ('pd' in basename.lower() and 'off' in basename.lower()):
            labels_dict[basename] = 4
            parkinson_patient += 1

    print(f"   Parkinson HC: {parkinson_healthy} subjects -> mapped to Class 0")
    print(f"   Parkinson OFF: {parkinson_patient} subjects -> mapped to Class 4")

    print(f"\n{'='*80}")
    print("COMBINED LABELS SUMMARY")
    print(f"{'='*80}")
    print(f"Total subjects with labels: {len(labels_dict)}")

    label_counts = {}
    for label in labels_dict.values():
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\n  Class 0 (Healthy/Control): {label_counts.get(0, 0)} subjects")
    print(f"  Class 1 (Cognitive Impairment): {label_counts.get(1, 0)} subjects")
    print(f"  Class 2 (Frontotemporal Dementia): {label_counts.get(2, 0)} subjects")
    print(f"  Class 3 (Schizophrenia): {label_counts.get(3, 0)} subjects")
    print(f"  Class 4 (Parkinson's OFF): {label_counts.get(4, 0)} subjects")
    print(f"{'='*80}\n")

    return labels_dict


# ============================================================================
# DATASET CLASS - Combines all six datasets
# ============================================================================

class CombinedEpochWithContextDataset(Dataset):
    """
    Dataset that combines all pickle sources (A/C/F + Schizophrenia + Parkinson's).
    Each EPOCH is a sample with sliding temporal context.
    """
    def __init__(self, acf_pickle_data, schizo_pickle_data, parkinson_pickle_folder,
                 labels_dict, context_window_size=13, verbose=True):
        self.samples = []
        self.context_window_size = context_window_size
        self.half_window = context_window_size // 2

        if verbose:
            print(f"\n{'='*80}")
            print(f"Creating combined dataset from all sources...")
            print(f"{'='*80}\n")

        # === 1. Process A/C/F dataset ===
        subjects_dict = acf_pickle_data['subjects']
        processed_acf = 0
        for subject_id, subject_data in subjects_dict.items():
            base_subject_id = subject_id.split('_')[0]
            if base_subject_id not in labels_dict:
                continue

            label = labels_dict[base_subject_id]
            features = subject_data['features']
            adjacency = subject_data['adjacency']
            n_epochs = subject_data['n_epochs']

            for epoch_idx in range(n_epochs):
                start_idx = max(0, epoch_idx - self.half_window)
                end_idx = min(n_epochs, epoch_idx + self.half_window + 1)
                context_features = features[start_idx:end_idx]
                target_position = epoch_idx - start_idx

                self.samples.append({
                    'context_features': context_features,
                    'spatial_features': features[epoch_idx],
                    'spatial_adjacency': adjacency[epoch_idx],
                    'label': label,
                    'target_position': target_position,
                    'context_size': context_features.shape[0],
                    'subject_id': subject_id,
                    'dataset': 'ACF'
                })
            processed_acf += 1

        # === 2. Process Schizophrenia dataset ===
        schizo_files = schizo_pickle_data['data']
        processed_schizo = 0
        for file_data in schizo_files:
            subject_id = file_data.get('subject_id', file_data.get('filename', ''))
            if subject_id not in labels_dict:
                continue

            label = labels_dict[subject_id]
            features = file_data['features']
            adjacency = file_data['adjacency']
            n_epochs = file_data['n_epochs']

            for epoch_idx in range(n_epochs):
                start_idx = max(0, epoch_idx - self.half_window)
                end_idx = min(n_epochs, epoch_idx + self.half_window + 1)
                context_features = features[start_idx:end_idx]
                target_position = epoch_idx - start_idx

                self.samples.append({
                    'context_features': context_features,
                    'spatial_features': features[epoch_idx],
                    'spatial_adjacency': adjacency[epoch_idx],
                    'label': label,
                    'target_position': target_position,
                    'context_size': context_features.shape[0],
                    'subject_id': subject_id,
                    'dataset': 'Schizo'
                })
            processed_schizo += 1

        # === 3. Process Parkinson's dataset ===
        pickle_folder = Path(parkinson_pickle_folder)
        pkl_files = sorted(glob.glob(os.path.join(pickle_folder, "*_fractal_8s.pkl")))
        processed_parkinson = 0
        for pkl_path in pkl_files:
            basename = os.path.basename(pkl_path)
            if basename not in labels_dict:
                continue

            label = labels_dict[basename]
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            features = data['features']
            adjacency = data['adjacency']
            n_epochs = data['n_epochs']

            for epoch_idx in range(n_epochs):
                start_idx = max(0, epoch_idx - self.half_window)
                end_idx = min(n_epochs, epoch_idx + self.half_window + 1)
                context_features = features[start_idx:end_idx]
                target_position = epoch_idx - start_idx

                self.samples.append({
                    'context_features': context_features,
                    'spatial_features': features[epoch_idx],
                    'spatial_adjacency': adjacency[epoch_idx],
                    'label': label,
                    'target_position': target_position,
                    'context_size': context_features.shape[0],
                    'subject_id': basename,
                    'dataset': 'Parkinson'
                })
            processed_parkinson += 1

        if verbose:
            print(f"Dataset created with {len(self.samples)} EPOCH samples")
            print(f"Processed: {processed_acf} ACF + {processed_schizo} Schizo + {processed_parkinson} Parkinson subjects")

            labels = [s['label'] for s in self.samples]
            print(f"\nSamples per class:")
            print(f"  Class 0 (Healthy/Control): {labels.count(0)} samples")
            print(f"  Class 1 (Cognitive Impairment): {labels.count(1)} samples")
            print(f"  Class 2 (Frontotemporal Dementia): {labels.count(2)} samples")
            print(f"  Class 3 (Schizophrenia): {labels.count(3)} samples")
            print(f"  Class 4 (Parkinson's OFF): {labels.count(4)} samples")
            print(f"  Context window size: {context_window_size} epochs")
            print(f"{'='*80}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context_features = torch.FloatTensor(sample['context_features'])
        spatial_features = torch.FloatTensor(sample['spatial_features'])
        spatial_adjacency = torch.FloatTensor(sample['spatial_adjacency'])
        return (context_features, spatial_features, spatial_adjacency,
                sample['label'], sample['target_position'], sample['context_size'])


# ============================================================================
# CORE COMPONENTS
# ============================================================================

class GraphConvolutionalLayer(nn.Module):
    """Graph Convolutional Layer with symmetric normalization."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, adjacency_matrix):
        batch_size, num_nodes, in_features = x.shape

        A_hat = adjacency_matrix + torch.eye(num_nodes, device=adjacency_matrix.device)
        D = torch.diag(A_hat.sum(dim=1))
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag() + 1e-8))
        A_normalized = D_inv_sqrt @ A_hat @ D_inv_sqrt

        support = torch.matmul(x, self.weight)
        output = torch.zeros(batch_size, num_nodes, self.weight.shape[1], device=x.device)
        for b in range(batch_size):
            output[b] = torch.matmul(A_normalized, support[b])

        if self.bias is not None:
            output = output + self.bias
        return output


class GraphConstrainedAttention(nn.Module):
    """Multi-head attention constrained by graph adjacency."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, adjacency_matrix):
        batch_size, num_nodes, _ = x.shape

        Q = self.W_q(x).view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        adjacency_bias = adjacency_matrix.clone()
        adjacency_bias = torch.where(
            adjacency_bias > 0,
            torch.zeros_like(adjacency_bias),
            torch.full_like(adjacency_bias, -1e9)
        )
        adjacency_bias = adjacency_bias.unsqueeze(0).unsqueeze(0)
        attention_scores = attention_scores + adjacency_bias

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.d_model)
        return self.W_o(context)


class PositionwiseFeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class GCNTransformerFusionLayer(nn.Module):
    """Fused GCN + Graph-Constrained Transformer Layer."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.gcn = GraphConvolutionalLayer(d_model, d_model)
        self.attention = GraphConstrainedAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, x, adjacency_matrix):
        gcn_out = self.gcn(x, adjacency_matrix)
        gcn_out = self.activation(gcn_out)
        x = self.norm1(x + self.dropout1(gcn_out))

        attn_out = self.attention(x, adjacency_matrix)
        x = self.norm2(x + self.dropout2(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


# ============================================================================
# SPATIAL STREAM
# ============================================================================

class SpatialStream(nn.Module):
    """Spatial Stream: Processes a single epoch's spatial connectivity."""

    def __init__(self, num_channels=19, fractal_features=14, d_model=128,
                 num_heads=4, num_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(fractal_features, d_model)
        self.input_gcn = GraphConvolutionalLayer(d_model, d_model)
        self.input_activation = nn.GELU()
        self.input_norm = nn.LayerNorm(d_model)

        self.fusion_layers = nn.ModuleList([
            GCNTransformerFusionLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, spatial_adjacency):
        x = self.input_projection(x)
        x = self.input_gcn(x, spatial_adjacency)
        x = self.input_activation(x)
        x = self.input_norm(x)

        for fusion_layer in self.fusion_layers:
            x = fusion_layer(x, spatial_adjacency)

        return x.mean(dim=1)


# ============================================================================
# TEMPORAL STREAM
# ============================================================================

class TemporalEpochEmbedding(nn.Module):
    """Embed each epoch in the sliding context window."""

    def __init__(self, num_channels=19, fractal_features=14, d_model=128):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(num_channels * fractal_features, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        batch_size, context_size, num_channels, fractal_features = x.shape
        x_flat = x.view(batch_size, context_size, -1)
        return self.embedding(x_flat)


def create_temporal_adjacency(context_size, device):
    """Create tridiagonal temporal adjacency for the context window."""
    adj = torch.zeros(context_size, context_size, device=device)
    for i in range(context_size - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


class TemporalContextStream(nn.Module):
    """Temporal Stream: Processes a sliding window of epochs."""

    def __init__(self, num_channels=19, fractal_features=14, d_model=128,
                 num_heads=4, d_ff=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.epoch_embedding = TemporalEpochEmbedding(num_channels, fractal_features, d_model)
        self.temporal_layers = nn.ModuleList([
            GCNTransformerFusionLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, context_features, target_positions):
        batch_size, context_size, num_channels, fractal_features = context_features.shape
        device = context_features.device

        x = self.epoch_embedding(context_features)
        temporal_adj = create_temporal_adjacency(context_size, device)

        for temporal_layer in self.temporal_layers:
            x = temporal_layer(x, temporal_adj)

        temporal_features = [x[b, target_positions[b], :] for b in range(batch_size)]
        return torch.stack(temporal_features, dim=0)


# ============================================================================
# DUAL-STREAM MODEL
# ============================================================================

class DualStreamEpochLevelModel(nn.Module):
    """Dual-Stream GCN-Transformer for epoch-level 5-class classification."""

    def __init__(self, num_channels=19, fractal_features=14, d_model=128,
                 num_heads=4, spatial_layers=3, temporal_layers=2,
                 d_ff=512, num_classes=5, dropout=0.1):
        super().__init__()

        self.spatial_stream = SpatialStream(
            num_channels, fractal_features, d_model,
            num_heads, spatial_layers, d_ff, dropout
        )
        self.temporal_stream = TemporalContextStream(
            num_channels, fractal_features, d_model,
            num_heads, d_ff, temporal_layers, dropout
        )

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, spatial_features, spatial_adjacency, context_features, target_positions):
        spatial_out = self.spatial_stream(spatial_features, spatial_adjacency)
        temporal_out = self.temporal_stream(context_features, target_positions)

        combined = torch.cat([spatial_out, temporal_out], dim=-1)
        fused = self.fusion(combined)
        return self.classifier(fused)


# ============================================================================
# COLLATE FUNCTION
# ============================================================================

def collate_epoch_context_batch(batch):
    """Collate function for epoch-level batches with temporal context."""
    context_features_list, spatial_features_list = [], []
    spatial_adjacency_list, labels_list = [], []
    target_positions_list, context_sizes = [], []

    for context_feat, spatial_feat, spatial_adj, label, target_pos, context_size in batch:
        context_features_list.append(context_feat)
        spatial_features_list.append(spatial_feat)
        spatial_adjacency_list.append(spatial_adj)
        labels_list.append(label)
        target_positions_list.append(target_pos)
        context_sizes.append(context_size)

    max_context = max(context_sizes)
    batch_size = len(context_features_list)
    num_channels = spatial_features_list[0].shape[0]
    fractal_features = spatial_features_list[0].shape[1]

    padded_context = torch.zeros(batch_size, max_context, num_channels, fractal_features)
    for i, context_feat in enumerate(context_features_list):
        context_size = context_feat.shape[0]
        padded_context[i, :context_size, :, :] = context_feat

    spatial_features_batch = torch.stack(spatial_features_list, dim=0)
    spatial_adjacency_batch = spatial_adjacency_list[0]
    labels_batch = torch.tensor(labels_list, dtype=torch.long)
    target_positions_batch = torch.tensor(target_positions_list, dtype=torch.long)

    return (padded_context, spatial_features_batch, spatial_adjacency_batch,
            labels_batch, target_positions_batch)


# ============================================================================
# TRAINING / EVALUATION UTILITIES
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    correct, total, total_loss = 0, 0, 0

    for context_feat, spatial_feat, spatial_adj, labels, target_pos in train_loader:
        context_feat = context_feat.to(device)
        spatial_feat = spatial_feat.to(device)
        spatial_adj = spatial_adj.to(device)
        labels = labels.to(device)
        target_pos = target_pos.to(device)

        optimizer.zero_grad()
        outputs = model(spatial_feat, spatial_adj, context_feat, target_pos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    return (correct / total) * 100, total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    """Evaluate on a validation or test set."""
    model.eval()
    correct, total, total_loss = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for context_feat, spatial_feat, spatial_adj, labels, target_pos in data_loader:
            context_feat = context_feat.to(device)
            spatial_feat = spatial_feat.to(device)
            spatial_adj = spatial_adj.to(device)
            labels = labels.to(device)
            target_pos = target_pos.to(device)

            outputs = model(spatial_feat, spatial_adj, context_feat, target_pos)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (correct / total) * 100
    avg_loss = total_loss / len(data_loader)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, avg_loss, precision, recall, f1, cm


# ============================================================================
# TRAINING SCRIPT (run this file directly to train)
# ============================================================================

def main():
    acf_pickle_path = "all_eeg_subjects_fractal_8s.pkl"
    schizo_pickle_path = "/content/preprocessed_gcn_lstm_8sec.pkl"
    parkinson_pickle_folder = "/content/extracted_content/preprocessed_parkinson_data"
    labels_csv_path = "/content/subject_labels_corrected.csv"
    context_window_size = 13
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.0001
    train_split, val_split, test_split = 0.7, 0.15, 0.15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("DUAL-STREAM EPOCH-LEVEL MODEL WITH TEMPORAL CONTEXT")
    print("5-CLASS CLASSIFICATION")
    print("0=Healthy | 1=Cognitive | 2=Dementia | 3=Schizophrenia | 4=Parkinson")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    labels_dict = load_all_labels(labels_csv_path, schizo_pickle_path, parkinson_pickle_folder)

    with open(acf_pickle_path, 'rb') as f:
        acf_data = pickle.load(f)
    with open(schizo_pickle_path, 'rb') as f:
        schizo_data = pickle.load(f)

    dataset = CombinedEpochWithContextDataset(
        acf_data, schizo_data, parkinson_pickle_folder, labels_dict, context_window_size
    )

    num_channels = dataset.samples[0]['spatial_features'].shape[0]

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_epoch_context_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_epoch_context_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_epoch_context_batch)

    model = DualStreamEpochLevelModel(
        num_channels=num_channels, fractal_features=14, d_model=128,
        num_heads=4, spatial_layers=3, temporal_layers=2, d_ff=512,
        num_classes=5, dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0
    for epoch in range(num_epochs):
        train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_loss, val_prec, val_rec, val_f1, val_cm = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train Acc: {train_acc:6.2f}% Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:6.2f}% Loss: {val_loss:.4f} | "
              f"P: {val_prec:.2f}% R: {val_rec:.2f}% F1: {val_f1:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_precision': val_prec,
                'val_recall': val_rec,
                'val_f1': val_f1,
                'confusion_matrix': val_cm,
            }, 'best_dual_stream_5class_model.pt')
            print(f"  -> Best model saved! Val Acc: {val_acc:.2f}%")

    print(f"\nTraining complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    print("Run eval.py to evaluate the saved checkpoint on the test set.")


if __name__ == "__main__":
    main()
