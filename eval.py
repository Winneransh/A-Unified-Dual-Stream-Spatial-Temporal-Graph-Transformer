import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from model import (
    DualStreamEpochLevelModel,
    CombinedEpochWithContextDataset,
    load_all_labels,
    collate_epoch_context_batch,
    evaluate,
    train_one_epoch,
)

CLASS_NAMES = [
    "Healthy/Control",
    "Cognitive Impairment (AD)",
    "Frontotemporal Dementia (FTD)",
    "Schizophrenia (SZ)",
    "Parkinson's OFF (PD)",
]

# Source-dataset name -> label(s) it contributes (for reference/printing only)
LODO_SOURCES = ["Warsaw", "SU-SZ", "UC_San_Diego", "UNM", "UI"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Dual-Stream GCN-Transformer model.")
    parser.add_argument("--mode", type=str, default="test", choices=["test", "lodo"],
                         help="'test': evaluate a saved checkpoint on the held-out test split. "
                              "'lodo': run leave-one-dataset-out evaluation (trains a fresh "
                              "model per fold, no fine-tuning from a shared checkpoint).")
    parser.add_argument("--checkpoint", type=str, default="best_dual_stream_5class_model.pt",
                         help="Path to the trained model checkpoint (.pt). Used in --mode test.")
    parser.add_argument("--acf_pickle_path", type=str,
                         default="all_eeg_subjects_fractal_8s.pkl",
                         help="Path to the AHEPA (AD/FTD/HC) preprocessed pickle.")
    parser.add_argument("--schizo_pickle_path", type=str,
                         default="/content/preprocessed_gcn_lstm_8sec.pkl",
                         help="Path to the schizophrenia preprocessed pickle.")
    parser.add_argument("--parkinson_pickle_folder", type=str,
                         default="/content/extracted_content/preprocessed_parkinson_data",
                         help="Path to the folder of Parkinson's preprocessed pickles.")
    parser.add_argument("--labels_csv_path", type=str,
                         default="/content/subject_labels_corrected.csv",
                         help="Path to the AHEPA A/C/F label CSV.")
    parser.add_argument("--context_window_size", type=int, default=13,
                         help="Temporal context window size (epochs).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_split", type=float, default=0.70)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42,
                         help="Must match the seed used to produce the checkpoint's test split.")
    parser.add_argument("--lodo_epochs", type=int, default=100,
                         help="Number of training epochs per LODO fold.")
    parser.add_argument("--lodo_lr", type=float, default=0.0001)
    return parser.parse_args()


def infer_source_dataset(sample):
    """
    Best-effort mapping from a dataset sample to one of the five LODO
    source datasets (Warsaw, SU-SZ, UC_San_Diego, UNM, UI), based on
    the coarse 'dataset' tag and subject_id/filename patterns. This is
    a heuristic, not a guaranteed-correct split -- adjust the patterns
    below if your filenames differ.
    """
    coarse = sample.get('dataset', '')
    sid = str(sample.get('subject_id', '')).lower()

    if coarse == 'Schizo':
        if 'su' in sid or 'sz' in sid:
            return 'SU-SZ'
        return 'Warsaw'
    if coarse == 'Parkinson':
        if 'unm' in sid or 'nm' in sid:
            return 'UNM'
        if 'ui' in sid or 'iowa' in sid:
            return 'UI'
        return 'UC_San_Diego'
    return None  # ACF (AHEPA) subjects are not part of the LODO protocol


def build_full_dataset(args):
    """Load labels and build the full combined dataset (all sources)."""
    labels_dict = load_all_labels(
        args.labels_csv_path, args.schizo_pickle_path, args.parkinson_pickle_folder
    )
    with open(args.acf_pickle_path, 'rb') as f:
        acf_data = pickle.load(f)
    with open(args.schizo_pickle_path, 'rb') as f:
        schizo_data = pickle.load(f)

    dataset = CombinedEpochWithContextDataset(
        acf_data, schizo_data, args.parkinson_pickle_folder,
        labels_dict, args.context_window_size
    )
    return dataset


def run_lodo(args):
    """
    Leave-one-dataset-out evaluation. For each of the five source
    datasets (Warsaw, SU-SZ, UC_San_Diego, UNM, UI), train a fresh
    model on all remaining sources (plus AHEPA, which is not part of
    the LODO protocol but is kept in training throughout) and test on
    the held-out source, with no fine-tuning on the target.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("LEAVE-ONE-DATASET-OUT (LODO) EVALUATION")
    print("=" * 80)

    dataset = build_full_dataset(args)
    source_tags = [infer_source_dataset(s) for s in dataset.samples]
    num_channels = dataset.samples[0]['spatial_features'].shape[0]

    results = []

    for held_out in LODO_SOURCES:
        print(f"\n{'-'*80}")
        print(f"FOLD: holding out '{held_out}'")
        print(f"{'-'*80}")

        train_idx = [i for i, tag in enumerate(source_tags) if tag != held_out]
        test_idx = [i for i, tag in enumerate(source_tags) if tag == held_out]

        if len(test_idx) == 0:
            print(f"  No samples found for source '{held_out}', skipping fold. "
                  f"(Check infer_source_dataset() patterns against your filenames.)")
            continue

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_epoch_context_batch
        )
        test_loader = DataLoader(
            test_subset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_epoch_context_batch
        )

        model = DualStreamEpochLevelModel(
            num_channels=num_channels, fractal_features=14, d_model=128,
            num_heads=4, spatial_layers=3, temporal_layers=2, d_ff=512,
            num_classes=5, dropout=0.1
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lodo_lr)

        for epoch in range(args.lodo_epochs):
            train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            if (epoch + 1) % 10 == 0 or epoch == args.lodo_epochs - 1:
                print(f"  Epoch [{epoch+1:3d}/{args.lodo_epochs}] "
                      f"Train Acc: {train_acc:6.2f}% Loss: {train_loss:.4f}")

        test_acc, test_loss, test_prec, test_rec, test_f1, test_cm = evaluate(
            model, test_loader, criterion, device
        )

        print(f"\n  Held-out source: {held_out}")
        print(f"  Test Accuracy:  {test_acc:.2f}%")
        print(f"  Test Precision: {test_prec:.2f}%")
        print(f"  Test Recall:    {test_rec:.2f}%")
        print(f"  Test F1-score:  {test_f1:.2f}%")

        results.append({
            'held_out': held_out,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1': test_f1,
        })

    print("\n" + "=" * 80)
    print("LODO SUMMARY (Table X style)")
    print("=" * 80)
    print(f"{'Held-out Source':<18}{'Train':>8}{'Test':>8}{'Acc (%)':>10}"
          f"{'Prec (%)':>10}{'Rec (%)':>10}{'F1 (%)':>10}")
    for r in results:
        print(f"{r['held_out']:<18}{r['n_train']:>8}{r['n_test']:>8}"
              f"{r['accuracy']:>10.2f}{r['precision']:>10.2f}{r['recall']:>10.2f}{r['f1']:>10.2f}")
    if results:
        mean_acc = np.mean([r['accuracy'] for r in results])
        print(f"\nMean LODO accuracy across {len(results)} fold(s): {mean_acc:.2f}%")
    print("=" * 80)


def build_test_loader(args):
    """Rebuild the same combined dataset and test split used during training."""
    labels_dict = load_all_labels(
        args.labels_csv_path, args.schizo_pickle_path, args.parkinson_pickle_folder
    )

    with open(args.acf_pickle_path, 'rb') as f:
        acf_data = pickle.load(f)
    with open(args.schizo_pickle_path, 'rb') as f:
        schizo_data = pickle.load(f)

    dataset = CombinedEpochWithContextDataset(
        acf_data, schizo_data, args.parkinson_pickle_folder,
        labels_dict, args.context_window_size
    )

    total_size = len(dataset)
    train_size = int(args.train_split * total_size)
    val_size = int(args.val_split * total_size)
    test_size = total_size - train_size - val_size

    torch.manual_seed(args.seed)
    _, _, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_epoch_context_batch
    )

    num_channels = dataset.samples[0]['spatial_features'].shape[0]
    return test_loader, num_channels


def print_per_class_report(all_labels, all_preds):
    """Print precision/recall/F1 per class plus macro/weighted averages."""
    print("\n" + "=" * 80)
    print("PER-CLASS PERFORMANCE")
    print("=" * 80)
    print(classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES,
        digits=4, zero_division=0
    ))


def print_confusion_matrix(cm):
    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    header = "        " + "".join(f"{name[:8]:>10}" for name in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>10d}" for v in row)
        print(f"{CLASS_NAMES[i][:8]:>8}{row_str}")
    print("=" * 80)


def collect_predictions(model, data_loader, device):
    """Run inference over a data loader and collect predictions/labels."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for context_feat, spatial_feat, spatial_adj, labels, target_pos in data_loader:
            context_feat = context_feat.to(device)
            spatial_feat = spatial_feat.to(device)
            spatial_adj = spatial_adj.to(device)
            target_pos = target_pos.to(device)

            outputs = model(spatial_feat, spatial_adj, context_feat, target_pos)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    args = parse_args()

    if args.mode == "lodo":
        run_lodo(args)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("EVALUATING DUAL-STREAM GCN-TRANSFORMER MODEL")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}\n")

    test_loader, num_channels = build_test_loader(args)

    model = DualStreamEpochLevelModel(
        num_channels=num_channels, fractal_features=14, d_model=128,
        num_heads=4, spatial_layers=3, temporal_layers=2, d_ff=512,
        num_classes=5, dropout=0.1
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')} "
          f"(validation accuracy: {checkpoint.get('val_acc', float('nan')):.2f}%)\n")

    criterion = nn.CrossEntropyLoss()
    test_acc, test_loss, test_prec, test_rec, test_f1, test_cm = evaluate(
        model, test_loader, criterion, device
    )

    print("=" * 80)
    print("OVERALL TEST SET PERFORMANCE")
    print("=" * 80)
    print(f"Accuracy:  {test_acc:.2f}%")
    print(f"Precision: {test_prec:.2f}%")
    print(f"Recall:    {test_rec:.2f}%")
    print(f"F1-score:  {test_f1:.2f}%")
    print(f"Loss:      {test_loss:.4f}")

    all_labels, all_preds = collect_predictions(model, test_loader, device)
    print_per_class_report(all_labels, all_preds)
    print_confusion_matrix(test_cm)


if __name__ == "__main__":
    main()
