import os
import json
import copy

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# -------------------------
# MODEL
# -------------------------
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),  # FC1
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------
# DATA
# -------------------------
def get_test_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = torchvision.datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


# -------------------------
# METRICS
# -------------------------
@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total


def tensor_stats(x: torch.Tensor):
    x = x.detach().cpu().float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "abs_mean": float(x.abs().mean().item()),
        "abs_max": float(x.abs().max().item()),
    }


# -------------------------
# MAIN
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    weights_path = os.path.join("models", "mnist_cnn.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError("Missing models/mnist_cnn.pth")

    model = MNIST_CNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    loader = get_test_loader(batch_size=128)

    # FC1 weights
    fc1: nn.Linear = model.classifier[1]
    W = fc1.weight.detach().cpu()  # [128, 3136]

    # -------------------------
    # (1) Weight size + sign + stats
    # -------------------------
    report = {}
    report["weights"] = {
        "W_shape": list(W.shape),
        "W_stats": tensor_stats(W),
        "sign_counts": {
            "positive": int((W > 0).sum().item()),
            "negative": int((W < 0).sum().item()),
            "zero": int((W == 0).sum().item()),
        },
    }

    # -------------------------
    # (2) Compute o (input to FC1) and w*o across all patterns
    # o = flatten(features(x)) shape [B, 3136]
    # contrib = w_ji * o_i  => [B, 128, 3136]
    # -------------------------
    total = 0
    contrib_sum = torch.zeros_like(W)         # sum of w*o
    contrib_abs_sum = torch.zeros_like(W)     # sum of |w*o|

    # per-class
    class_abs_sum = {c: torch.zeros_like(W) for c in range(10)}
    class_count = {c: 0 for c in range(10)}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        feats = model.features(images)
        o = torch.flatten(feats, start_dim=1).detach().cpu()  # [B, 3136]

        B = o.shape[0]
        total += B

        contrib = W.unsqueeze(0) * o.unsqueeze(1)  # [B, 128, 3136]

        contrib_sum += contrib.sum(dim=0)
        contrib_abs_sum += contrib.abs().sum(dim=0)

        for c in range(10):
            mask = (labels == c).detach().cpu()
            if mask.any():
                class_count[c] += int(mask.sum().item())
                class_abs_sum[c] += contrib[mask].abs().sum(dim=0)

    contrib_mean = contrib_sum / total
    contrib_abs_mean = contrib_abs_sum / total

    report["w_times_o_all_patterns"] = {
        "patterns": int(total),
        "contrib_mean_stats": tensor_stats(contrib_mean),
        "contrib_abs_mean_stats": tensor_stats(contrib_abs_mean),
    }

    # -------------------------
    # (3) Contribution ratio per neuron j:
    # ratio_ji = |w_ji * o_i| / sum_i |w_ji * o_i|
    # -------------------------
    eps = 1e-12
    per_neuron_sum = contrib_abs_mean.sum(dim=1, keepdim=True) + eps  # [128, 1]
    ratio = contrib_abs_mean / per_neuron_sum  # [128, 3136]

    report["contribution_ratio"] = {
        "ratio_stats": tensor_stats(ratio),
        "top_inputs_first5_neurons": []
    }

    for j in range(5):
        vals, idxs = torch.topk(ratio[j], k=10)
        report["contribution_ratio"]["top_inputs_first5_neurons"].append({
            "neuron": j,
            "top_inputs": [int(i) for i in idxs.tolist()],
            "top_ratios": [float(v) for v in vals.tolist()],
        })

    # -------------------------
    # (4) Class-specific |w*o| (mean) and top inputs per class
    # -------------------------
    report["class_specific"] = {}
    for c in range(10):
        if class_count[c] == 0:
            continue
        class_abs_mean = class_abs_sum[c] / class_count[c]  # [128, 3136]
        per_input = class_abs_mean.mean(dim=0)              # [3136]
        vals, idxs = torch.topk(per_input, k=20)
        report["class_specific"][str(c)] = {
            "count": int(class_count[c]),
            "class_abs_mean_stats": tensor_stats(class_abs_mean),
            "top20_inputs": [int(i) for i in idxs.tolist()],
            "top20_scores": [float(v) for v in vals.tolist()],
        }

    # -------------------------
    # (5) Weight-level ablation: zero top-K |w| weights & measure accuracy drop
    # -------------------------
    base_acc = accuracy(model, loader, device)
    flat_abs = W.abs().view(-1)
    top_vals, top_idx = torch.topk(flat_abs, k=50)

    coords = []
    for idx in top_idx.tolist():
        j = idx // W.shape[1]
        i = idx % W.shape[1]
        coords.append((j, i))

    ablation_results = []
    for K in [1, 5, 10, 25, 50]:
        m = MNIST_CNN().to(device)
        m.load_state_dict(torch.load(weights_path, map_location=device))
        m.eval()

        fc1m: nn.Linear = m.classifier[1]
        with torch.no_grad():
            for (j, i) in coords[:K]:
                fc1m.weight[j, i] = 0.0

        acc_k = accuracy(m, loader, device)
        ablation_results.append({
            "zeroed_topK_by_absW": K,
            "baseline_acc": float(base_acc),
            "ablated_acc": float(acc_k),
            "acc_drop": float(base_acc - acc_k),
        })

    report["weight_ablation"] = ablation_results

    # -------------------------
    # SAVE
    # -------------------------
    out_dir = os.path.join("outputs", "weight_analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fc1_weight_contribution_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved:", out_path)
    print("\nWeight ablation results:")
    for r in ablation_results:
        print(r)


if __name__ == "__main__":
    main()