# Turn off neurons / layers & measure
import os
import copy
import json

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# -------------------------
# 1) MODEL (MATCHES NOTEBOOK)
# -------------------------
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # features[0]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),  # features[3]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),       # classifier[1]
            nn.ReLU(),
            nn.Linear(128, 10)                # classifier[3]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------
# 2) DATA
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
# 3) EVAL
# -------------------------
@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# -------------------------
# 4) ABLATION METHODS
# -------------------------
def ablate_conv_output_channel(model: MNIST_CNN, conv_index_in_features: int, channel_idx: int):
    """
    Turn off ONE conv output channel by zeroing its filter weights and bias.
    conv_index_in_features should be 0 (conv1) or 3 (conv2) in your notebook.
    """
    conv = model.features[conv_index_in_features]
    if not isinstance(conv, nn.Conv2d):
        raise ValueError(f"model.features[{conv_index_in_features}] is not a Conv2d layer.")

    conv.weight.data[channel_idx].zero_()
    if conv.bias is not None:
        conv.bias.data[channel_idx] = 0.0


def ablate_fc_neuron(model: MNIST_CNN, fc_index_in_classifier: int, neuron_idx: int):
    """
    Turn off ONE neuron in a Linear layer by zeroing its outgoing weights and bias.
    fc_index_in_classifier should be 1 (fc1) or 3 (fc2) in your notebook.
    """
    fc = model.classifier[fc_index_in_classifier]
    if not isinstance(fc, nn.Linear):
        raise ValueError(f"model.classifier[{fc_index_in_classifier}] is not a Linear layer.")

    fc.weight.data[neuron_idx].zero_()   # zero that row (this neuron's outputs)
    if fc.bias is not None:
        fc.bias.data[neuron_idx] = 0.0


# -------------------------
# 5) MAIN EXPERIMENT
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_loader = get_test_loader(batch_size=128)

    # Load trained model
    model = MNIST_CNN().to(device)
    weights_path = os.path.join("models", "mnist_cnn.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Cannot find {weights_path}. Train + save the model in your notebook first."
        )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    base_acc = accuracy(model, test_loader, device)
    print(f"Baseline accuracy: {base_acc:.4f}")

    os.makedirs(os.path.join("outputs", "ablation_results"), exist_ok=True)

    # -------------------------
    # A) ABLATE CONV2 CHANNELS (features[3])  -> 64 channels
    # -------------------------
    conv_layer_index = 3
    conv_out_channels = model.features[conv_layer_index].out_channels

    conv_results = []
    for ch in range(conv_out_channels):
        m = copy.deepcopy(model)
        ablate_conv_output_channel(m, conv_layer_index, ch)
        ablated_acc = accuracy(m, test_loader, device)
        drop = base_acc - ablated_acc

        conv_results.append({
            "type": "conv_channel",
            "layer": f"features[{conv_layer_index}]",
            "channel": ch,
            "baseline_acc": base_acc,
            "ablated_acc": ablated_acc,
            "acc_drop": drop
        })

        if ch % 8 == 0:
            print(f"[CONV] channel {ch:02d}/{conv_out_channels-1}: ablated_acc={ablated_acc:.4f}, drop={drop:.4f}")

    conv_results_path = os.path.join("outputs", "ablation_results", "ablation_conv2_channels.json")
    with open(conv_results_path, "w", encoding="utf-8") as f:
        json.dump(conv_results, f, indent=2)
    print(f"\nSaved conv ablation results to: {conv_results_path}")

    # Show top-10 conv channels by drop
    conv_sorted = sorted(conv_results, key=lambda x: x["acc_drop"], reverse=True)
    print("\nTop 10 most important CONV2 channels (largest accuracy drop):")
    for r in conv_sorted[:10]:
        print(f'  {r["layer"]} channel {r["channel"]}: drop={r["acc_drop"]:.4f} (ablated_acc={r["ablated_acc"]:.4f})')

    # -------------------------
    # B) ABLATE FC1 NEURONS (classifier[1]) -> 128 neurons
    # -------------------------
    fc_index = 1
    fc_out = model.classifier[fc_index].out_features  # 128

    fc_results = []
    for n in range(fc_out):
        m = copy.deepcopy(model)
        ablate_fc_neuron(m, fc_index, n)
        ablated_acc = accuracy(m, test_loader, device)
        drop = base_acc - ablated_acc

        fc_results.append({
            "type": "fc_neuron",
            "layer": f"classifier[{fc_index}]",
            "neuron": n,
            "baseline_acc": base_acc,
            "ablated_acc": ablated_acc,
            "acc_drop": drop
        })

        if n % 16 == 0:
            print(f"[FC ] neuron {n:03d}/{fc_out-1}: ablated_acc={ablated_acc:.4f}, drop={drop:.4f}")

    fc_results_path = os.path.join("outputs", "ablation_results", "ablation_fc1_neurons.json")
    with open(fc_results_path, "w", encoding="utf-8") as f:
        json.dump(fc_results, f, indent=2)
    print(f"\nSaved fc ablation results to: {fc_results_path}")

    # Show top-10 fc neurons by drop
    fc_sorted = sorted(fc_results, key=lambda x: x["acc_drop"], reverse=True)
    print("\nTop 10 most important FC1 neurons (largest accuracy drop):")
    for r in fc_sorted[:10]:
        print(f'  {r["layer"]} neuron {r["neuron"]}: drop={r["acc_drop"]:.4f} (ablated_acc={r["ablated_acc"]:.4f})')


if __name__ == "__main__":
    main()