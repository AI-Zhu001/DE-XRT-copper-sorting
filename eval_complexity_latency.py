import os
import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms, models


# ============================================================
# 1. Model definitions
# ============================================================
class TICL_PCI_Net_Pro(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18()
        self.resnet.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        return self.classifier(self.resnet(self.normalize(x)))


class CBAL2_Net(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.stream_high = models.resnet18()
        self.stream_high.fc = nn.Identity()

        self.stream_low = models.resnet18()
        self.stream_low.fc = nn.Identity()

        self.attn_gate = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, h, l):
        h = h.repeat(1, 3, 1, 1)
        l = l.repeat(1, 3, 1, 1)

        f_h = self.stream_high(self.normalize(h))
        f_l = self.stream_low(self.normalize(l))

        f = torch.cat([f_h, f_l], dim=1)
        f = f * self.attn_gate(f)

        return self.classifier(f)


class MobileNetV3_DEXRT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = models.mobilenet_v3_small()

        in_f = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_f, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        return self.backbone(self.normalize(x))


class EfficientNetB0_DEXRT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = models.efficientnet_b0()

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        return self.backbone(self.normalize(x))


# ============================================================
# 2. PCI input and Soft APC input
# ============================================================
def build_pci_input(h, l):
    eps = 1e-6
    d = torch.log(h + eps) - torch.log(l + eps)
    return torch.cat([h, l, d], dim=1)


def build_soft_apc_input(
    h,
    l,
    m_ref=-0.5185,
    dead_zone=0.08,
    gain=1.1,
    max_comp=0.25
):
    eps = 1e-6

    log_h = torch.log(h + eps)
    log_l = torch.log(l + eps)

    m_curr = 0.5 * (log_h + log_l).mean().item()
    diff = m_ref - m_curr
    abs_d = abs(diff)

    if abs_d > dead_zone:
        d_comp = np.sign(diff) * (abs_d - dead_zone) * gain
    else:
        d_comp = 0.0

    d_comp = float(np.clip(d_comp, -max_comp, max_comp))

    h_c = torch.clamp(torch.exp(log_h + d_comp), 0, 1)
    l_c = torch.clamp(torch.exp(log_l + d_comp), 0, 1)

    d_c = torch.log(h_c + eps) - torch.log(l_c + eps)

    return torch.cat([h_c, l_c, d_c], dim=1)


# ============================================================
# 3. Parameter count
# ============================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ============================================================
# 4. Lightweight FLOPs counter
# ============================================================
def add_flops_counter(model):
    handles = []

    def conv_hook(module, inputs, output):
        x = inputs[0]
        batch_size = x.shape[0]
        out_channels = output.shape[1]
        out_h = output.shape[2]
        out_w = output.shape[3]
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups

        filters_per_channel = out_channels // groups
        conv_per_position_flops = kernel_h * kernel_w * in_channels * filters_per_channel
        active_elements_count = batch_size * out_h * out_w
        total_flops = conv_per_position_flops * active_elements_count

        if module.bias is not None:
            total_flops += out_channels * active_elements_count

        module.__flops__ += int(total_flops)

    def linear_hook(module, inputs, output):
        x = inputs[0]
        batch_size = x.shape[0] if x.dim() > 1 else 1
        total_flops = batch_size * module.in_features * module.out_features

        if module.bias is not None:
            total_flops += batch_size * module.out_features

        module.__flops__ += int(total_flops)

    def bn_hook(module, inputs, output):
        module.__flops__ += int(output.numel() * 2)

    def activation_hook(module, inputs, output):
        module.__flops__ += int(output.numel())

    def pool_hook(module, inputs, output):
        module.__flops__ += int(output.numel())

    for m in model.modules():
        m.__flops__ = 0

        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            handles.append(m.register_forward_hook(bn_hook))
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.Sigmoid)):
            handles.append(m.register_forward_hook(activation_hook))
        elif isinstance(m, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
            handles.append(m.register_forward_hook(pool_hook))

    return handles


def compute_flops(model, forward_fn):
    model.eval()

    handles = add_flops_counter(model)

    with torch.no_grad():
        _ = forward_fn()

    total_flops = 0
    for m in model.modules():
        total_flops += getattr(m, "__flops__", 0)

    for h in handles:
        h.remove()

    return total_flops / 1e9


# ============================================================
# 5. Latency measurement
# ============================================================
def measure_latency(forward_fn, device, warmup=100, runs=500):
    with torch.no_grad():
        for _ in range(warmup):
            _ = forward_fn()

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(runs):
            _ = forward_fn()

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

    latency_ms = (end - start) * 1000.0 / runs
    fps = 1000.0 / latency_ms

    return latency_ms, fps


# ============================================================
# 6. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    os.makedirs("results", exist_ok=True)

    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        print(f"CPU threads: {torch.get_num_threads()}")
    else:
        print("Using CUDA device.")

    h = torch.rand(1, 1, 192, 192).to(device)
    l = torch.rand(1, 1, 192, 192).to(device)
    x = build_pci_input(h, l)

    models_dict = {
        "Baseline": TICL_PCI_Net_Pro().to(device).eval(),
        "Proposed": TICL_PCI_Net_Pro().to(device).eval(),
        "CBAL2-Net": CBAL2_Net().to(device).eval(),
        "MobileNetV3-Small": MobileNetV3_DEXRT().to(device).eval(),
        "EfficientNet-B0": EfficientNetB0_DEXRT().to(device).eval()
    }

    records = []

    for name, model in models_dict.items():
        print(f"\nMeasuring {name}...")

        if name == "CBAL2-Net":
            # CBAL2-Net uses H and L as two separate streams.
            forward_for_flops = lambda m=model: m(h, l)
            forward_for_latency = lambda m=model: m(h, l)

        elif name == "Proposed":
            # FLOPs: report network FLOPs only, same backbone as Baseline.
            # Latency: include Soft APC preprocessing + network forward.
            forward_for_flops = lambda m=model: m(x)

            def forward_for_latency(m=model):
                x_apc = build_soft_apc_input(h, l)
                return m(x_apc)

        else:
            forward_for_flops = lambda m=model: m(x)
            forward_for_latency = lambda m=model: m(x)

        params_m = count_params(model)
        flops_g = compute_flops(model, forward_for_flops)

        latency_ms, fps = measure_latency(
            forward_for_latency,
            device=device,
            warmup=args.warmup,
            runs=args.runs
        )

        records.append({
            "Model": name,
            "Params_M": params_m,
            "FLOPs_G": flops_g,
            "Latency_ms": latency_ms,
            "FPS": fps,
            "Device": str(device),
            "Threads": args.threads if device.type == "cpu" else "N/A",
            "InputSize": "1x192x192",
            "LatencyNote": "Proposed latency includes Soft APC preprocessing"
            if name == "Proposed" else "Network forward only"
        })

    df = pd.DataFrame(records)

    df["Params_M"] = df["Params_M"].map(lambda x: f"{x:.2f}")
    df["FLOPs_G"] = df["FLOPs_G"].map(lambda x: f"{x:.3f}")
    df["Latency_ms"] = df["Latency_ms"].map(lambda x: f"{x:.2f}")
    df["FPS"] = df["FPS"].map(lambda x: f"{x:.2f}")

    if device.type == "cpu":
        out_path = f"results/MODEL_COMPLEXITY_LATENCY_{device}_{args.threads}threads.csv"
    else:
        out_path = f"results/MODEL_COMPLEXITY_LATENCY_{device}.csv"

    df.to_csv(out_path, index=False)

    print("\n" + "=" * 110)
    print("Model complexity and latency")
    print("=" * 110)
    print(df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()