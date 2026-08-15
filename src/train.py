"""Train the base ResNet-20 on clean CIFAR-10 (host, MPS-accelerated).

Produces checkpoints/resnet20_cifar10.pt — the source model that all test-time
adaptation experiments start from. ~60-90 epochs reaches ~91-92% clean test acc;
for a quick Phase 0 smoke test, fewer epochs is fine.
"""
import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from models import resnet20
from data import clean_loaders


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "tinyimagenet"])
    ap.add_argument("--arch", default="resnet20", choices=["resnet20", "mobilenetv2"])
    ap.add_argument("--width", type=float, default=0.5, help="MobileNetV2 width multiplier")
    ap.add_argument("--stem-stride", type=int, default=None,
                    help="2 downsamples 64x64 inputs early; default 2 for tinyimagenet, else 1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    # Tiny-ImageNet (64x64) was trained with the strided stem; the released checkpoint
    # requires it. Note stem_stride does not change any weight shape, so a mismatch
    # loads silently and destroys accuracy -- hence the dataset-derived default.
    if args.stem_stride is None:
        args.stem_stride = 2 if args.dataset == "tinyimagenet" else 1
    if args.out is None:
        args.out = f"checkpoints/{args.arch}_{args.dataset}.pt"

    from models import build_model
    device = pick_device()
    print(f"device: {device} | arch: {args.arch} | dataset: {args.dataset}")
    if args.dataset == "tinyimagenet":
        import data_tin
        nc = data_tin.NUM_CLASSES
        train_loader, test_loader = data_tin.clean_loaders(args.data, args.batch_size)
    else:
        from data import num_classes
        nc = num_classes(args.dataset)
        train_loader, test_loader = clean_loaders(args.data, args.batch_size, dataset=args.dataset)

    model = build_model(args.arch, num_classes=nc, width=args.width, stem_stride=args.stem_stride).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        # quick test-acc check
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.numel()
        acc = 100.0 * correct / total
        print(f"epoch {epoch+1}: clean test acc {acc:.2f}%")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), args.out)
    print(f"best clean test acc {best:.2f}% -> {args.out}")


if __name__ == "__main__":
    main()
