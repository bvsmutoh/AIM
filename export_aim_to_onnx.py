import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn

# --- repo paths --------------------------------------------------------------
repo_root = Path(__file__).resolve().parent
sys.path.append(str(repo_root / "core"))  # so we can `from network.AimNet import AimNet`

# --- imports from your repo ---------------------------------------------------
from core.network.AimNet import AimNet


# --- utils -------------------------------------------------------------------
def remove_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

class OutputPicker(nn.Module):
    """Wrap AimNet to export just one matte tensor as [N,1,H,W]."""
    def __init__(self, model: nn.Module, which: str = "fusion"):
        super().__init__()
        self.model = model
        self.which = which  # 'fusion' | 'local' | 'global'

    def forward(self, x):
        global_sigmoid, local_sigmoid, fusion_sigmoid = self.model(x)
        if self.which == "fusion":
            y = fusion_sigmoid
        elif self.which == "local":
            y = local_sigmoid
        elif self.which == "global":
            # global is 3 channels; convert to single-channel matte via mean
            # (or change to any rule you prefer)
            if global_sigmoid.dim() == 4 and global_sigmoid.size(1) == 3:
                y = global_sigmoid.mean(dim=1, keepdim=True)
            else:
                y = global_sigmoid
        else:
            raise ValueError(f"Unknown output '{self.which}'")
        # Ensure [N,1,H,W]
        if y.dim() == 3:
            y = y.unsqueeze(1)
        if y.size(1) != 1:
            y = y.mean(dim=1, keepdim=True)
        return y

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(repo_root / "models" / "pretrained" / "aimnet_pretrained_matting.pth"))
    p.add_argument("--backbone", default=str(repo_root / "models" / "pretrained" / "r34mp_pretrained_imagenet.pth.tar"),
                  help="If your resnet_mp tries to load this at runtime, keep the default or point to the file.")
    p.add_argument("--output", default=str(repo_root / "models" / "aim_matte.onnx"))
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--opset", type=int, default=13)
    p.add_argument("--out", choices=["fusion", "local", "global"], default="fusion",
                  help="Which AimNet output to export.")
    p.add_argument("--dynamic", action="store_true",
                  help="Enable dynamic height/width axes in the exported ONNX graph.")
    args = p.parse_args()

    # Sanity: weights present?
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing AIMNet checkpoint: {ckpt_path}")

    # Optional: check backbone if your resnet loader requires it at init time
    backbone_path = Path(args.backbone)
    if not backbone_path.exists():
        print(f"[WARN] Backbone weights not found at {backbone_path}. "
              "If your resnet tries to load them, it may fall back to random init.")

    # Build model + load weights
    model = AimNet().eval()
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = remove_module_prefix(state)
    model.load_state_dict(state, strict=False)

    # Wrap to select the desired output
    model = OutputPicker(model, which=args.out).eval()

    # Dummy input (H and W are dynamic, this is just a sample)
    dummy = torch.randn(1, 3, args.image_size, args.image_size)

    # Export
    export_kwargs = {}
    if args.dynamic:
        export_kwargs["dynamic_axes"] = {
            "image": {2: "h", 3: "w"},
            "alpha": {2: "h", 3: "w"},
        }

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["alpha"],
        opset_version=args.opset,
        do_constant_folding=True,
        **export_kwargs,
    )
    print(f"Exported {args.out} matte to: {args.output}")

if __name__ == "__main__":
    main()
