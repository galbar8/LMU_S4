from typing import Dict, Any, Tuple, Optional
from .spec import TaskSpec

# ==== ESC-50 bindings ====
from src.datasets.esc50.esc50_dataset2 import make_esc50_loaders

def _esc50_d_in(args: Dict[str, Any]) -> int:
    return args["n_mels"] if args.get("feature","melspec") == "melspec" else 1

def _esc50_theta(args: Dict[str, Any]) -> int:
    return int(args["target_num_frames"])

def _esc50_ncls(args: Dict[str, Any]) -> int:
    return 50

ESC50 = TaskSpec(
    name="esc50",
    make_loaders=lambda **kw: make_esc50_loaders(**kw),
    infer_input_dim=_esc50_d_in,
    infer_theta=_esc50_theta,
    infer_num_classes=_esc50_ncls,
    problem_type="multiclass",
    class_names=None,
)

# ==== PTB-XL bindings ====
# You already mentioned: make_ptbxl_loaders(data_root, batch_size, sampling="lr100", length=1000)
from src.datasets.ptbxl.ptbxl_dataset import make_ptbxl_loaders  # <- your function

def _ptbxl_d_in(args: Dict[str, Any]) -> int:
    # 12-lead ECG is common; if your loader supports variable leads, surface it via args
    return int(args.get("n_leads", 12))

def _ptbxl_theta(args: Dict[str, Any]) -> int:
    # length=1000 for lr100 in your example
    return int(args.get("length", 1000))

def _ptbxl_ncls(args: Dict[str, Any]) -> int:
    # Choose your label space (e.g., 5 superclasses or 21 diagnostics). Make it explicit in args.
    return int(args.get("n_classes", 5))

PTBXL = TaskSpec(
    name="ptbxl",
    make_loaders=lambda **kw: make_ptbxl_loaders(**kw),
    infer_input_dim=_ptbxl_d_in,
    infer_theta=_ptbxl_theta,
    infer_num_classes=_ptbxl_ncls,
    # PTB-XL is often multilabel; if you configured your dataset to be single label, flip this.
    problem_type="multilabel",
    class_names=None,
)

REGISTRY = {
    "esc50": ESC50,
    "ptbxl": PTBXL,
}

def get_task(name: str) -> TaskSpec:
    try:
        return REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown task '{name}'. Options: {list(REGISTRY)}")
