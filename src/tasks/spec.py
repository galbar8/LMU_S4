from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple

@dataclass
class TaskSpec:
    name: str
    make_loaders: Callable[..., Tuple[Any, Any, Optional[Any]]]
    # Returns (train_loader, val_loader, test_loader|None)

    # Infer model/data details from args (so one Trainer can be reused)
    infer_input_dim: Callable[[Dict[str, Any]], int]       # -> d_in
    infer_theta: Callable[[Dict[str, Any]], int]           # -> sequence length / frames
    infer_num_classes: Callable[[Dict[str, Any]], int]     # -> n_classes

    # Classification type decides loss + metrics
    problem_type: str  # "multiclass" | "multilabel"

    # Optional nice-to-haves
    class_names: Optional[list] = None
