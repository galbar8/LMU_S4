from __future__ import annotations
import torch, torch.nn as nn
from typing import Dict, Any, Callable, Optional
from tqdm import tqdm

from src.utils.common import amp_autocast
from src.utils.metrics import top1
from src.utils.logging import Timer

CLIP_NORM = 1.0

def _unpack_batch(batch):
    # supports (x,y) or (x,y,meta)
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            xb, yb = batch; return xb, yb, None
        elif len(batch) == 3:
            xb, yb, meta = batch; return xb, yb, meta
    raise ValueError("Unexpected batch format. Expected (x,y) or (x,y,meta).")

def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    amp: bool,
    lossfn: nn.Module,
    ema = None,
    *,
    metrics_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]] = None,
    grad_clip: float = CLIP_NORM,
) -> Dict[str, Any]:
    """
    metrics_fn: optional callable computing extra metrics from (logits, targets).
                Return dict like {"acc": 0.93} or {"auroc": 0.88, "f1_micro": 0.71}.
                If None, we compute multiclass top1 'acc' by default.
    """
    model.train()
    tot_loss = 0.0
    sums: Dict[str, float] = {}  # accumulate metric * batch_size
    n = 0
    saw_opt_step = False

    with Timer() as t:
        for batch in tqdm(loader, desc="train", leave=False):
            xb, yb, _ = _unpack_batch(batch)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(amp):
                logits = model(xb)
                loss = lossfn(logits, yb)

            if amp and device.type == "cuda" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            saw_opt_step = True
            bs = xb.size(0)
            tot_loss += loss.item() * bs

            # Metrics: default to multiclass top1 if no custom metrics_fn
            if metrics_fn is None:
                val = top1(logits.detach(), yb)
                sums["acc"] = sums.get("acc", 0.0) + val * bs
            else:
                with torch.no_grad():
                    m = metrics_fn(logits.detach(), yb)
                for k, v in m.items():
                    sums[k] = sums.get(k, 0.0) + float(v) * bs

            n += bs
            if ema is not None:
                ema.update(model)

    out = {"loss": tot_loss / n, "time_s": t.dt, "lr": current_lr(optimizer), "stepped": saw_opt_step}
    # finalize averages
    for k, v in sums.items():
        out[k] = v / n
    return out

@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    amp: bool,
    loss_fn: nn.Module,
    ema = None,
    *,
    metrics_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    See train_one_epoch for metrics_fn contract.
    """
    if ema is not None:
        ema.apply(model)

    model.eval()
    tot_loss = 0.0
    sums: Dict[str, float] = {}
    n = 0

    with Timer() as t:
        for batch in tqdm(loader, desc="val", leave=False):
            xb, yb, _ = _unpack_batch(batch)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with amp_autocast(amp):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            bs = xb.size(0)
            tot_loss += loss.item() * bs

            if metrics_fn is None:
                val = top1(logits, yb)
                sums["acc"] = sums.get("acc", 0.0) + val * bs
            else:
                m = metrics_fn(logits, yb)
                for k, v in m.items():
                    sums[k] = sums.get(k, 0.0) + float(v) * bs

            n += bs

    if ema is not None:
        ema.restore(model)

    out = {"loss": tot_loss / n, "time_s": t.dt}
    for k, v in sums.items():
        out[k] = v / n
    return out
