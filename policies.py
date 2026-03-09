import torch


class UnmaskPolicy:
    name = "base"

    def select_positions(self, masked, probs, k):
        raise NotImplementedError


class LeftToRightPolicy(UnmaskPolicy):
    name = "left_to_right"

    def select_positions(self, masked, probs, k):
        bsz, seqlen = masked.shape
        out = torch.full((bsz, k), -1, dtype=torch.long, device=masked.device)
        for b in range(bsz):
            idx = torch.nonzero(masked[b], as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                take = idx[:k]
                out[b, : take.numel()] = take
        return out


class ConfidenceFirstPolicy(UnmaskPolicy):
    name = "confidence_first"

    def select_positions(self, masked, probs, k):
        bsz, seqlen, _ = probs.shape
        top1 = probs.max(dim=-1).values
        score = torch.where(masked, top1, torch.full_like(top1, -1.0))
        out = torch.full((bsz, k), -1, dtype=torch.long, device=masked.device)
        for b in range(bsz):
            valid = int(masked[b].sum().item())
            if valid == 0:
                continue
            take = min(k, valid)
            order = torch.argsort(score[b], descending=True, stable=True)
            out[b, :take] = order[:take]
        return out


class MarginFirstPolicy(UnmaskPolicy):
    name = "margin_first"

    def select_positions(self, masked, probs, k):
        top2 = torch.topk(probs, k=2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        score = torch.where(masked, margin, torch.full_like(margin, -1.0))
        bsz, _ = masked.shape
        out = torch.full((bsz, k), -1, dtype=torch.long, device=masked.device)
        for b in range(bsz):
            valid = int(masked[b].sum().item())
            if valid == 0:
                continue
            take = min(k, valid)
            order = torch.argsort(score[b], descending=True, stable=True)
            out[b, :take] = order[:take]
        return out


def build_policy(name):
    name = name.lower()
    if name == "left_to_right":
        return LeftToRightPolicy()
    if name == "confidence_first":
        return ConfidenceFirstPolicy()
    if name == "margin_first":
        return MarginFirstPolicy()
    raise ValueError(f"Unknown policy: {name}")
