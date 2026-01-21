import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        else:
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        targets = targets.long()
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        logpt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -(1.0 - pt).pow(self.gamma) * logpt

        if self.alpha is not None:
            if self.alpha.numel() > 1:
                loss = loss * self.alpha.gather(0, targets)
            else:
                loss = loss * self.alpha

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# Assumes student/teacher models follow:
#   - student(x, domain_id=...) -> logits OR (logits, features) if extract_features=True
#   - teacher(x) -> logits OR (logits, features) if extract_features=True


class MultiDomainDistiller(nn.Module):
    """
    Multi-domain knowledge distillation with:
      - Task-ID routing (per-sample domain_id)
      - Subset-aware KD (match only the relevant logit slice per domain)
      - Optional feature-level distillation
      - Optional DSBN support via student(domain_id=...)
    """

    def __init__(
        self,
        student: nn.Module,
        teachers_dict: dict,
        domain_configs: dict,
        temperature: float = 6.0,
        w_ce: float = 0.3,
        w_kd: float = 0.5,
        w_feat: float = 0.2,
        alpha_by_domain: dict | None = None,
        feature_kd_weight: float = 1.0,
        use_feature_kd: bool = True,
    ):

        """
        Args:
            student: Student model (MultiDomainResNet18 with 32 classes)
            teachers_dict: {'chest': teacher1, 'brain': teacher2, 'organ': teacher3}
            domain_configs: {
                'chest': {'id': 0, 'start_idx': 0, 'end_idx': 4, 'num_classes': 4},
                'brain': {'id': 1, 'start_idx': 4, 'end_idx': 11, 'num_classes': 7},
                'organ': {'id': 2, 'start_idx': 11, 'end_idx': 22, 'num_classes': 11}
            }  # TOTAL_CLASSES = 22

            
            temperature: Temperature for KD
            w_ce/alpha_ce: Weight for cross-entropy loss
            w_kd/alpha_kd: Weight for KL divergence loss
            w_feat/alpha_feat: Weight for feature distillation loss
        """
        super().__init__()

        self.student = student
        self.teachers = nn.ModuleDict(teachers_dict)
        self.domain_configs = domain_configs

        # Loss weights
        self.temperature = float(temperature)
        self.w_ce = float(w_ce)
        self.w_kd = float(w_kd)
        self.w_feat = float(w_feat)

        self.use_feature_kd = bool(use_feature_kd)
        self.feature_kd_weight = float(feature_kd_weight)

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()

        self.alpha_by_domain = alpha_by_domain or {}

        # Create focal loss per domain (ModuleDict so it follows .to(device))
        self.ce_loss_by_domain = nn.ModuleDict()
        for domain_name, cfg in domain_configs.items():
            alpha_vec = self.alpha_by_domain.get(domain_name, None)
            self.ce_loss_by_domain[domain_name] = SoftmaxFocalLoss(
                gamma=2.0,
                alpha=alpha_vec,
                reduction="mean",
            )

        # Enable feature extraction flags (if your model uses this convention)
        if hasattr(self.student, "extract_features"):
            self.student.extract_features = self.use_feature_kd

        for teacher in self.teachers.values():
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            if self.use_feature_kd and hasattr(teacher, "extract_features"):
                teacher.extract_features = True

        # Domain mapping
        self.domain_to_id = {name: cfg["id"] for name, cfg in domain_configs.items()}
        self.id_to_domain = {cfg["id"]: name for name, cfg in domain_configs.items()}

        self.num_domains = len(self.domain_to_id)

        # Total output classes (avoid hard-coding 32)
        self.total_classes = getattr(self.student, "num_classes", None)
        if self.total_classes is None:
            self.total_classes = max(cfg["end_idx"] for cfg in domain_configs.values())

        # Statistics
        self.training_stats = defaultdict(list)
        self.current_epoch = 0

    def train(self, mode: bool = True):
        """
        Override train() so:
          - student follows mode
          - teachers always stay in eval mode
        """
        super().train(mode)
        self.student.train(mode)
        for teacher in self.teachers.values():
            teacher.eval()
        return self

    @staticmethod
    def _maybe_bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
        """
        Converts BHWC -> BCHW if it looks like channels-last.
        Heuristic:
          - x is 4D
          - second dim is NOT (1 or 3)
          - last dim IS (1 or 3)
        """
        if x.dim() == 4 and x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def _global_to_local_labels(
        self,
        global_labels: torch.Tensor,
        domain_name: str,
        domain_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert global labels to local (domain-specific) labels.
        domain_mask is accepted for compatibility; if provided, global_labels can be full-batch.
        """
        cfg = self.domain_configs[domain_name]
        start_idx = int(cfg["start_idx"])

        if domain_mask is not None:
            global_labels = global_labels[domain_mask]

        local = global_labels - start_idx
        return local

    def _compute_kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between student and teacher softened distributions.
        Shapes must match: [N, C_domain].
        """
        if student_logits.shape != teacher_logits.shape:
            # Skip KD if mismatch rather than crashing; fix configs/teacher heads if this triggers.
            return torch.zeros((), device=student_logits.device)

        t = self.temperature
        student_logp = F.log_softmax(student_logits / t, dim=-1)
        teacher_p = F.softmax(teacher_logits / t, dim=-1)

        kd = self.kl_loss(student_logp, teacher_p) * (t ** 2)

        if torch.isnan(kd) or torch.isinf(kd):
            # Return a safe zero (grad will be zero)
            return torch.zeros((), device=student_logits.device)

        return kd

    def _compute_feature_loss(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        """
        Feature-level distillation loss:
          - cosine loss + MSE on L2-normalized features
        """
        if student_features.shape != teacher_features.shape:
            # Skip if features don't align; you may need an adapter/projection.
            return torch.zeros((), device=student_features.device)

        s = F.normalize(student_features, p=2, dim=1)
        t = F.normalize(teacher_features, p=2, dim=1)

        cosine_loss = 1.0 - (s * t).sum(dim=1).mean()
        mse_loss = self.mse_loss(s, t)

        return 0.5 * cosine_loss + 0.5 * mse_loss

    def forward_train(self, image: torch.Tensor, target: torch.Tensor, task_ids: torch.Tensor):
        """
        Training forward pass with task-ID routing.

        Args:
            image: [B, C, H, W] or [B, H, W, C]
            target: global target labels [B] (e.g., 0..total_classes-1)
            task_ids: domain IDs [B] (e.g., 0..num_domains-1)

        Returns:
            full_logits: [B, total_classes]
            losses: dict with keys: ce, kd, feat(optional), total
        """
        image = self._maybe_bhwc_to_bchw(image)

        batch_size = image.size(0)
        device = image.device

        # Ensure integer task ids on correct device
        if task_ids.dtype != torch.long:
            task_ids = task_ids.long()
        task_ids = task_ids.to(device)

        # Use tensor accumulators to avoid float/tensor mixing issues
        total_ce_loss = torch.zeros((), device=device)
        total_kd_loss = torch.zeros((), device=device)
        total_feat_loss = torch.zeros((), device=device)
        samples_processed = 0

        # Fill full logits directly from per-domain student outputs (no extra forward pass)
        full_logits = torch.zeros(batch_size, self.total_classes, device=device, dtype=torch.float32)

        for domain_name, cfg in self.domain_configs.items():
            domain_id = int(cfg["id"])
            start_idx = int(cfg["start_idx"])
            end_idx = int(cfg["end_idx"])

            domain_mask = (task_ids == domain_id)
            num_domain_samples = int(domain_mask.sum().item())
            if num_domain_samples == 0:
                continue

            domain_images = image[domain_mask]
            domain_global_labels = target[domain_mask].to(device)

            # Convert to local labels
            domain_local_labels = self._global_to_local_labels(domain_global_labels, domain_name, domain_mask=None)

            # Student forward (domain-specific BN via domain_id)
            if self.use_feature_kd:
                out = self.student(domain_images, domain_id=domain_id)
                if not isinstance(out, tuple):
                    raise ValueError("Student is expected to return (logits, features) when use_feature_kd=True")
                student_full_logits, student_features = out
            else:
                student_full_logits = self.student(domain_images, domain_id=domain_id)
                if isinstance(student_full_logits, tuple):
                    student_full_logits = student_full_logits[0]
                student_features = None

            # Put student logits back into the right batch positions
            student_full_logits = student_full_logits.float()          # safe for AMP
            if student_features is not None:
                student_features = student_features.float()

            
            full_logits[domain_mask] = student_full_logits           
            student_domain_logits = student_full_logits[:, start_idx:end_idx]

            # Teacher forward
            teacher = self.teachers[domain_name]
            with torch.no_grad():
                if self.use_feature_kd:
                    t_out = teacher(domain_images)
                    if not isinstance(t_out, tuple):
                        raise ValueError("Teacher is expected to return (logits, features) when use_feature_kd=True")
                    teacher_logits, teacher_features = t_out
                else:
                    teacher_logits = teacher(domain_images)
                    if isinstance(teacher_logits, tuple):
                        teacher_logits = teacher_logits[0]
                    teacher_features = None

            # Losses
            ce_loss = self.ce_loss_by_domain[domain_name](student_domain_logits, domain_local_labels)
            kd_loss = self._compute_kd_loss(student_domain_logits, teacher_logits)

            feat_loss = torch.zeros((), device=device)
            if self.use_feature_kd and (student_features is not None) and (teacher_features is not None):
                feat_loss = self._compute_feature_loss(student_features, teacher_features)

            # Weighted accumulation by number of samples in this domain
            n = float(num_domain_samples)
            total_ce_loss = total_ce_loss + ce_loss * n
            total_kd_loss = total_kd_loss + kd_loss * n
            if self.use_feature_kd:
                total_feat_loss = total_feat_loss + feat_loss * n

            samples_processed += num_domain_samples

            # Stats
            self.training_stats[f"{domain_name}_ce_loss"].append(float(ce_loss.detach().cpu().item()))
            self.training_stats[f"{domain_name}_kd_loss"].append(float(kd_loss.detach().cpu().item()))
            if self.use_feature_kd:
                self.training_stats[f"{domain_name}_feat_loss"].append(float(feat_loss.detach().cpu().item()))

        if samples_processed > 0:
            denom = float(samples_processed)
            total_ce_loss = total_ce_loss / denom
            total_kd_loss = total_kd_loss / denom
            if self.use_feature_kd:
                total_feat_loss = total_feat_loss / denom

        # Total loss
        total_loss = (self.w_ce * total_ce_loss) + (self.w_kd * total_kd_loss)
        if self.use_feature_kd:
            total_loss = total_loss + (self.w_feat * total_feat_loss * self.feature_kd_weight)

        losses = {"ce": total_ce_loss, "kd": total_kd_loss, "total": total_loss}
        if self.use_feature_kd:
            losses["feat"] = total_feat_loss

        return full_logits, losses

    def forward_test(self, image: torch.Tensor, task_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Test/validation forward pass.

        If task_ids is None:
          - defaults to domain 0 for all samples (only meaningful if you know the domain beforehand).
        """
        image = self._maybe_bhwc_to_bchw(image)

        batch_size = image.size(0)
        device = image.device

        if task_ids is None:
            task_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            if task_ids.dtype != torch.long:
                task_ids = task_ids.long()
            task_ids = task_ids.to(device)

        logits = torch.zeros(batch_size, self.total_classes, device=device)

        # Group by domain for proper BN routing
        for domain_id in range(self.num_domains):
            mask = (task_ids == domain_id)
            if not mask.any():
                continue

            domain_imgs = image[mask]
            out = self.student(domain_imgs, domain_id=domain_id)
            if isinstance(out, tuple):
                out = out[0]
            logits[mask] = out.to(logits.dtype)

        return logits

    def forward(self, image: torch.Tensor, target: torch.Tensor | None = None, task_ids: torch.Tensor | None = None):
        if self.training:
            if target is None or task_ids is None:
                raise ValueError("target and task_ids are required during training")
            return self.forward_train(image, target, task_ids)
        return self.forward_test(image, task_ids)

    def update_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def get_training_statistics(self):
        stats = {}
        for key, values in self.training_stats.items():
            if values:
                stats[f"avg_{key}"] = float(np.mean(values[-100:]))
                stats[f"current_{key}"] = float(values[-1])
        return stats

    def reset_training_statistics(self):
        self.training_stats = defaultdict(list)
