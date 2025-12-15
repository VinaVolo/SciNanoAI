from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from cellpose import core, dynamics, io, models, train

from src.data.datasets import DatasetBundle, MaskDatasetBuilder
from src.train.config import CellposeTrainingConfig
from src.train.metrics import InstanceMetrics


@dataclass
class TrainingResult:
    model_path: Path
    train_losses: np.ndarray
    test_losses: np.ndarray
    metrics_history: Dict[str, np.ndarray]
    loss_components: Dict[str, np.ndarray]


class CellposeTrainer:
    """OOP wrapper around Cellpose training with validation and logging."""

    def __init__(self, config: CellposeTrainingConfig):
        self.config = config
        self.device: Optional[torch.device] = None
        self.model: Optional[models.CellposeModel] = None
        self.net = None
        self._flow_cache: dict[str, np.ndarray] = {}
        self._class_weights: torch.Tensor | None = None

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

    def train(self, dataset: DatasetBundle | None = None) -> TrainingResult:
        self._initialize_model()
        self._patch_get_batch()
        bundle = dataset or self._build_datasets()
        self.config.ensure_output_dirs()
        return self._train_seg_timed(bundle)

    def _initialize_model(self) -> None:
        use_gpu = core.use_gpu()
        # Cellpose 4.x does not support training from scratch; when pretrained_model is not
        # provided we fall back to the library default pretrained weights.
        if self.config.pretrained_model is None:
            self.model = models.CellposeModel(gpu=use_gpu)
        else:
            self.model = models.CellposeModel(gpu=use_gpu, pretrained_model=self.config.pretrained_model)
        self.net = self.model.net
        self.device = self.net.device

        if self.config.freeze_encoder:
            for name, param in self.net.named_parameters():
                param.requires_grad = "encoder.neck" in name

    def _build_datasets(self) -> DatasetBundle:
        builder = MaskDatasetBuilder(
            train_dir=self.config.train_dir,
            test_dir=self.config.test_dir,
            image_suffix=self.config.image_suffix,
            mask_suffix=self.config.mask_suffix,
        )
        bundle = builder.build(
            max_train_images=self.config.max_train_images,
            max_test_images=self.config.max_test_images,
            seed=self.config.random_seed,
        )

        if bundle.missing_train or bundle.missing_test:
            train.train_logger.warning(
                "Found images without masks: train=%d, test=%d",
                len(bundle.missing_train),
                len(bundle.missing_test),
            )

        if not bundle.train.images or not bundle.test.images:
            raise ValueError("No train/test data found. Please check dataset paths and suffixes.")

        return bundle

    def _patch_get_batch(self) -> None:
        flow_cache = self._flow_cache
        trainer = self

        def _get_batch_with_channel_axis(
            inds,
            data=None,
            labels=None,
            files=None,
            labels_files=None,
            channel_axis=None,
            normalize_params=None,
        ):
            if normalize_params is None:
                normalize_params = {"normalize": False}

            def ensure_flows(lbl, lbl_path=None):
                if lbl.ndim >= 3 and lbl.shape[0] >= 3:
                    return lbl[1:] if lbl.shape[0] >= 4 else lbl
                if lbl_path and lbl_path in flow_cache:
                    return flow_cache[lbl_path]
                flow = dynamics.labels_to_flows([lbl], device=trainer.device, return_flows=True)[0]
                flow = flow[1:]
                if lbl_path:
                    flow_cache[lbl_path] = flow
                return flow

            if data is None:
                lbls = None
                imgs = [io.imread(files[i]) for i in inds]
                imgs = train._reshape_norm(imgs, channel_axis=channel_axis, normalize_params=normalize_params)
                if labels_files is not None:
                    raw_lbls = [io.imread(labels_files[i]) for i in inds]
                    lbls = [ensure_flows(lbl, labels_files[i]) for lbl, i in zip(raw_lbls, inds)]
            else:
                imgs = [data[i] for i in inds]
                lbls_raw = [labels[i] for i in inds]
                lbls = [ensure_flows(lbl) for lbl in lbls_raw]
            return imgs, lbls

        train._get_batch = _get_batch_with_channel_axis

    def _train_seg_timed(self, dataset: DatasetBundle) -> TrainingResult:
        cfg = self.config
        train_files = dataset.train.images
        train_labels_files = dataset.train.masks
        test_files = dataset.test.images
        test_labels_files = dataset.test.masks

        original_net_dtype = None
        if self.device and self.device.type == "mps" and self.net.dtype == torch.bfloat16:
            original_net_dtype = torch.bfloat16
            train.train_logger.warning(
                "Training with bfloat16 on MPS is not supported, switching network to float32",
            )
            self.net.dtype = torch.float32
            self.net.to(torch.float32)

        scale_range = 0.5 if cfg.scale_range is None else cfg.scale_range

        if isinstance(cfg.normalize, dict):
            normalize_params = {**models.normalize_default, **cfg.normalize}
        elif not isinstance(cfg.normalize, bool):
            raise ValueError("normalize parameter must be a bool or a dict")
        else:
            normalize_params = dict(models.normalize_default)
            normalize_params["normalize"] = cfg.normalize

        out = train._process_train_test(
            train_data=None,
            train_labels=None,
            train_files=train_files,
            train_labels_files=train_labels_files,
            train_probs=None,
            test_data=None,
            test_labels=None,
            test_files=test_files,
            test_labels_files=test_labels_files,
            test_probs=None,
            load_files=False,
            min_train_masks=cfg.min_train_masks,
            compute_flows=cfg.compute_flows,
            channel_axis=cfg.channel_axis,
            normalize_params=normalize_params,
            device=self.device,
        )
        (
            train_data,
            train_labels,
            train_files,
            train_labels_files,
            train_probs,
            diam_train,
            test_data,
            test_labels,
            test_files,
            test_labels_files,
            test_probs,
            diam_test,
            normed,
        ) = out

        get_batch_kwargs = {} if normed else {"normalize_params": normalize_params, "channel_axis": cfg.channel_axis}
        self.net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(self.device)

        if cfg.class_weights is not None:
            self._class_weights = torch.as_tensor(cfg.class_weights, dtype=torch.float32, device=self.device)

        nimg = len(train_data) if train_data is not None else len(train_files)
        nimg_test = len(test_data) if test_data is not None else len(test_files)
        if train_probs is None:
            train_probs = np.ones(nimg, dtype=np.float32) / float(nimg)
        nimg_per_epoch = nimg if cfg.nimg_per_epoch is None else cfg.nimg_per_epoch
        nimg_test_per_epoch = nimg_test if cfg.nimg_test_per_epoch is None else cfg.nimg_test_per_epoch

        lr_schedule = self._build_lr_schedule(cfg.n_epochs, cfg.learning_rate)
        train.train_logger.info(">>> n_epochs=%d, n_train=%d, n_test=%s", cfg.n_epochs, nimg, nimg_test)
        train.train_logger.info(
            ">>> AdamW, learning_rate=%0.5f, weight_decay=%0.5f",
            cfg.learning_rate,
            cfg.weight_decay,
        )
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        t0 = time.time()
        model_dir = cfg.model_dir
        filename = model_dir / "weights.pth"
        train.train_logger.info(">>> saving model to %s", filename)

        lavg, nsum = 0.0, 0
        last_log_time = time.time()
        last_test_loss = None

        train_losses = np.zeros(cfg.n_epochs)
        test_losses = np.zeros(cfg.n_epochs)
        loss_components = {
            "train_flow": np.zeros(cfg.n_epochs),
            "train_prob": np.zeros(cfg.n_epochs),
            "train_cls": np.zeros(cfg.n_epochs),
            "test_flow": np.zeros(cfg.n_epochs),
            "test_prob": np.zeros(cfg.n_epochs),
            "test_cls": np.zeros(cfg.n_epochs),
        }
        metrics_history = {
            "box_precision": np.zeros(cfg.n_epochs),
            "box_recall": np.zeros(cfg.n_epochs),
            "box_map50": np.zeros(cfg.n_epochs),
            "box_map5095": np.zeros(cfg.n_epochs),
            "mask_precision": np.zeros(cfg.n_epochs),
            "mask_recall": np.zeros(cfg.n_epochs),
            "mask_map50": np.zeros(cfg.n_epochs),
            "mask_map5095": np.zeros(cfg.n_epochs),
        }

        for iepoch in range(cfg.n_epochs):
            np.random.seed(iepoch)
            if nimg != nimg_per_epoch:
                rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,), p=train_probs)
            else:
                rperm = np.random.permutation(np.arange(0, nimg))

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[iepoch]
            self.net.train()

            for k in range(0, nimg_per_epoch, cfg.batch_size):
                kend = min(k + cfg.batch_size, nimg_per_epoch)
                inds = rperm[k:kend]
                imgs, lbls = train._get_batch(
                    inds,
                    data=train_data,
                    labels=train_labels,
                    files=train_files,
                    labels_files=train_labels_files,
                    **get_batch_kwargs,
                )
                diams = np.array([diam_train[i] for i in inds])
                rsc = diams / self.net.diam_mean.item() if cfg.rescale else np.ones(len(diams), "float32")
                imgi, lbl = train.random_rotate_and_resize(
                    imgs, Y=lbls, rescale=rsc, scale_range=scale_range, xy=(cfg.bsize, cfg.bsize)
                )[:2]
                X = torch.from_numpy(imgi).to(self.device)
                lbl = torch.from_numpy(lbl).to(self.device)

                if X.dtype != self.net.dtype:
                    X = X.to(self.net.dtype)
                    lbl = lbl.to(self.net.dtype)

                y = self.net(X)[0]
                loss, lf, lp, lc = self._compute_loss(lbl, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss = loss.item() * len(imgi)
                lavg += train_loss
                nsum += len(imgi)
                train_losses[iepoch] += train_loss
                loss_components["train_flow"][iepoch] += lf * len(imgi)
                loss_components["train_prob"][iepoch] += lp * len(imgi)
                loss_components["train_cls"][iepoch] += lc * len(imgi)

            train_losses[iepoch] /= nimg_per_epoch
            loss_components["train_flow"][iepoch] /= nimg_per_epoch
            loss_components["train_prob"][iepoch] /= nimg_per_epoch
            loss_components["train_cls"][iepoch] /= nimg_per_epoch

            eval_out = self._evaluate_validation(
                epoch_idx=iepoch,
                net=self.net,
                test_data=test_data,
                test_labels=test_labels,
                test_files=test_files,
                test_labels_files=test_labels_files,
                get_batch_kwargs=get_batch_kwargs,
                diam_test=diam_test,
                nimg_test=nimg_test,
                nimg_test_per_epoch=nimg_test_per_epoch,
                batch_size=cfg.batch_size,
                scale_range=scale_range,
                bsize=cfg.bsize,
                rescale=cfg.rescale,
                channel_axis=cfg.channel_axis,
                normalize_params=normalize_params,
                model_for_eval=self.model,
                view_dir=cfg.view_dir,
            )

            if eval_out is not None:
                test_loss_val, tlf, tlp, tlc, metrics_out, vis_path = eval_out
                test_losses[iepoch] = test_loss_val
                loss_components["test_flow"][iepoch] = tlf
                loss_components["test_prob"][iepoch] = tlp
                loss_components["test_cls"][iepoch] = tlc
                last_test_loss = test_loss_val
                if metrics_out is not None:
                    metrics_history["box_precision"][iepoch] = metrics_out["box_precision"]
                    metrics_history["box_recall"][iepoch] = metrics_out["box_recall"]
                    metrics_history["box_map50"][iepoch] = metrics_out["box_map50"]
                    metrics_history["box_map5095"][iepoch] = metrics_out["box_map5095"]
                    metrics_history["mask_precision"][iepoch] = metrics_out["mask_precision"]
                    metrics_history["mask_recall"][iepoch] = metrics_out["mask_recall"]
                    metrics_history["mask_map50"][iepoch] = metrics_out["mask_map50"]
                    metrics_history["mask_map5095"][iepoch] = metrics_out["mask_map5095"]
                if vis_path:
                    train.train_logger.info("Saved validation view to %s", vis_path)

            train.train_logger.info(
                "[epoch %d] train_loss=%.4f (flow=%.4f, prob=%.4f, cls=%.4f) "
                "val_loss=%.4f (flow=%.4f, prob=%.4f, cls=%.4f) "
                "Box P/R=%.3f/%.3f Box mAP50=%.3f "
                "Mask P/R=%.3f/%.3f Mask mAP50=%.3f",
                iepoch,
                train_losses[iepoch],
                loss_components["train_flow"][iepoch],
                loss_components["train_prob"][iepoch],
                loss_components["train_cls"][iepoch],
                test_losses[iepoch],
                loss_components["test_flow"][iepoch],
                loss_components["test_prob"][iepoch],
                loss_components["test_cls"][iepoch],
                metrics_history["box_precision"][iepoch],
                metrics_history["box_recall"][iepoch],
                metrics_history["box_map50"][iepoch],
                metrics_history["mask_precision"][iepoch],
                metrics_history["mask_recall"][iepoch],
                metrics_history["mask_map50"][iepoch],
            )

            now = time.time()
            if now - last_log_time >= cfg.log_every_seconds:
                tl = train_losses[iepoch]
                vll = last_test_loss if last_test_loss is not None else 0.0
                train.train_logger.info(
                    "[timed] epoch=%d train_loss=%.4f test_loss=%.4f elapsed=%.1fs",
                    iepoch,
                    tl,
                    vll,
                    now - t0,
                )
                last_log_time = now

            self._save_plots(
                current_epoch=iepoch,
                train_losses=train_losses,
                test_losses=test_losses,
                loss_components=loss_components,
                metrics_history=metrics_history,
                log_dir=cfg.logging_dir,
            )

            if iepoch == cfg.n_epochs - 1 or (iepoch % cfg.save_every == 0 and iepoch != 0):
                filename0 = filename if not cfg.save_each or iepoch == cfg.n_epochs - 1 else model_dir / f"weights_epoch_{iepoch:04d}.pth"
                train.train_logger.info("saving network parameters to %s", filename0)
                self.net.save_model(filename0)

        self.net.save_model(filename)

        if original_net_dtype is not None:
            self.net.dtype = original_net_dtype
            self.net.to(original_net_dtype)

        return TrainingResult(
            model_path=filename,
            train_losses=train_losses,
            test_losses=test_losses,
            metrics_history=metrics_history,
            loss_components=loss_components,
        )

    def _compute_loss(self, lbl, y):
        criterion = torch.nn.MSELoss(reduction="mean")
        criterion2 = torch.nn.BCEWithLogitsLoss(reduction="mean")
        veci = 5.0 * lbl[:, -2:]
        loss_flow = criterion(y[:, -3:-1], veci) / 2.0
        loss_prob = criterion2(y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype))
        loss_cls = None
        if y.shape[1] > 3:
            criterion3 = torch.nn.CrossEntropyLoss(reduction="mean", weight=self._class_weights)
            loss_cls = criterion3(y[:, :-3], lbl[:, 0].long())
        total_loss = loss_flow + loss_prob + (loss_cls if loss_cls is not None else 0.0)
        return total_loss, loss_flow.item(), loss_prob.item(), (loss_cls.item() if loss_cls is not None else 0.0)

    def _evaluate_validation(
        self,
        *,
        epoch_idx: int,
        net,
        test_data,
        test_labels,
        test_files,
        test_labels_files,
        get_batch_kwargs,
        diam_test,
        nimg_test,
        nimg_test_per_epoch,
        batch_size,
        scale_range,
        bsize,
        rescale,
        channel_axis,
        normalize_params,
        model_for_eval,
        view_dir: Path,
    ):
        if test_data is None and test_files is None:
            return None

        np.random.seed(42)
        lavgt = 0.0
        loss_flow_sum = 0.0
        loss_prob_sum = 0.0
        loss_cls_sum = 0.0
        n_seen = 0
        rperm = np.random.permutation(np.arange(0, nimg_test))
        rperm = rperm[:nimg_test_per_epoch]
        for ibatch in range(0, len(rperm), batch_size):
            with torch.no_grad():
                net.eval()
                inds = rperm[ibatch : ibatch + batch_size]
                imgs, lbls = train._get_batch(
                    inds,
                    data=test_data,
                    labels=test_labels,
                    files=test_files,
                    labels_files=test_labels_files,
                    **get_batch_kwargs,
                )
                diams = np.array([diam_test[i] for i in inds])
                rsc = diams / net.diam_mean.item() if rescale else np.ones(len(diams), "float32")
                imgi, lbl = train.random_rotate_and_resize(
                    imgs, Y=lbls, rescale=rsc, scale_range=scale_range, xy=(bsize, bsize)
                )[:2]
                X = torch.from_numpy(imgi).to(self.device)
                lbl = torch.from_numpy(lbl).to(self.device)

                if X.dtype != net.dtype:
                    X = X.to(net.dtype)
                    lbl = lbl.to(net.dtype)

                y = net(X)[0]
                loss, lf, lp, lc = self._compute_loss(lbl, y)
                test_loss = loss.item() * len(imgi)
                lavgt += test_loss
                loss_flow_sum += lf * len(imgi)
                loss_prob_sum += lp * len(imgi)
                loss_cls_sum += lc * len(imgi)
                n_seen += len(imgi)
        lavgt /= len(rperm)
        loss_flow_mean = loss_flow_sum / max(1, n_seen)
        loss_prob_mean = loss_prob_sum / max(1, n_seen)
        loss_cls_mean = loss_cls_sum / max(1, n_seen)

        metrics_out = None
        vis_path = None
        if model_for_eval is not None and test_files is not None and len(test_files) > 0:
            eval_count = nimg_test_per_epoch if nimg_test_per_epoch is not None else len(test_files)
            eval_indices = np.random.choice(
                np.arange(0, len(test_files)), size=min(len(test_files), eval_count), replace=False
            )
            pred_boxes_all, pred_scores_all, gt_boxes_all = [], [], []
            pred_masks_all, pred_mask_scores_all, gt_masks_all = [], [], []
            sample_idx = int(random.choice(eval_indices)) if len(eval_indices) > 0 else None

            for idx in eval_indices:
                img = io.imread(test_files[idx])
                gt_mask_raw = io.imread(test_labels_files[idx]) if test_labels_files is not None else None
                gt_mask = None
                if gt_mask_raw is not None:
                    if gt_mask_raw.ndim > 2:
                        gt_mask = gt_mask_raw[0] if gt_mask_raw.shape[0] > 1 else gt_mask_raw.squeeze()
                    else:
                        gt_mask = gt_mask_raw
                masks_pred, flows_pred, _ = model_for_eval.eval(
                    img,
                    channel_axis=channel_axis,
                    normalize=normalize_params,
                    compute_masks=True,
                    tile_overlap=0.1,
                    bsize=bsize,
                    augment=False,
                )
                if isinstance(masks_pred, list):
                    masks_pred = masks_pred[0]
                if isinstance(flows_pred, list) and len(flows_pred) > 0 and isinstance(flows_pred[0], list):
                    flows_pred = flows_pred[0]
                cellprob_map = None
                if isinstance(flows_pred, list) and len(flows_pred) >= 3:
                    cellprob_map = flows_pred[2]
                boxes_pred, scores_pred, inst_masks_pred = InstanceMetrics.mask_to_instances(
                    masks_pred if isinstance(masks_pred, np.ndarray) else None, score_map=cellprob_map
                )
                boxes_gt, _, inst_masks_gt = InstanceMetrics.mask_to_instances(
                    gt_mask if gt_mask is not None else None, score_map=None
                )
                pred_boxes_all.append(boxes_pred)
                pred_scores_all.append(scores_pred)
                gt_boxes_all.append(boxes_gt)
                pred_masks_all.append(inst_masks_pred)
                pred_mask_scores_all.append(scores_pred)
                gt_masks_all.append(inst_masks_gt)

                if sample_idx is not None and idx == sample_idx:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(img)
                    if gt_mask is not None:
                        axes[0].imshow(gt_mask, cmap="nipy_spectral", alpha=0.35)
                    axes[0].set_title("GT")
                    axes[0].axis("off")
                    axes[1].imshow(img)
                    axes[1].imshow(masks_pred, cmap="nipy_spectral", alpha=0.35)
                    axes[1].set_title("Pred")
                    axes[1].axis("off")
                    plt.tight_layout()
                    vis_path = view_dir / f"epoch_{epoch_idx:04d}.png"
                    plt.savefig(vis_path, dpi=200)
                    plt.close(fig)

            thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
            box_precision, box_recall, box_map50, box_map5095 = InstanceMetrics.evaluate_map(
                pred_boxes_all, pred_scores_all, gt_boxes_all, thresholds, InstanceMetrics.boxes_iou
            )
            mask_precision, mask_recall, mask_map50, mask_map5095 = InstanceMetrics.evaluate_map(
                pred_masks_all, pred_mask_scores_all, gt_masks_all, thresholds, InstanceMetrics.mask_iou
            )
            metrics_out = {
                "box_precision": box_precision,
                "box_recall": box_recall,
                "box_map50": box_map50,
                "box_map5095": box_map5095,
                "mask_precision": mask_precision,
                "mask_recall": mask_recall,
                "mask_map50": mask_map50,
                "mask_map5095": mask_map5095,
            }

        return lavgt, loss_flow_mean, loss_prob_mean, loss_cls_mean, metrics_out, vis_path

    def _save_plots(
        self,
        *,
        current_epoch: int,
        train_losses: np.ndarray,
        test_losses: np.ndarray,
        loss_components: Dict[str, np.ndarray],
        metrics_history: Dict[str, np.ndarray],
        log_dir: Path,
    ) -> None:
        ep_range = np.arange(current_epoch + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(ep_range, train_losses[: current_epoch + 1], label="train total")
        axes[0].plot(ep_range, test_losses[: current_epoch + 1], label="val total")
        axes[0].plot(ep_range, loss_components["train_flow"][: current_epoch + 1], "--", label="train flow")
        axes[0].plot(ep_range, loss_components["train_prob"][: current_epoch + 1], "--", label="train prob")
        axes[0].plot(ep_range, loss_components["train_cls"][: current_epoch + 1], "--", label="train cls")
        axes[0].plot(ep_range, loss_components["test_flow"][: current_epoch + 1], ":", label="val flow")
        axes[0].plot(ep_range, loss_components["test_prob"][: current_epoch + 1], ":", label="val prob")
        axes[0].plot(ep_range, loss_components["test_cls"][: current_epoch + 1], ":", label="val cls")
        axes[0].set_title("Loss components")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(ep_range, metrics_history["box_precision"][: current_epoch + 1], label="Precision (Box)")
        axes[1].plot(ep_range, metrics_history["box_recall"][: current_epoch + 1], label="Recall (Box)")
        axes[1].plot(ep_range, metrics_history["box_map50"][: current_epoch + 1], label="mAP50 (Box)")
        axes[1].plot(ep_range, metrics_history["box_map5095"][: current_epoch + 1], label="mAP50-95 (Box)")
        axes[1].set_title("Detection (boxes)")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True)
        axes[1].legend()

        axes[2].plot(ep_range, metrics_history["mask_precision"][: current_epoch + 1], label="Precision (Mask)")
        axes[2].plot(ep_range, metrics_history["mask_recall"][: current_epoch + 1], label="Recall (Mask)")
        axes[2].plot(ep_range, metrics_history["mask_map50"][: current_epoch + 1], label="mAP50 (Mask)")
        axes[2].plot(ep_range, metrics_history["mask_map5095"][: current_epoch + 1], label="mAP50-95 (Mask)")
        axes[2].set_title("Segmentation (masks)")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True)
        axes[2].legend()

        plt.tight_layout()
        plot_path = log_dir / "metrics.png"
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)

    @staticmethod
    def _build_lr_schedule(n_epochs: int, learning_rate: float) -> np.ndarray:
        lr = np.linspace(0, learning_rate, 10)
        lr = np.append(lr, learning_rate * np.ones(max(0, n_epochs - 10)))
        if n_epochs > 300:
            lr = lr[:-100]
            for _ in range(10):
                lr = np.append(lr, lr[-1] / 2 * np.ones(10))
        elif n_epochs > 99:
            lr = lr[:-50]
            for _ in range(10):
                lr = np.append(lr, lr[-1] / 2 * np.ones(5))
        return lr
