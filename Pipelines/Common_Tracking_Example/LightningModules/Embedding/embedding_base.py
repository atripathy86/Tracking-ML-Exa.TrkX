"""The base classes for the embedding process.

The embedding process here is both the embedding models (contained in Models/) and the training procedure, which is a Siamese network strategy. Here, the network is run on all points, and then pairs (or triplets) are formed by one of several strategies (e.g. random pairs (rp), hard negative mining (hnm)) upon which some sort of contrastive loss is applied. The default here is a hinge margin loss, but other loss functions can work, including cross entropy-style losses. Also available are triplet loss approaches.

Example:
    See Quickstart for a concrete example of this process.
    
Todo:
    * Refactor the training & validation pipeline, since the use of the different regimes (rp, hnm, etc.) looks very messy
"""

# System imports
import sys
import os
import logging

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_cluster import radius_graph
import numpy as np

# Local Imports
from .utils import graph_intersection, split_datasets, build_edges

device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)

    def setup(self, stage):
        self.trainset, self.valset, self.testset = split_datasets(**self.hparams)
        print(torch.cuda.max_memory_allocated() / 1024**3, "Gb")

    def train_dataloader(self):
        if len(self.trainset) > 0:
            return DataLoader(self.trainset, batch_size=1, num_workers=0)
        else:
            return None

    def val_dataloader(self):
        if len(self.valset) > 0:
            return DataLoader(self.valset, batch_size=1, num_workers=0)
        else:
            return None

    def test_dataloader(self):
        if len(self.testset):
            return DataLoader(self.testset, batch_size=1, num_workers=1)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    def get_input_data(self, batch):

        if self.hparams["cell_channels"] > 0:
            input_data = torch.cat(
                [batch.cell_data[:, : self.hparams["cell_channels"]], batch.x], axis=-1
            )
            input_data[input_data != input_data] = 0
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0

        return input_data

    def get_query_points(self, batch, spatial):

        if "query_all_points" in self.hparams["regime"]:
            query_indices = torch.arange(len(spatial)).to(spatial.device)
        elif "query_noise_points" in self.hparams["regime"]:
            query_indices = torch.cat(
                [torch.where(batch.pid == 0)[0], batch.signal_true_edges.unique()]
            )
        else:
            query_indices = batch.signal_true_edges.unique()

        query_indices = query_indices[torch.randperm(len(query_indices))][
            : self.hparams["points_per_batch"]
        ]
        query = spatial[query_indices]

        return query_indices, query

    def append_hnm_pairs(self, e_spatial, query, query_indices, spatial, r_train=None, knn=None):
        if r_train is None:
            r_train = self.hparams["r_train"]
        if knn is None:
            knn = self.hparams["knn"]

        if "low_purity" in self.hparams["regime"]:
            knn_edges = build_edges(
                query, spatial, query_indices, r_train, 500
            )
            knn_edges = knn_edges[
                :,
                torch.randperm(knn_edges.shape[1])[
                    : int(r_train * len(query))
                ],
            ]

        else:
            knn_edges = build_edges(
                query,
                spatial,
                query_indices,
                r_train,
                knn,
            )

        e_spatial = torch.cat(
            [
                e_spatial,
                knn_edges,
            ],
            axis=-1,
        )

        return e_spatial

    def append_random_pairs(self, e_spatial, query_indices, spatial):
        n_random = int(self.hparams["randomisation"] * len(query_indices))
        indices_src = torch.randint(
            0, len(query_indices), (n_random,), device=self.device
        )
        indices_dest = torch.randint(0, len(spatial), (n_random,), device=self.device)
        random_pairs = torch.stack([query_indices[indices_src], indices_dest])

        e_spatial = torch.cat(
            [e_spatial, random_pairs],
            axis=-1,
        )
        return e_spatial

    def get_true_pairs(self, e_spatial, y_cluster, e_bidir, new_weights=None):
        e_spatial = torch.cat(
            [
                e_spatial.to(self.device),
                e_bidir,
            ],
            axis=-1,
        )
        y_cluster = torch.cat([y_cluster.int(), torch.ones(e_bidir.shape[1])])
        if new_weights is not None:
            new_weights = torch.cat(
                [new_weights, torch.ones(e_bidir.shape[1], device=self.device)]
            )
        return e_spatial, y_cluster, new_weights

    def get_hinge_distance(self, spatial, e_spatial, y_cluster):

        logging.info(f"Memory at hinge start: {torch.cuda.max_memory_allocated() / 1024**3} Gb")

        hinge = y_cluster.float().to(self.device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        try:
            d = torch.sum((reference - neighbors) ** 2, dim=-1)
        except RuntimeError:
            # Need to perform this step in chunks?
            d = []
            for ref, nei in zip(reference.chunk(10), neighbors.chunk(10)):
                d.append(torch.sum((ref - nei) ** 2, dim=-1))
            d = torch.cat(d)

        logging.info(f"Memory at hinge end: {torch.cuda.max_memory_allocated() / 1024**3} Gb")
        return hinge, d

    def get_truth(self, batch, e_spatial, e_bidir):

        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        return e_spatial, y_cluster

    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        logging.info(f"Memory at train start: {torch.cuda.max_memory_allocated() / 1024**3} Gb")

        # Instantiate empty prediction edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Forward pass of model, handling whether Cell Information (ci) is included
        input_data = self.get_input_data(batch)

        with torch.no_grad():
            spatial = self(input_data)

        query_indices, query = self.get_query_points(batch, spatial)
        
        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            e_spatial = self.append_hnm_pairs(e_spatial, query, query_indices, spatial)

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            e_spatial = self.append_random_pairs(e_spatial, query_indices, spatial)

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        e_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], axis=-1
        )

        # Calculate truth from intersection between Prediction graph and Truth graph
        if "module_veto" in self.hparams["regime"]:
            e_spatial = e_spatial[:, batch.modules[e_spatial][0] != batch.modules[e_spatial][1]]
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)
        new_weights = y_cluster.to(self.device)

        # Append all positive examples and their truth and weighting
        e_spatial, y_cluster, new_weights = self.get_true_pairs(
            e_spatial, y_cluster, new_weights, e_bidir
        )

        included_hits = e_spatial.unique()
        spatial[included_hits] = self(input_data[included_hits])

        hinge, d = self.get_hinge_distance(spatial, e_spatial, y_cluster)

        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == -1],
            hinge[hinge == -1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == 1],
            hinge[hinge == 1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        loss = negative_loss + self.hparams["weight"] * positive_loss

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False):

        input_data = self.get_input_data(batch)
        spatial = self(input_data)

        e_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], axis=-1
        )

        # Build whole KNN graph
        e_spatial = build_edges(
            spatial, spatial, indices=None, r_max=knn_radius, k_max=knn_num
        )

        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)

        hinge, d = self.get_hinge_distance(
            spatial, e_spatial.to(self.device), y_cluster
        )

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"]**2, reduction="mean"
        )

        cluster_true = e_bidir.shape[1]
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])

        eff = cluster_true_positive / cluster_true
        pur = cluster_true_positive / cluster_positive
        if "module_veto" in self.hparams["regime"]:
            module_veto_pur = cluster_true_positive / (batch.modules[e_spatial[0]] != batch.modules[e_spatial[1]]).sum()
        else:
            module_veto_pur = 0
        
        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"val_loss": loss, "eff": eff, "pur": pur, "module_veto_pur": module_veto_pur, "current_lr": current_lr}
            )
        logging.info("Efficiency: {}".format(eff))
        logging.info("Purity: {}".format(pur))
        logging.info(batch.event_file)

        return {
            "loss": loss,
            "distances": d,
            "preds": e_spatial,
            "truth": y_cluster,
            "truth_graph": e_bidir,
        }

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        knn_val = 500 if "knn_val" not in self.hparams else self.hparams["knn_val"]
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_val"], knn_val, log=True
        )

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_test"], 1000, log=False
        )

        return outputs

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
        logging.info("Optimizer step for batch {}".format(batch_idx))
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

        logging.info("Optimizer step done for batch {}".format(batch_idx))
