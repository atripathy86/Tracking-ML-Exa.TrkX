"""
This script runs step 2 of the TrackML Quickstart example: Inferencing the metric learning to construct graphs.
"""

import sys
import os
import yaml
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
import torch

from cmflib import cmf
from cmflib import dvc_wrapper

sys.path.append("../../")

from Pipelines.TrackML_Example.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
from utils.convenience_utils import headline, delete_directory
from Pipelines.TrackML_Example.notebooks.build_embedding import EmbeddingInferenceBuilder

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("2_Run_Metric_Learning.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()

def train(config_file="pipeline_config.yaml"):

    logging.info(headline("Step 2: Constructing graphs from metric learning model"))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    metric_learning_configs = all_configs["metric_learning_configs"]

    logging.info(headline("a) Loading trained model"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayerlessEmbedding.load_from_checkpoint(os.path.join(common_configs["artifact_directory"], "metric_learning", common_configs["experiment_name"]+".ckpt")).to(device)

    logging.info(headline("b) Running inferencing"))
    if common_configs["clear_directories"]:
        delete_directory(metric_learning_configs["output_dir"])

    graph_builder = EmbeddingInferenceBuilder(model, metric_learning_configs["train_split"], overwrite=True, knn_max=1000, radius=metric_learning_configs["r_test"])
    graph_builder.build()

    cmf_logger = cmf.Cmf(filename="mlmd",pipeline_name="exatrkx", graph = True)
    context=cmf_logger.create_context(pipeline_stage="2. Metric Learning Inference") #TODO: custom_properties={"TBD":"TBD"}
    execution=cmf_logger.create_execution(execution_type="Infer1", custom_properties = metric_learning_configs)

    cmf_logger.log_dataset(metric_learning_configs["output_dir"], "input") #TODO: custom_properties={"TBD":"TBD"}
    #cmf_logger.log_dataset(metric_learning_configs["output_dir"],"output") #TODO: custom_properties={"TBD":"TBD"}

    save_directory = os.path.join(common_configs["artifact_directory"], "metric_learning")
    cmf_logger.log_model(
        path=os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"), 
        event="input", 
        model_framework="PyTorchLightning", 
        model_type="MLP Embedding",
        model_name="LayerlessEmbedding")

    return graph_builder


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    gb = train(config_file) 