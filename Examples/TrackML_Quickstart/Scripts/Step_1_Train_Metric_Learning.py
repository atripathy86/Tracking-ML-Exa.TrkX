"""
This script runs step 1 of the TrackML Quickstart example: Training the metric learning model.
"""

import sys
import os
import yaml
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

from cmflib import cmf
from cmflib import dvc_wrapper

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import CMFLogger
import torch

sys.path.append("../../")

from Pipelines.TrackML_Example.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
from utils.convenience_utils import headline

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("1_Train_Metric_Learning.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


def train(config_file="pipeline_config.yaml"):

    logging.info(headline("Step 1: Running metric learning training"))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    metric_learning_configs = all_configs["metric_learning_configs"]

    logging.info(headline("a) Initialising model"))

    model = LayerlessEmbedding(metric_learning_configs)

    logging.info(headline("b) Running training" ))

    save_directory = os.path.join(common_configs["artifact_directory"], "metric_learning")
    logger = CSVLogger(save_directory, name=common_configs["experiment_name"])
    cmflogger = CMFLogger("mlmd", pipeline_name="exatrkx", pipeline_stage="1.TrainMetricLearning", execution_type="Train1", graph=True)
    print("My Execution ID="+str(cmflogger._execution.id))

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else None,
        gpus=common_configs["gpus"],
        max_epochs=metric_learning_configs["max_epochs"],
        logger=[logger,cmflogger]
    )

    trainer.fit(model)

    logging.info(headline("c) Saving model") )

    os.makedirs(save_directory, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))

    cmf_logger = cmf.Cmf(filename="mlmd",pipeline_name="exatrkx", graph = True)
    context=cmf_logger.create_context(pipeline_stage="1.TrainMetricLearning") #TODO: custom_properties={"TBD":"TBD"}
    #execution=cmf_logger.create_execution(execution_type="Train1", custom_properties = metric_learning_configs)
    cmf_logger.update_execution(execution_id=cmflogger._execution.id, custom_properties = metric_learning_configs)

    cmf_logger.log_dataset(metric_learning_configs["input_dir"], "input") #TODO: custom_properties={"TBD":"TBD"}
    cmf_logger.log_dataset(metric_learning_configs["output_dir"],"output") #TODO: custom_properties={"TBD":"TBD"}

    cmf_logger.log_model(
        path=os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"), 
        event="output", 
        model_framework="PyTorchLightning", 
        model_type="MLP Embedding",
        model_name="LayerlessEmbedding")

    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    trainer, model = train(config_file)    

