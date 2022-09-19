"""
This script runs step 3 of the TrackML Quickstart example: Training the graph neural network.
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

sys.path.append("../../")

from Pipelines.TrackML_Example.LightningModules.GNN.Models.interaction_gnn import InteractionGNN
from utils.convenience_utils import headline

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("3_Train_GNN.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


def train(config_file="pipeline_config.yaml"):

    logging.info(headline(" Step 3: Running GNN training "))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    gnn_configs = all_configs["gnn_configs"]

    logging.info(headline("a) Initialising model" ))

    model = InteractionGNN(gnn_configs)

    logging.info(headline( "b) Running training" ))

    save_directory = os.path.join(common_configs["artifact_directory"], "gnn")
    logger = CSVLogger(save_directory, name=common_configs["experiment_name"])
    cmflogger = CMFLogger("mlmd", pipeline_name="exatrkx", pipeline_stage="3. Train GNN", execution_type="TrainGNN", graph=True)
    print("My Execution ID="+str(cmflogger._execution.id))

    trainer = Trainer(
        gpus=common_configs["gpus"],
        max_epochs=gnn_configs["max_epochs"],
        logger=[logger, cmflogger]
    )

    trainer.fit(model)

    logging.info(headline( "c) Saving model" ))
    
    os.makedirs(save_directory, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))

    cmf_logger = cmf.Cmf(filename="mlmd",pipeline_name="exatrkx", graph = True)
    context=cmf_logger.create_context(pipeline_stage="3. Train GNN") #TODO: custom_properties={"TBD":"TBD"}
    #execution=cmf_logger.create_execution(execution_type="TrainGNN", custom_properties = gnn_configs)
    #We update execution
    cmf_logger.update_execution(execution_id=cmflogger._execution.id, custom_properties = gnn_configs)


    cmf_logger.log_dataset(gnn_configs["input_dir"], "input") #TODO: custom_properties={"TBD":"TBD"}
    cmf_logger.log_dataset(gnn_configs["output_dir"],"output") #TODO: custom_properties={"TBD":"TBD"}

    cmf_logger.log_model(
        path=os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"), 
        event="output", 
        model_framework="PyTorchLightning", 
        model_type="GNN",
        model_name="GNN")



    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    train(config_file)    

