
import sys
sys.path.append("../../")
from Scripts import train_metric_learning, run_metric_learning_inference, train_gnn, run_gnn_inference, build_track_candidates, evaluate_candidates
from Scripts.utils.convenience_utils import get_example_data, plot_true_graph, get_training_metrics, plot_training_metrics, plot_neighbor_performance, plot_predicted_graph, plot_track_lengths, plot_edge_performance, plot_graph_sizes
import yaml

import warnings
warnings.filterwarnings("ignore")
CONFIG_ORIG = 'pipeline_config-default.yaml'
CONFIG = 'pipeline_config.yaml'


#Update config
def update_config(configs):
    with open(CONFIG,'w') as file:
        updated_configs=yaml.dump(configs, file)
        file.close()

def execution(CONFIG):
    #1. Train Metric Learning
    metric_learning_trainer, metric_learning_model = train_metric_learning(CONFIG)
    #embedding_metrics = get_training_metrics(metric_learning_trainer)

    #2. Construct graphs from metric learning inference
    graph_builder = run_metric_learning_inference(CONFIG)

    #3. Train graph neural networks
    gnn_trainer, gnn_model = train_gnn(CONFIG)

    #4 GNN inference
    run_gnn_inference(CONFIG)

    #5 Build Track Candidates from GNN
    build_track_candidates(CONFIG)

    #6 Evaluate Track Candidates
    evaluated_events, reconstructed_particles, particles, matched_tracks, tracks = evaluate_candidates(CONFIG)


#Load Config
configs={}
with open(CONFIG_ORIG, 'r') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

#Run Defaults
execution(CONFIG)    

#emb_hidden_opts=[ 512 , 256, 2048]
#for x in emb_hidden_opts:
    #print(configs["metric_learning_configs"]["emb_hidden"])
#    configs["metric_learning_configs"]["emb_hidden"] = x
#    update_config(configs)
#    #execution(CONFIG)

#16 did not run
# metric_nb_layer_opts=[2,8,16]
# for x in metric_nb_layer_opts:
#     configs["metric_learning_configs"]["nb_layer"] = x
#     update_config(configs)
#     execution(CONFIG)

# n_graph_hidden_opts=[64,256,512]
# for x in n_graph_hidden_opts:
#     configs["gnn_configs"]["hidden"] = x
#     update_config(configs)
#     execution(CONFIG)

# n_graph_nb_node_layer_opts=[6,9,12]
# for x in n_graph_nb_node_layer_opts:
#     configs["gnn_configs"]["nb_node_layer"] = x
#     update_config(configs)
#     execution(CONFIG)
    
    
# n_graph_nb_node_layer_opts=[6,9,12]
# for x in n_graph_nb_node_layer_opts:
#     configs["gnn_configs"]["nb_node_layer"] = x
#     configs["gnn_configs"]["nb_edge_layer"] = x
#     update_config(configs)
#     execution(CONFIG)
    

#GNN fails with: RuntimeError: CUDA out of memory. Tried to allocate 3.15 GiB (GPU 0; 15.78 GiB total capacity; 3.47 GiB already allocated; 1.76 GiB free; 9.30 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
# configs["metric_learning_configs"]["emb_hidden"] = 128
# configs["gnn_configs"]["hidden"] = 32
# print("--------------------Now executing 128x32------------------")
# update_config(configs)
# execution(CONFIG)

# configs["metric_learning_configs"]["emb_hidden"] = 128
# configs["gnn_configs"]["hidden"] = 64
# print("--------------------Now executing 128x64------------------")
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["emb_hidden"] = 128
# configs["gnn_configs"]["hidden"] = 128
# print("--------------------Now executing 128x128------------------")
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["emb_hidden"] = 128
# configs["gnn_configs"]["hidden"] = 256
# print("--------------------Now executing 128x256------------------")
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["emb_hidden"] = 256
# configs["gnn_configs"]["hidden"] = 32
# print("--------------------Now executing 256x32------------------")
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["emb_hidden"] = 256
# configs["gnn_configs"]["hidden"] = 64
# print("--------------------Now executing 256x64------------------")
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["emb_hidden"] = 256
# configs["gnn_configs"]["hidden"] = 256
# print("--------------------Now executing 256x256------------------")
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["emb_hidden"] = 512
# configs["gnn_configs"]["hidden"] = 32
# print("--------------------Now executing 512x32------------------")
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["emb_hidden"] = 512
# configs["gnn_configs"]["hidden"] = 64
# print("--------------------Now executing 512x64------------------")
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["emb_hidden"] = 512
# configs["gnn_configs"]["hidden"] = 256
# print("--------------------Now executing 512x256------------------")
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["emb_hidden"] = 1024
# configs["gnn_configs"]["hidden"] = 32
# print("--------------------Now executing 1024x32------------------")
# update_config(configs)
# execution(CONFIG)

# configs["metric_learning_configs"]["emb_hidden"] = 1024
# configs["gnn_configs"]["hidden"] = 64
# print("--------------------Now executing 1024x64------------------")
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["emb_hidden"] = 2048
# configs["gnn_configs"]["hidden"] = 32
# print("--------------------Now executing 2048x32------------------")
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["emb_hidden"] = 2048
# configs["gnn_configs"]["hidden"] = 64
# print("--------------------Now executing 2048x64------------------")
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["emb_hidden"] = 2048
# configs["gnn_configs"]["hidden"] = 256
# print("--------------------Now executing 2048x256------------------")
# update_config(configs)
# execution(CONFIG)


## One time
# configs["metric_learning_configs"]["emb_hidden"] = 1024
# configs["metric_learning_configs"]["nb_layer"] = 16 
# update_config(configs)
# execution(CONFIG)


# configs["metric_learning_configs"]["nb_layer"] = 12
# configs["gnn_configs"]["nb_node_layer"] = 6 
# update_config(configs)
# execution(CONFIG)

#3x12 did not run
# n_graph_nb_node_layer_opts=[3] #Removed 6,9,12 since it has run
# metric_nb_layer_opts=[2,4,8,12] #removed 16. Removed 2,4 since it has run
# for x in n_graph_nb_node_layer_opts:
#     for y in metric_nb_layer_opts:
#         configs["gnn_configs"]["nb_node_layer"] = x
#         configs["metric_learning_configs"]["nb_layer"] = y
#         update_config(configs)
#         execution(CONFIG)










    