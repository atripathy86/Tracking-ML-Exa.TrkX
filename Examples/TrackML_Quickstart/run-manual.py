
import sys
sys.path.append("../../")
from Scripts import train_metric_learning, run_metric_learning_inference, train_gnn, run_gnn_inference, build_track_candidates, evaluate_candidates
from Scripts.utils.convenience_utils import get_example_data, plot_true_graph, get_training_metrics, plot_training_metrics, plot_neighbor_performance, plot_predicted_graph, plot_track_lengths, plot_edge_performance, plot_graph_sizes
import yaml

import warnings
warnings.filterwarnings("ignore")
CONFIG = 'pipeline_config.yaml'      

#Load Config
with open(CONFIG, 'r') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    
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


    
    