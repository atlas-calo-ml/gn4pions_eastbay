# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_utils.ipynb (unless otherwise specified).

__all__ = ['convert_to_tuple']

# Cell
import numpy as np
import tensorflow as tf
from graph_nets.graphs import GraphsTuple

# Cell
def convert_to_tuple(graphs):
    nodes = []
    edges = []
    global_nodes = []
    senders = []
    receivers = []
    n_node = []
    n_edge = []
    offset = 0
    cluster_energies = []
    cluster_etas = []
    cluster_EM_probs = []
    cluster_calib_Es = []
    cluster_had_weights = []
    truth_particle_es = []
    truth_particle_pts = []
    track_pts = []
    track_etas = []
    sum_cluster_energies = []
    sum_lcw_energies = []

    for graph in graphs:
        nodes.append(graph['nodes'])
        edges.append(graph['edges'])
        global_nodes.append([graph['globals']])
        senders.append(graph['senders'] + offset)
        receivers.append(graph['receivers'] + offset)
        n_node.append(graph['nodes'].shape[:1])
        n_edge.append(graph['edges'].shape[:1])
        cluster_energies.append(graph['cluster_E_0'])
        cluster_etas.append(graph['cluster_eta'])
        cluster_EM_probs.append(graph['cluster_EM_prob'])
        cluster_calib_Es.append(graph['cluster_calib_E'])
        cluster_had_weights.append(graph['cluster_HAD_WEIGHT'])
        # truth_particle_es.append(graph['truthPartE'])
        # truth_particle_pts.append(graph['truthPartPt'])
        # track_pts.append(graph['track_pt'])
        # track_etas.append(graph['track_eta'])
        sum_cluster_energies.append(graph['sum_cluster_E'])
        sum_lcw_energies.append(graph['sum_lcw_E'])

        offset += len(graph['nodes'])

    nodes = tf.convert_to_tensor(np.concatenate(nodes))
    edges = tf.convert_to_tensor(np.concatenate(edges))
    global_nodes = tf.convert_to_tensor(np.concatenate(global_nodes))
    senders = tf.convert_to_tensor(np.concatenate(senders))
    receivers = tf.convert_to_tensor(np.concatenate(receivers))
    n_node = tf.convert_to_tensor(np.concatenate(n_node))
    n_edge = tf.convert_to_tensor(np.concatenate(n_edge))

    graph = GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=global_nodes,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge
        )

    return graph, cluster_energies, cluster_etas, cluster_EM_probs, cluster_calib_Es, cluster_had_weights, truth_particle_es, truth_particle_pts, track_pts, track_etas, sum_cluster_energies, sum_lcw_energies