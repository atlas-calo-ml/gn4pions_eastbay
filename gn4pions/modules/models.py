# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_models.ipynb (unless otherwise specified).


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


__all__ = ['make_mlp_model', 'MLPGraphIndependent', 'MLPGraphNetwork', 'MLPDeepSets', 'MultiOutWeightedRegressModel']

# Cell
#nbdev_comment from __future__ import absolute_import
#nbdev_comment from __future__ import division
#nbdev_comment from __future__ import print_function

from graph_nets import modules
from graph_nets import utils_tf
from six.moves import range
import sonnet as snt
import tensorflow as tf

# Cell
_EDGE_BLOCK_OPT = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": True,
    }

_NODE_BLOCK_OPT = {
    "use_received_edges": True,
    "use_sent_edges": False,
    "use_nodes": True,
    "use_globals": True,
    }

_GLOBAL_BLOCK_OPT = {
    "use_edges": False,
    "use_nodes": True,
    "use_globals": True,
    }

# Cell
def make_mlp_model(latent_size, num_layers, activate_final=True):
        """Instantiates a new MLP, followed by LayerNorm.

        The parameters of each new MLP are not shared with others generated by
        this function.

        Returns:
        A Sonnet module which contains the MLP and LayerNorm.
        """
        return snt.Sequential([
            snt.nets.MLP([latent_size] * num_layers, activate_final=activate_final),
            snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
        ])

# Cell
class MLPGraphIndependent(snt.Module):
    """GraphIndependent with MLP edge, node, and global models."""
    def __init__(self,
               edge_model_fn=None,
               node_model_fn=None,
               global_model_fn=None,
               name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        self._network = modules.GraphIndependent(
            edge_model_fn=edge_model_fn,
            node_model_fn=node_model_fn,
            global_model_fn=global_model_fn)

    def __call__(self, inputs):
        return self._network(inputs)

# Cell
class MLPGraphNetwork(snt.Module):
    """GraphNetwork with MLP edge, node, and global models."""
    def __init__(self, name='MLPGraphNetwork', **model_config):
        super(MLPGraphNetwork, self).__init__(name=name)

        if model_config['reducer'] == 'mean':
            reducer=tf.math.unsorted_segment_mean
        elif model_config['reducer'] == 'sum':
            reducer=tf.math.unsorted_segment_sum

        mlp_fn = lambda: make_mlp_model(model_config['latent_size'], model_config['num_layers'])

        self._network = modules.GraphNetwork(
            edge_model_fn=mlp_fn,
            node_model_fn=mlp_fn,
            global_model_fn=mlp_fn,
            reducer=reducer,
            edge_block_opt=model_config['edge_block_opt'],
            node_block_opt=model_config['node_block_opt'],
            global_block_opt=model_config['global_block_opt'])


    def __call__(self, inputs):
        return self._network(inputs)

# Cell
class MLPDeepSets(snt.Module):
    """Deepsets with MLP node, and global models."""
    def __init__(self, name='MLPDeepSets', **model_config):
        super(MLPDeepSets, self).__init__(name=name)

        if model_config['reducer'] == 'mean':
            reducer=tf.math.unsorted_segment_mean
        elif model_config['reducer'] == 'sum':
            reducer=tf.math.unsorted_segment_sum

        mlp_fn = lambda: make_mlp_model(model_config['latent_size'], model_config['num_layers'])

        self._network = modules.DeepSets(
            node_model_fn=mlp_fn,
            global_model_fn=mlp_fn,
            reducer=reducer)


    def __call__(self, inputs):
        return self._network(inputs)

# Cell
class MultiOutWeightedRegressModel(snt.Module):
    """

    """
    def __init__(self,
               global_output_size=1,
               num_outputs=1,
               model_config=None,
               name="MultiOutWeightedRegressModel"):
        super(MultiOutWeightedRegressModel, self).__init__(name=name)

        self._num_blocks = model_config['num_blocks']
        self._concat_input = model_config['concat_input']
        self._model_config = model_config
        self._num_outputs = num_outputs

        if self._model_config['block_type'] == 'graphnet':
            block_type = MLPGraphNetwork
        elif self._model_config['block_type'] == 'deepsets':
            block_type = MLPDeepSets


        self._core = [
                block_type(name="core_"+str(i), **self._model_config) for i in range(self._num_blocks)
                ]


        # Transforms the outputs into the appropriate shapes.
        edge_fn = None
        node_fn = None
        global_fn = []

        for i in range(self._num_outputs):
            global_fn.append(lambda: snt.Linear(global_output_size, name="global_output_"+str(i)))

        self._output_transform = []
        for i in range(self._num_outputs):
            self._output_transform.append(modules.GraphIndependent(
                edge_fn, node_fn, global_fn[i], name="network_output_"+str(i)))

        global_regress_fn = lambda: snt.Linear(1, name="regress_linear")
        self._regress_transform =  modules.GraphIndependent(
                edge_fn, node_fn, global_regress_fn, name="reression_output")

    def __call__(self, input_op):
        latent = self._core[0](input_op)
        latent_all = [input_op]
        for i in range(1, self._num_blocks):
            if self._concat_input:
                core_input = utils_tf.concat([latent, latent_all[-1]], axis=1)
            else:
                core_input = latent

            latent_all.append(latent)
            latent = self._core[i](core_input)

        latent_all.append(latent)
        stacked_latent = utils_tf.concat(latent_all, axis=1)
        independent_output = []
        for i in range(self._num_outputs):
            independent_output.append(self._output_transform[i](stacked_latent))

        stacked_independent = utils_tf.concat(independent_output, axis=1)
        regress_output = self._regress_transform(stacked_independent)
        class_output = independent_output[1]
        return regress_output, class_output