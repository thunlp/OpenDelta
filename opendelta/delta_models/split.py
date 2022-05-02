from functools import partial
from random import random
from typing import Optional, Union

from cv2 import accumulate
from opendelta.utils.signature import get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
import loralib as lora
import torch.nn as nn
import torch
import math
from opendelta.delta_models.layers.activations import Activations
import inspect
from opendelta import BaseDeltaConfig
import opendelta.utils.logging as logging
import numpy as np
from opendelta import global_setting
logger = logging.get_logger(__name__)
from itertools import accumulate
from opendelta.delta_models.adapter import AdapterLayer


class SplitLayer(nn.Module):
    r"""A layer of splitting module.
    """
    def __init__(self, batch_size:list):
        super().__init__()
        self.batch_size = list(accumulate(batch_size))
        self.modulelist = nn.ModuleList()
        self.pseudo_inited = False

    def append(self, module):
        self.modulelist.append(module)

    def post_forward(self, output):
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError
        if hiddens.shape[0] != self.batch_size[-1]:
            if self.pseudo_inited:
                raise RuntimeError('The batch size of the input is not consistent with split config.')
            self.pseudo_inited = True
            outputs = None
            for i in range(len(self.batch_size)):
                outputs = self.modulelist[i].post_forward(
                    hiddens
                )
            merge_output = outputs
        else:
            split_outputs = [None]*len(self.batch_size)
            for i in range(len(self.batch_size)):
                split_outputs[i] = self.modulelist[i].post_forward(
                    hiddens[(0 if i==0 else self.batch_size[i-1]):self.batch_size[i]]
                )
            merge_output = torch.cat(split_outputs)

        if isinstance(output, tuple):
            output = (merge_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = merge_output
        else:
            raise TypeError
        return output

class SplitConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`~SplitModel`

    """
    def __init__(
        self,
        batch_size: list = [8, 1, 7],
        **kwargs
    ):
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])


class SplitModel(DeltaBase):
    r""" The implementation of Adapter(`Parameter-Efficient Transfer Learning for NLP <https://arxiv.org/abs/1902.00751>`_ ) .
    Add adapter to the designated ``modified_modules``. In sequential paradigm, The modules' output is then passed into the adapter's
    post_forward.

    .. note::
        We **assume** the output of the modified module is the hidden state or a tuple where hidden state is the
        first element. This is true for most PLMs. However, we admit that currently it's not rigorous, We will improve
        it in the next version. Currently, if you encount an error here for you backbone, you can modify the code to
        get the hidden state.

    class attributes:
        - default_modified_modules = ["attn", "ff"] According to the Adapter paper, we add adapter to the attention layer
          and feed forward layer.
        - delta_type = "adapter"

    Args:
        backbone_model (:obj:`transformers.PretrainedModels`): The backbone model to be modified.
        bottleneck_dim (:obj:`int`): The dimension of the adapter's bottleneck.
        non_linearity (:obj:`str`): The non linearity of the adapter.
        sequential (:obj:`str`): Whether insert the adapter in a sequential manner, as opposed to a parallel manner.
                        See `Towards a Unified View of Parameter-Efficient Transfer Learning <https://arxiv.org/abs/2110.04366>`_
                        for detail.
        modified_modules (:obj:`List[str]`): For prefix tuning, the it must refer to an attention layer (Currently, only
                        the implemented ones)
        unfrozen_modules (:obj:`List[str]`, *optional*, default to :obj:`None`): The modules that should be unfrozen
                         together with the prefix parameters.
        common_structure (:obj:`bool`): whether using name-based addressing with a common structure mapping.

    """
    config_class = SplitConfig
    delta_type = "adapter"
    default_modified_modules = ["attn", "ff"]
    def __init__(self,
                 backbone_model: nn.Module,
                 batch_size: list = [8, 1, 7],
                 modified_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 ):
        DeltaBase.__init__(self,
                           backbone_model,
                           modified_modules=modified_modules,
                           exclude_modules=exclude_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           interactive_modify=interactive_modify,
                           )
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.delta_modules = nn.ModuleList()

        self.add_all_delta_to_backbone(self.backbone_model,
                                   self.modified_modules,
                                   )

    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        splitlayer = SplitLayer(self.batch_size)
        for b in self.batch_size:
            splitlayer.append(self.new_module_like(ref))
        self.insert_sequential_module(ref, delta_module=splitlayer, delta_name="split")

    def new_module_like(self, module):
        module_device = get_device(module)
        adapterlayer = AdapterLayer()
        self.delta_modules.append(adapterlayer)
        return adapterlayer
