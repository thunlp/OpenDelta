from functools import partial
from hashlib import sha1
from random import random
from sqlite3 import adapters
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
from opendelta.delta_models.lora import LowRankLinear

class _SplitLayer(nn.Module):
    r"""A layer of splitting module.
    """
    def __init__(self):
        super().__init__()
        self.module_dict = nn.ModuleDict()

    def attach(self, module_name: str, module: nn.Module):
        if module_name in self.module_dict:
            return False
        self.module_dict[module_name] = module
        return True

    def detach(self, module_name: str):
        if module_name not in self.module_dict:
            return False
        self.module_dict.pop(module_name)
        return True

class SplitSequentialLayer(_SplitLayer):
    def __init__(self):
        super().__init__()

    def post_forward(self, output):
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError

        split_outputs = []
        for module_name, module in self.module_dict.items():
            split_outputs.append( module.post_forward(
                hiddens
            ) )
        print(len(split_outputs))
        merge_output = torch.sum(torch.stack(split_outputs, dim=0), dim=0)

        if isinstance(output, tuple):
            output = (merge_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = merge_output
        else:
            raise TypeError
        return output

class SplitParallelLayer(_SplitLayer):
    r"""A layer of splitting module.
    """
    def __init__(self):
        super().__init__()

    def forward(self, hiddens):
        split_outputs = []
        for module_name, module in self.module_dict.items():
            split_outputs.append( module(
                hiddens
            ) )
        print(len(split_outputs))
        merge_output = torch.sum(torch.stack(split_outputs, dim=0), dim=0)
        return merge_output

class SplitConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`~SplitModel`

    """
    def __init__(
        self,
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
        - delta_type = "batch_split"

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
    delta_type = "batch_split"
    default_modified_modules = ["attn", "ff"]
    def __init__(self,
                 backbone_model: nn.Module,
                 modified_modules: Optional[List[str]] = [],
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

        self.modified_points = {}

    def add_all_delta_to_backbone(self,
                 backbone: nn.Module,
                 modified_modules: List[str],
                ) -> nn.Module:
        r"""The main function to add delta models to the backbone model based on the :obj:`modified_modules`.

        Args:
            backbone_model (:obj:`nn.Module`, *required*)  backbone model that the delta models are build opon. The
                modification to the backbone model are in place.
            modified_modules (:obj:`List[str]`, *optional*, default to :obj:`None`) The modules are subjected to update.
                leave this argument :obj:`None` will make the delta model return to the default setting, which add the delta
                models to the position experimented the paper. In this setting, the common structure mapping is loaded to
                addressing the corresponding modules.

        Returns:
            :obj:`nn.Module` The modified backbone model.

        """
        self.plm_total_params = sum(p.numel() for p in backbone.parameters())
        # create a new key list to avoid recursion.
        self.backbone_key_list = [key for key, _ in backbone.named_modules()]
        return backbone
        
    def attach(self, modified_point: Union[str, List[str]], module_name: str, module_type: str, module: Optional[nn.Module] = None):
        if isinstance(modified_point, str):
            modified_point = [modified_point]

        if module_type not in ["adapter", "lora"]:
            raise ValueError("module_type must be either adapter or lora")

        for key in self.backbone_key_list:
            if self.find_key(key, modified_point): # TODO may have bugs when commonstructure has a virtual node and it's refered
                logger.debug("find key: {}".format(key))
                _, _, ref = self.find_module(self.backbone_model, key)

                if key not in self.modified_points:
                    if module_type == "adapter":
                        splitlayer = SplitSequentialLayer()
                        self.insert_sequential_module(ref, delta_module=splitlayer, delta_name="split_sequential")
                    if module_type == "lora":
                        splitlayer = SplitParallelLayer()
                        self.insert_parallel_module(ref, delta_module=splitlayer, delta_name="split_parallel")
                    self.modified_points[key] = splitlayer

                splitlayer = self.modified_points[key] 
                if (module_type == "adapter" and not isinstance(splitlayer, SplitSequentialLayer)) or \
                    (module_type == "lora" and not isinstance(splitlayer, SplitParallelLayer)):
                    raise ValueError("one modified_point can have at most one module_type")

                if module is None:
                    if module_type == "adapter":
                        module = self.new_adapter_like(ref)
                    if module_type == "lora":
                        module = self.new_lora_like(ref)

                if not splitlayer.attach(module_name, module):
                    raise ValueError("another module with the same name '{}' has been added to {}".format(module_name, key))
                
    def update(self):
        self._pseudo_data_to_instantiate(self.backbone_model)
        self.mark_as_delta()

    def detach(self, modified_point: str, module_name: str):
        if isinstance(modified_point, str):
            modified_point = [modified_point]

        for key in self.backbone_key_list:
            if self.find_key(key, modified_point): # TODO may have bugs when commonstructure has a virtual node and it's refered
                logger.debug("find key: {}".format(key))
                _, _, ref = self.find_module(self.backbone_model, key)

                if key not in self.modified_points:
                    raise ValueError("no module has been added to {}".format(key))

                splitlayer = self.modified_points[key]

                if not splitlayer.detach(module_name):
                    raise ValueError("no module with the name '{}' has been added to {}".format(module_name, key))

    def new_adapter_like(self, module):
        adapterlayer = AdapterLayer()
        self.delta_modules.append(adapterlayer)
        return adapterlayer

    def new_lora_like(self, child_module):
        if isinstance(child_module, nn.Linear):
            in_features, out_features = child_module.in_features, child_module.out_features
            new_module = LowRankLinear(in_features = in_features,
                                     out_features = out_features,
                                     weight = child_module.weight,
                                     r=self.lora_r,
                                     lora_alpha=self.lora_alpha,
                                     lora_dropout=self.lora_dropout)
            self.delta_modules.append(new_module)
        else:
            raise NotImplementedError
        return new_module
