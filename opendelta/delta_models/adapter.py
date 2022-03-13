from functools import partial
from random import random
from typing import Optional, Union
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
logger = logging.get_logger(__name__)

class AdapterLayer(nn.Module):
    r"""A layer of adapter tuning module. 
    """
    layer_count = 0

    @classmethod
    def count_layer(cls):
        cls.layer_count += 1
    
    @classmethod
    def get_layer_count(cls):
        return cls.layer_count

    def __init__(self, bottleneck_dim=24, non_linearity='gelu_new', device=None):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.device = device
        self.instantiated = False
        self.non_linearity = non_linearity
        
        self.layer_id = AdapterLayer.get_layer_count()
        AdapterLayer.count_layer()
        
    
    def instantiate(self, hidden_dim):
        self.modulelist = nn.Sequential()
        self.modulelist.add_module("down_proj",nn.Linear(hidden_dim, self.bottleneck_dim, device=self.device))

        # select non-linearity
        self.modulelist.add_module("non_linear", Activations(self.non_linearity.lower()))

        self.modulelist.add_module("up_proj", nn.Linear(self.bottleneck_dim, self.hidden_dim,  device=self.device))

        # TODO:
        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        # if self.add_layer_norm_after:
        #     self.adapter_norm_after = nn.LayerNorm(self.input_size)

        self.instantiated = True
        # initialize the weight, which is important for fast convergence and better performance. 
        self.apply(self._init_weight)
    
    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01) 
            if module.bias is not None:
                module.bias.data.zero_()
        
    
    def post_forward(self, output):
        r""" Get the hidden_states from the PLM's layer output, pass it into the adapter, 
        then combined with the main hidden_states. Finally pass it into the subsequent layer.

        """
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError


        if not self.instantiated:
            self.hidden_dim = hiddens.shape[-1]
            logger.debug(f"Got hidden dim hidden_dim {self.hidden_dim}")
            self.instantiate(hidden_dim=self.hidden_dim)
                

        adapter_output = self.modulelist(hiddens)
        modified_output = adapter_output + hiddens # TODO option: disable residual_connection
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output
    
  

class AdapterConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`~AdapterModel`

    """
    def __init__(
        self, 
        bottleneck_dim: Optional[int]=24, 
        non_linearity: Optional[str]='gelu_new',
        sequential: Optional[str] = True,
        **kwargs
    ): 
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])



class AdapterModel(DeltaBase):
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
        common_structure (:obj:`bool`): whether using name-based addressing witha common structure mapping.

    """
    config_class = AdapterConfig
    delta_type = "adapter"
    default_modified_modules = ["attn", "ff"]
    def __init__(self,
                 backbone_model: nn.Module, 
                 bottleneck_dim: Optional[int]=24, 
                 non_linearity: Optional[str]='gelu_new',
                 sequential: Optional[str] = True,
                 modified_modules: Optional[bool] = None,
                 unfrozen_modules: Optional[bool] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 ):
        DeltaBase.__init__(self, 
                           backbone_model, 
                           modified_modules=modified_modules,
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
  
        
    def add_all_delta_to_backbone(self, 
                 module: nn.Module, 
                 modified_modules: List[str],
                ) -> nn.Module:
        for key, _ in module.named_modules():
            if self.find_key(key, modified_modules):
                self.update_module(module, key)
        self._pseudo_data_to_instantiate(module)
        self.mark_as_delta()
        return module
    
    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        adapterlayer = self.new_module_like(ref)
        self.insert_sequential_module(ref, delta_module=adapterlayer, delta_name="adapter")
    
    def new_module_like(self, module):
        module_device = get_device(module)
        adapterlayer = AdapterLayer(bottleneck_dim=self.bottleneck_dim, non_linearity=self.non_linearity, device=module_device)
        self.delta_modules.append(adapterlayer)  
        return adapterlayer
    