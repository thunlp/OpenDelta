from typing import Optional, Union
from opendelta.utils.signature import get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.basemodel import DeltaBase, is_leaf_module
from transformers.models.t5 import T5ForConditionalGeneration
import loralib as lora
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertForMaskedLM
import torch
from torch.nn import init
import math
from opendelta.utils.structure_mapping import transform
from opendelta import BaseDeltaConfig
import opendelta.utils.logging as logging
logger = logging.get_logger(__name__)


class BitFitConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`~BitFitModel`

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

class BiasLayer(nn.Module):
    def __init__(self, init_method="zero"):
        super().__init__()
        self.init_method=init_method
        self.instantiated = False

    def instantiate(self, hidden_dim):
        if self.init_method == "zero":
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            raise NotImplementedError
        self.instantiated = True
    
    def post_forward(self, output):
        r"""Presuming the first argument is the tensor to add bias along the last dimension.
        In most cases, it is correct. However, be aware of the possibility that the presumption
        doesn't hold. 
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

        modified_output = hiddens + self.bias
        
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output



class BitFitModel(DeltaBase):
    r""" The implementation of `BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models <https://arxiv.org/abs/2106.10199>`_ .
    Unfreeze bias term (or add bias term if bias term is absent in the backbone, e.g. T5) to the modules of
    a transformer block. 

    .. note:: 

        **Broadcast to Submodule**: We modify all potential positions  of the specified 
        ``modified_modules``. That is to say, if we specify ``attn`` in the modified_modules, then all position
        including the q, k, v and out linear layer of the attention layer are added bias layer (or unfreezing).
        The potential position is determined according to equation (1)-(5) and the previous three 
        equations.
    

    class attributes:
        - default_modified_modules = ["attn", "ff", "layer_norm","lm_head.proj"] According to the paper and the 
          implementation in `Compacter's baseline <https://github.com/rabeehk/compacter>`_ , we modify the
          bias term in the above modules. 
        - delta_type = "bitfit"




    Args:
        backbone_model (:obj:`transformers.PretrainedModels`): The backbone model to be modified. 
        modified_modules (:obj:`List[str]`): For prefix tuning, the it must refer to an attention layer (Currently, only
                        the implemented ones)
        unfrozen_modules (:obj:`List[str]`, *optional*, default to :obj:`None`): The modules that should be unfrozen
                         together with the prefix parameters.
        common_structure (:obj:`bool`): whether using name-based addressing witha common structure mapping.

    """


    config_class = BitFitConfig
    delta_type = "bitfit"
    default_modified_modules = ["attn", "ff", "layer_norm","lm_head.proj"] # modify all the bias parameter in attention and feed-forward layer.
    def __init__(self,
                 backbone_model: nn.Module, 
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

        self.delta_params = nn.ParameterList()
        self.delta_modules = nn.ModuleList()

        self.add_all_delta_to_backbone(self.backbone_model,
                                       self.modified_modules,
                                   )
    
    
    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        self.modify_module(ref)
        

    def modify_module(self,
                      module: nn.Module, 
                      ):
        if is_leaf_module(module):
            # if it is a leaf module, add bias to it regardless of its type.
            if isinstance(module, nn.Linear):
                self.add_bias_to_linear(module)
            else:
                # for example, layer_norms, lm_heads.
                self.add_bias_to_others(module)
        else:
            # for the non-leaf modules, by default it will add bias only to the linear submodules.
            for n, c in module.named_modules():
                if isinstance(c, nn.Linear):
                    if c.bias is None:
                        bias = nn.Parameter(torch.empty(c.out_features), requires_grad=True)
                        c.register_parameter('bias', bias)
                        self._reset_bias_parameters(c)
                        self.delta_params.append(bias)
                    else:
                        c.bias.requires_grad = True
                        self.delta_params.append(c.bias)
                else:
                    pass
   
    def add_bias_to_linear(self, c):
        if c.bias is None:
            bias = nn.Parameter(torch.empty(c.out_features), requires_grad=True)
            c.register_parameter('bias', bias)
            self._reset_bias_parameters(c)
            self.delta_params.append(bias)
        else:
            c.bias.requires_grad = True
            self.delta_params.append(c.bias)
    
    def add_bias_to_others(self, c):
        new_bias = BiasLayer()
        self.insert_sequential_module(c, delta_module=new_bias, delta_name="bitfit") # name shouldn't be `bias` here, since
                                        # the name `bias` is reserved for some module such as  roberta's LayerNorm.
        self.delta_modules.append(new_bias)


    
    @staticmethod
    def _reset_bias_parameters(linear_module):
        fan_in, _ = init._calculate_fan_in_and_fan_out(linear_module.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(linear_module.bias, -bound, bound)

    def detach(self, module):
        r"""Not implemented for BitFit yet. Please wait for the next version.
        """
        raise NotImplementedError
    
    def attach(self, module):
        r"""Not implemented for BitFit yet. Please wait for the next version.
        """
        raise NotImplementedError
