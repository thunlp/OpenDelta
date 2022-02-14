from typing import OrderedDict
import copy
import opendelta.utils.logging as logging
from opendelta.utils.visualization import Visualization
logger = logging.get_logger(__name__)
t5_mapping = {
    "shared": {"__name__":"embeddings"},
    "encoder": {"__name__":"encoder",
        "embed_tokens": {"__name__":"embeddings"},
        "block": {"__name__":"block",
            "$": {"__name__":"$",
                "layer.0": {"__name__":"attn",
                    "SelfAttention.q": {"__name__":"q"},
                    "SelfAttention.k": {"__name__":"k"},
                    "SelfAttention.v": {"__name__":"v"},
                    "SelfAttention.o": {"__name__":"proj"},
                    "SelfAttention.relative_attention_bias": {"__name__":""},
                    "layer_norm": {"__name__":"layer_norm"},
                },
                "layer.1": {"__name__":"ff",
                    "DenseReluDense.wi": {"__name__":"w1"},
                    "layer_norm": {"__name__":"layer_norm"},
                    "DenseReluDense.wo": {"__name__":"w2"},
                }
            }
        },
        "final_layer_norm": {"__name__":"layer_norm"},  
    },
    "decoder": {"__name__":"decoder",
        "embed_tokens": {"__name__":"embeddings"},
        "block": {"__name__":"block",
            "$": {"__name__":"$",
                "layer.0": {"__name__":"attn",
                    "SelfAttention.q": {"__name__":"q"},
                    "SelfAttention.k": {"__name__":"k"},
                    "SelfAttention.v": {"__name__":"v"},
                    "SelfAttention.o": {"__name__":"proj"},
                    "SelfAttention.relative_attention_bias": {"__name__":""},
                    "layer_norm": {"__name__":"layer_norm"},
                },
                "layer.1": {"__name__":"crossattn",
                    "EncDecAttention.q": {"__name__":"q"},
                    "EncDecAttention.k": {"__name__":"k"},
                    "EncDecAttention.v": {"__name__":"v"},
                    "EncDecAttention.o": {"__name__":"proj"},
                    "layer_norm": {"__name__":"layer_norm"},
                },
                "layer.2": {"__name__":"ff",
                    "DenseReluDense.wi": {"__name__":"w1"},
                    "layer_norm": {"__name__":"layer_norm"},
                    "DenseReluDense.wo": {"__name__":"w2"},
                }
            }
        },
        "final_layer_norm": {"__name__":"layer_norm"},
    }
}


roberta_mapping = {
    "roberta.embeddings.word_embeddings": {"__name__":"embeddings"},
    "roberta.embeddings.position_embeddings": {"__name__":""},
    "roberta.embeddings.token_type_embeddings": {"__name__":""},
    "roberta.embeddings.LayerNorm": {"__name__":""},
    "roberta.encoder": {"__name__":"encoder",
        "layer": {"__name__":"block",
            "$": {"__name__":"$",
                "attention": {"__name__":"attn",
                    "self.query": {"__name__":"q"},
                    "self.key": {"__name__":"k"},
                    "self.value": {"__name__":"v"},
                    "output.dense": {"__name__":"proj"},
                    "output.LayerNorm": {"__name__":"layer_norm"},
                },
                "output": {"__name__":"ff",
                            "dense": {"__name__":"w2"},
                            "LayerNorm": {"__name__":"layer_norm"}
                },
                "intermediate.dense": {"__name__":"ff.w1"},
            }
        }
    },
    "lm_head": {"__name__":"lm_head",
        "dense": {"__name__":""},
        "layer_norm": {"__name__":""},
        "decoder": {"__name__":"proj"},
    },
}



bert_mapping = {
    "bert.embeddings.word_embeddings": {"__name__":"embeddings"},
    "bert.embeddings.position_embeddings": {"__name__":""},
    "bert.embeddings.token_type_embeddings": {"__name__":""},
    "bert.embeddings.LayerNorm": {"__name__":""},
    "bert.encoder": {"__name__":"encoder",
        "layer": {"__name__":"block",
            "$": {"__name__":"$",
                "attention": {"__name__":"attn",
                    "self.query": {"__name__":"q"},
                    "self.key": {"__name__":"k"},
                    "self.value": {"__name__":"v"},
                    "output.dense": {"__name__":"proj"},
                    "output.LayerNorm": {"__name__":"layer_norm"},
                },
                "output": {"__name__":"ff",
                            "dense": {"__name__":"w2"},
                            "LayerNorm": {"__name__":"layer_norm"}
                },
                "intermediate.dense": {"__name__":"ff.w1"},
            }
        }
    },
    "cls.predictions": {"__name__": "lm_head",
        "transform.dense": {"__name__":""},
        "transform.LayerNorm": {"__name__":""},
        "decoder": {"__name__":"proj"},
    }
}

debertav2_mapping = {
    "deberta.embeddings.word_embeddings": {"__name__":"embeddings"},
    "deberta.embeddings.LayerNorm": {"__name__":""},
    "deberta.encoder": {"__name__":"encoder",
        "layer": {"__name__":"block",
            "$": {"__name__":"$",
                "attention": {"__name__":"attn",
                    "self.query_proj": {"__name__":"q"},
                    "self.key_proj": {"__name__":"k"},
                    "self.value_proj": {"__name__":"v"},
                    "output.dense": {"__name__":"proj"},
                    "output.LayerNorm": {"__name__":"layer_norm"},
                },
                "output": {"__name__":"ff",
                            "dense": {"__name__":"w2"},
                            "LayerNorm": {"__name__":"layer_norm"}
                },
                "intermediate.dense": {"__name__":"ff.w1"},
            }
        },
        "rel_embeddings": {"__name__": ""},
        "LayerNorm": {"__name__": ""},
        "conv": {"__name__": "",
            "conv": {"__name__": ""},
            "LayerNorm": {"__name__": ""}
        }
    },
    "lm_predictions.lm_head": {"__name__":"lm_head",
        "dense": {"__name__":""},
        "LayerNorm": {"__name__":""},
        "bias": {"__name__": ""}
    },
}

gpt2_mapping = {
    "transformer.wte": {"__name__":"embeddings"},
    "transformer.wpe": {"__name__":""},
    "transformer.h": {"__name__":"decoder.block",
        "$": {"__name__":"$",
            "attn": {"__name__":"attn",
                "c_attn": {"__name__":"q,k,v"},
                "c_proj": {"__name__":"proj"},
            },
            "ln_1": {"__name__":"attn.layer_norm"},
            "mlp":{ "__name__": "ff",
               "c_fc": {"__name__":"w1"},
               "c_proj": {"__name__":"w2"}
            },
            "ln_2": {"__name__":"ff.layer_norm"},
        },
    },
    "transformer.ln_f": {"__name__":"decoder.layernorm"},
    "lm_head": {"__name__":"lm_head.proj"},
}

distilbert_mapping = {
    "distilbert.embeddings.word_embeddings": {"__name__":"embeddings"},
    "distilbert.embeddings.position_embeddings": {"__name__":""},
    "distilbert.embeddings.token_type_embeddings": {"__name__":""},
    "distilbert.embeddings.LayerNorm": {"__name__":""},
    "distilbert.transformer": {"__name__":"encoder",
        "layer": {"__name__":"block",
            "$": {"__name__":"$",
                "attention": {"__name__":"attn",
                    "q_lin": {"__name__":"q"},
                    "k_lin": {"__name__":"k"},
                    "v_lin": {"__name__":"v"},
                    "out_lin": {"__name__":"proj"},
                },
                "ffn": {"__name__":"ff",
                      "lin1": {"__name__":"w1"},
                      "lin2": {"__name__":"w2"},
                },
                "sa_layer_norm": {"__name__":"attn.layer_norm"},
                "output_layer_norm":{"__name__": "ff.layer_norm"}
            }
        }
    }
}

def transform(org_key, mapping, strict=True, warning=False, verbose=False):
    
    chain = org_key.split(".")
    query = ""
    node = mapping

    new_chain = []
    for elem in chain:
        query += elem
        if query in node:
            node = node[query]
            new_elem = node["__name__"]
            if new_elem == "":
                if strict:
                    if warning:
                        print(f"'{org_key}' has no common mapping.")
                    return 
                else:
                    new_chain.append(query)
            else:
                new_chain.append(new_elem)
            query = ""
        elif "$" in node:
            node = node["$"]
            new_chain.append(query)
            query = ""
        else:
            query += "." 
    if query!="":
        if strict:
            if warning:
                print("A part of the orginial key hasn't been matched!")
            return 
        else:
            new_chain.append(query.strip(".")) # tailing query
    new_key = ".".join(new_chain)
    if verbose:
        print(f"{org_key} => {new_key}")
    return new_key
    



def mapping_for_SequenceClassification(mapping, type):
    mapping = copy.deepcopy(mapping)
    if type == "roberta":
        mapping.pop("lm_head")
        mapping['classifier'] = {"__name__":"classifier",
            "dense": {"__name__": "dense"},
            "out_proj": {"__name__":"out_proj"}
        }
    elif type == "bert":
        mapping.pop("lm_head")
        mapping["classifier"] = {"__name__": "classifier"}
    elif type == "deberta":
        mapping.pop("lm_predictions.lm_head")
        mapping["pooler"] = {"__name__": "classifier"} 
        mapping["classifier"] = {"__name__": "classifier"}
    else:
        raise NotImplementedError
    return mapping

def mapping_for_ConditionalGeneration(mapping, type):
    mapping = copy.deepcopy(mapping)
    if type == "t5":
        mapping["lm_head"] = {"__name__":"lm_head.proj"}
    else:
        raise NotImplementedError
    return mapping

class _LazyLoading(OrderedDict):
    def __init__(self, mapping):
        self._mapping_string = mapping
        self._mapping = {}
    
    def __getitem__(self, key):
        if key not in self._mapping_string:
            raise KeyError(key)
        value = self._mapping_string[key]
        self._mapping[key] = eval(value)
        return self._mapping[key] 
    
    def keys(self):
        return list(self._mapping_string.keys())
    
    def __contains__(self, item):

        return item in self._mapping_string


class CommonStructureMap(object):
    r""" A lazy loading structure map.
    """
    Mappings = _LazyLoading({
        "RobertaForSequenceClassification": """mapping_for_SequenceClassification(roberta_mapping, "roberta")""",
        "RobertaForMaskedLM": "roberta_mapping",
        "BertForMaskedLM": "bert_mapping",
        "BertForSequenceClassification": """mapping_for_SequenceClassification(bert_mapping, "bert")""",
        "T5ForConditionalGeneration": """mapping_for_ConditionalGeneration(t5_mapping, "t5")""",
        "DebertaV2ForSequenceClassification": """mapping_for_SequenceClassification(debertav2_mapping, "deberta")"""
    })

    SpecialModelInverseMaps = {
    }
    def __init__(self, mapping):
        if not isinstance(mapping, dict):
            raise TypeError(f"Initial a {CommonStructureMap.__name__} using a non-dict object. Consider using `load` instead.")
        self.mapping = mapping


    @classmethod
    def load(cls, backbone_model, strict=True, warining=False, visualize=True):
        r"""Doc
        """
        backbone_class = type(backbone_model).__name__
        if backbone_class not in cls.Mappings:
            raise KeyError(backbone_class)
        mapping = cls.Mappings[backbone_class]
        if visualize:
            logger.info("Since you are using the common structure mapping, draw the transformed parameter structure for checking.")
            vis = Visualization(backbone_model)
            vis.structure_graph(common_structure=True, mapping=mapping)
        return cls(mapping)

    def __repr__(self,):
        return self.mapping


    def transform(self, org_key, strict=True, warning=False):
        return transform(org_key, self.mapping, strict, warning)



if __name__ == "__main__":
    from openprompt.plms import load_plm
    import argparse
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", type=str, default='t5-lm', help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
    parser.add_argument("--model_name_or_path", default="t5-base-lm-adapt")
    parser.add_argument("--cache_base", default='/home/hushengding/plm_cache/')
    parser.add_argument("--keep_non_params", action="store_true")
    parser.add_argument("--expand_params", action="store_true")
    args = parser.parse_args()
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.cache_base+args.model_name_or_path)

    for name, _ in plm.named_modules():
        transform(name, t5_mapping, strict=True, warning=False)
    