# Adapted from Tevatron (https://github.com/texttron/tevatron)

from argparse import ArgumentParser
import logging
import os
import sys
import torch.nn as nn

logger = logging.getLogger(__name__)


class UnitTest:
    def __init__(self, models):
        self.models = models
    
        self.Configs = {}
        self.Configs[0] =  {
            "delta_type": "lora",
        }
    
        self.Configs[1] =  {
            "delta_type": "bitfit",
        }
    
        self.Configs[2] =  {
            "delta_type": "adapter",
        }
    
        self.Configs[3] =  {
            "delta_type": "compacter",
        }
    
        self.Configs[4] =  {
            "delta_type": "prefix",
        }

        self.Configs[5] =  {
            "delta_type": "soft_prompt",
        }
    
        self.Configs[6] =  {
            "delta_type": "low_rank_adapter",
        }

    def get_delta_config(self, config_id):
        return self.Configs[config_id]


    def unitTest0(self, delta_config_dict):
        model = self.models[0]
        from opendelta import Visualization
        Visualization(model).structure_graph()

        from opendelta import AutoDeltaConfig, AutoDeltaModel

        delta_config = AutoDeltaConfig.from_dict(delta_config_dict)
        delta_model = AutoDeltaModel.from_config(delta_config, backbone_model = model)

        from opendelta import Visualization
        Visualization(model).structure_graph()
    
    def unitTest1(self, delta_config_dict):
        class Mymodel(nn.Module):
            def __init__(self, a,b):
                super().__init__()
                self.a = a
                self.b = b
        
        model  = Mymodel(self.models[0], self.models[1])
        from opendelta import Visualization
        Visualization(model).structure_graph()

        from opendelta import AutoDeltaConfig, AutoDeltaModel

        delta_config = AutoDeltaConfig.from_dict(delta_config_dict)
        delta_model = AutoDeltaModel.from_config(delta_config, backbone_model = model)

        from opendelta import Visualization
        Visualization(model).structure_graph()
        delta_model.save_finetuned("./tmp")

        delta_model.freeze_module(exclude=['deltas'])
        delta_model.save_finetuned("./tmp")

        model = Mymodel(self.models[0], self.models[1])
        Visualization(model).structure_graph()
        delta_model = AutoDeltaModel.from_finetuned("./tmp", backbone_model=model)
        Visualization(model).structure_graph()


        



    
    def unit_test(self, test_id, config_id):
        delta_config_dict = self.Configs[config_id]
        if test_id == 0:
            self.unitTest0(delta_config_dict)
        elif test_id == 1:
            self.unitTest1(delta_config_dict)


from dataclasses import dataclass, field

@dataclass
class UnitTestArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    config_id: int = field(
        default=0,
    )
    test_id: int = field(
        default=0,
    )
    model_name_or_path: str =field(
        default='bert-base-cased', 
        metadata={"help": "tested: bert-base-cased, roberta-base, rinna/japanese-gpt2-small, t5-small, facebook/opt-125m"}
    )


from transformers import HfArgumentParser,TrainingArguments, AutoModel, GPT2Model

def main():
    parser = HfArgumentParser((TrainingArguments, UnitTestArguments))


    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        training_args, unit_test_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        training_args, unit_test_args = parser.parse_args_into_dataclasses()
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)


    model = AutoModel.from_pretrained(unit_test_args.model_name_or_path)

    import torch
    import copy
    models = [model, copy.deepcopy(model)]


    unit_test = UnitTest(models)


    unit_test.unit_test(unit_test_args.test_id, unit_test_args.config_id)

    
    





if __name__ == "__main__":
    main()
