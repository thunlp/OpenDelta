# Adapted from Tevatron (https://github.com/texttron/tevatron)

import logging
import os
import sys

from openmatch.arguments import DataArguments
from openmatch.arguments import DRTrainingArguments as TrainingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import QPCollator, DRTrainDataset, DREvalDataset
from openmatch.modeling import DRModel
from openmatch.trainer import DRTrainer as Trainer
from openmatch.trainer import GCDenseTrainer
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from transformers.integrations import TensorBoardCallback

logger = logging.getLogger(__name__)


class UnitTest:
    def __init__(self, model):
        self.model = model
    
    
    def unitTest1(self, delta_config_dict):
        model = self.model
        from opendelta import Visualization
        Visualization(model).structure_graph()

        from opendelta import AutoDeltaConfig, AutoDeltaModel

        delta_config = AutoDeltaConfig.from_dict(delta_config_dict)
        delta_model = AutoDeltaModel.from_config(delta_config, backbone_model = model)

        from opendelta import Visualization
        Visualization(model).structure_graph()


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
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
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = DRModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )


    unit_test = UnitTest(model)

    # unit_test.unitTest1({
    #         "delta_type": "bitfit",
    #     })
    # unit_test.unitTest1({
    #         "delta_type": "lora",
    #     })
    unit_test.unitTest1({
            "delta_type": "adapter",
        })
    



    # train_dataset = DRTrainDataset(tokenizer, data_args, shuffle_seed=training_args.seed, cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    # eval_dataset = DREvalDataset(tokenizer, data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir) if data_args.eval_path is not None else None

    # tb_callback = TensorBoardCallback()

    # trainer_cls = GCDenseTrainer if training_args.grad_cache else Trainer
    # trainer = trainer_cls(
    #     model=model,
    #     args=training_args,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     data_collator=QPCollator(
    #         tokenizer,
    #         max_p_len=data_args.p_max_len,
    #         max_q_len=data_args.q_max_len
    #     ),
    #     callbacks=[tb_callback]
    # )
    # train_dataset.trainer = trainer

    # trainer.train()
    # trainer.save_model()
    # if trainer.is_world_process_zero():
    #     tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
