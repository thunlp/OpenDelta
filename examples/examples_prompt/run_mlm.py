# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import functools
import logging
# from opendelta.utils.delta_center import create_hub_repo_name
import torch
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import sys
import subprocess
from typing import Optional, List

from datasets import load_dataset, load_metric, concatenate_datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
# from ..seq2seq.utils import get_adapter_config
from examples_prompt.data_processors import AutoTask, TaskDataCollatorForSeq2Seq, AutoPostProcessor, data_collator
from examples_prompt.seq2seq_trainer import Seq2SeqTrainer
# from training_args import AdapterTrainingArguments
from examples_prompt.trainers.trainer_utils import save_training_config
from dataclasses import dataclass, field

from transformers.models.t5.modeling_t5 import T5Config, T5ForConditionalGeneration
from examples_prompt.trainers.model_args import ModelArguments
from examples_prompt.trainers.trainer_args import TrainingArguments, DataTrainingArguments
from transformers.trainer import Trainer
from examples_prompt.metrics.metrics import transform_for_generation
import json
import numpy as np
logger = logging.getLogger(__name__)



TASK_TO_METRICS = {"mrpc": ["accuracy", "f1"],
                  "cola": ['matthews_correlation'],
                  "stsb": ['pearson', 'spearmanr'],
                  'sst2': ['accuracy'],
                  "mnli": ["accuracy"],
                  "mnli_mismatched": ["accuracy"],
                  "mnli_matched": ["accuracy"],
                  "qnli": ["accuracy"],
                  "rte": ["accuracy"],
                  "wnli": ["accuracy"],
                  "qqp": ["accuracy", "f1"],
                  "superglue-boolq": ["accuracy"],
                  "superglue-rte": ["accuracy"],
                  "superglue-cb": ["f1_multiclass", "accuracy"],
                  "superglue-copa": ["accuracy"],
                  "superglue-multirc": ["f1", "em"],
                  "superglue-wic": ["accuracy"],
                  "superglue-wsc.fixed": ["accuracy"],
                  "superglue-record": ["f1", "em"]
         }


class RemainArgHfArgumentParser(HfArgumentParser):
    def parse_json_file(self, json_file: str, return_remaining_args=True ):
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        import argparse
        import json
        from pathlib import Path
        import dataclasses

        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: data.pop(k) for k in list(data.keys()) if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)

        remain_args = argparse.ArgumentParser()
        remain_args.__dict__.update(data)
        if return_remaining_args:
            return (*outputs, remain_args)
        else:
            return (*outputs,)



def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = RemainArgHfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, delta_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, delta_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)


    print(f"{training_args.output_dir}/results.json")
    # exit()
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("#### last_checkpoint ", last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            '''
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            '''
            pass
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    # logger.info("Training/evaluation parameters %s", training_args, model_args, data_args, delta_args)
    logger.info("{}\n{}\n{}\n{}".format(training_args, model_args, data_args, delta_args))


    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.dropout_rate = 0.0
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))

    from openprompt.prompts import ManualTemplate
    from openprompt.plms import MLMTokenizerWrapper
    template = ManualTemplate(tokenizer, text="""sentence1: {"meta": 'premise'}, sentence2:,"""+
        """{"meta":"hypothesis", "shortenable":True}, The answer was {"mask"} .""")
    tokenizer_wrapper = MLMTokenizerWrapper(max_seq_length=data_args.max_source_length, tokenizer=tokenizer, truncate_method='tail')




    if delta_args.delta_type.lower() != "none":
        from opendelta import AutoDeltaConfig,AutoDeltaModel
        delta_config = AutoDeltaConfig.from_dict(vars(delta_args))
        delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=model)
        delta_model.freeze_module(set_state_dict = True)
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)


    # model parallelize
    if hasattr(training_args, "model_parallel") and training_args.model_parallel:
        logger.info('parallelize model!')
        model.parallelize()

    data_args.dataset_name = [data_args.task_name]
    data_args.eval_dataset_name = [data_args.eval_dataset_name]
    data_args.test_dataset_name = [data_args.test_dataset_name]
    data_args.dataset_config_name = [data_args.dataset_config_name]
    data_args.eval_dataset_config_name = [data_args.eval_dataset_config_name]
    data_args.test_dataset_config_name = [data_args.test_dataset_config_name]
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(data_args.test_dataset_config_name)

    # Temporarily set max_target_length for training.
    #max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples, **kwargs):
        # max_target_length += 1
        tokenizer = kwargs['tokenizer']
        data_args = kwargs['data_args']




        print("max_length", data_args.max_source_length)
        model_inputs = tokenizer(examples['source'], max_length=data_args.max_source_length,
                                 padding="max_length", truncation=True)

        # mask_position = [(id, input_id.index(tokenizer.mask_token_id)) for id, input_id in enumerate(model_inputs.input_ids)]# [[-100 if i != tokenizer.mask_token_id else tokenizer.convert_tokens_to_ids(target) for i in input_id] for input_id, target in zip(model_inputs.input_ids, examples['target'])]
        # model_inputs["mask_position"] = mask_position
        model_inputs["extra_fields"] = examples['extra_fields']
        # from IPython import embed; embed(header="Therer")
        return model_inputs


    column_names = ['source', 'target', 'label', 'extra_fields']
    performance_metrics = {}

    if training_args.do_train:

        train_task = AutoTask.get(data_args.task_name,
                                       data_args.dataset_config_name,
                                       data_args=data_args,
                                       tokenizer=tokenizer,
                                       predict_with_generate=training_args.predict_with_generate,
                                       seed=data_args.data_seed)

        train_dataset = train_task.get(split='train',
                                   split_validation_test=training_args.split_validation_test,
                                   n_obs=data_args.max_train_samples)


        train_dataset = train_dataset.map(
                        functools.partial(preprocess_function,
                                            data_args=data_args,
                                            tokenizer=tokenizer),
                            batched=True,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=[x for x in train_dataset.features if x not in ("label",)], # if train_dataset != "superglue-record" else column_names+["answers"],
                            load_from_cache_file=not data_args.overwrite_cache,
                        )



    eval_splits_names = []

    if training_args.do_eval:
        eval_splits_names.append("eval")
    if training_args.do_test:
        eval_splits_names.append("test")
    eval_splits = {}
    for split_name in eval_splits_names:
        _tasks = {dataset_name: AutoTask.get(dataset_name,
                                       dataset_config_name,
                                       data_args=data_args,
                                       tokenizer=tokenizer,
                                       predict_with_generate=training_args.predict_with_generate,
                                       seed=data_args.data_seed)
            for dataset_name, dataset_config_name\
            in zip(getattr(data_args,f"{split_name}_dataset_name"), getattr(data_args, f"{split_name}_dataset_config_name"))}

        _datasets = {dataset_name: task.get(split=split_name,
                                   split_validation_test=training_args.split_validation_test,
                                   n_obs=data_args.max_train_samples)
                        for dataset_name, task in _tasks.items()
        }

        _datasets = {dataset_name: d.map(
                        functools.partial(preprocess_function,
                                            data_args=data_args,
                                            tokenizer=tokenizer),
                            batched=True,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=[x for x in d.features if x not in ("label",)], # if train_dataset != "superglue-record" else column_names+["answers"],
                            load_from_cache_file=not data_args.overwrite_cache,
                        )
                        for dataset_name, d in _datasets.items()
        }

        eval_splits[split_name] = _datasets
        if split_name == "test":
            eval_metrics = {dataset_name:task.metric for dataset_name, task in _tasks.items()}
            verbalizers = {dataset_name:task.verbalizer for dataset_name, task in _tasks.items()}


    # Metric, we assume we have only one training task.
    # eval_metrics = [task.metric for task in
    #     for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)][0]

    # Extracts the extra information needed to evaluate on each dataset.
    # These information are only used in the compute_metrics.
    # We will assume that the test/eval dataloader does not change the order of
    # the data.
    # data_info = {"eval": eval_datasets[data_args.eval_dataset_name[0]]['extra_fields'],
    #              "test": test_datasets[data_args.test_dataset_name[0]]['extra_fields'],
    #              "train": train_dataset['extra_fields']}
    def compute_metrics(eval_preds, dataset_name):

        preds, labels = eval_preds.predictions, eval_preds.label_ids

        preds = np.argmax(preds, axis=-1)

        result = {}
        average_metrics = []
        for metric in eval_metrics[dataset_name]:
            metric_item = metric(preds, labels)
            metric_value =  list(metric_item.values())
            result.update(metric_item)
            average_metrics.extend(metric_value)
        print("average:",average_metrics)
        average_metric = sum(average_metrics)/len(average_metrics)
        result.update({"average_metrics":average_metric})
        return result

    # from IPython import embed; embed(header="isseq2seq")
    # Initialize our Trainer
    # if training_args.is_seq2seq == True:
    #     trainer = Seq2SeqTrainer(
    #         model=model,
    #         args=training_args,
    #         delta_args=delta_args,
    #         train_dataset=splits['train'] if training_args.do_train else None,
    #         eval_dataset=list(splits['validation'].values())[0] if training_args.do_eval else None,
    #         data_info = data_info,
    #         tokenizer=tokenizer,
    #         data_collator=data_collator,
    #         compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    #         evaluation_metrics = TASK_TO_METRICS[data_args.dataset_name[0]],
    #     )
    # else:

    class MLMTrainer(Trainer):
        _verbalizers = verbalizers

        # def training_step(self, model, inputs):
        #     from IPython import embed; embed(header="in trainstep")
        #     return super().training_step(model, inputs)
        def compute_loss(self, model, inputs, return_outputs=False):

            labels = inputs.pop('labels')
            extra_fields = inputs.pop("extra_fields")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            input_ids = inputs['input_ids']



            # from IPython import embed; embed(header="382")
            verbalizer = self._verbalizers[extra_fields[0]['dataset_name']].cuda()
            logits_at_mask = logits[torch.where(input_ids == verbalizer.tokenizer.mask_token_id)]

            # colidx = torch.where(input_ids == verbalizer.tokenizer.mask_token_id)[0].cpu()
            # print(colidx)
            # missing = set([i for i in range(input_ids.size(0))]) - set(colidx.numpy())
            # print(missing)
            # if len(missing) > 0:
                # print("missing")
                # missing = list(missing)[0]
                # input_ids_missing = input_ids[missing]
                # print(input_ids_missing)
                # missing_tokens = verbalizer.tokenizer.convert_ids_to_tokens(input_ids_missing)
                # print(missing_tokens)
            label_logits = verbalizer.process_logits(logits_at_mask)
            loss_fct = torch.nn.CrossEntropyLoss()
            # from IPython import embed; embed(header="In compute loss")
            loss = loss_fct(label_logits, labels)
            outputs.logits = label_logits
            return (loss, outputs) if return_outputs else loss

        # def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        #     aa = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        #     # from IPython import embed; embed()
        #     return aa
    from transformers.data.data_collator import torch_default_data_collator , DataCollatorMixin
    class DataCollatorWithExtraFields(DataCollatorMixin):
        return_tensors: str = "pt"
        def torch_call(self, features):
            print(len(features))
            extra_fields = [f.pop('extra_fields') for f in features]
            batch = torch_default_data_collator(features)
            batch['extra_fields'] =extra_fields
            print(batch['input_ids'].size())
            print(batch['labels'].size())
            return batch


    # from transformers.data.data_collator import DefaultDataCollator
    # class CustomDataCollator(DefaultDataCollator):

    #     def __call__(self, features):
    #         mask_position = [d.pop('mask_position') for d in features]
    #     #    self.check_uniqueness(tasks)
    #         from IPython import embed; embed(header="featurres")
    #         output = super().__call__(features)
    #         # mask_positions = [d.pop('mask_position') for d in features]
    #         output["mask_position"] = mask_position
    #         return output


    training_args.remove_unused_columns = False

    trainer = MLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_splits['eval'][data_args.task_name] if training_args.do_eval else None,
        compute_metrics=functools.partial(compute_metrics, dataset_name=data_args.task_name),
        # tokenizer=tokenizer,
        data_collator=DataCollatorWithExtraFields(),
    )


    # Saves training config.
    if trainer.is_world_process_zero():
       os.makedirs(training_args.output_dir, exist_ok=True)
       save_training_config(sys.argv[1], training_args.output_dir)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_time in minutes ": total_time})

        trainer.save_model()  # Saves the tokenizer too for easy upload
        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        train_metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        print(performance_metrics)
        trainer.save_metrics("performance", performance_metrics)

    # Evaluation
    all_results = {}

    all_results['evaluate'] = {}

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        for dataset_name, eval_dataset in eval_splits['eval'].items():
            metrics = trainer.evaluate(eval_dataset=eval_dataset,
            )
            trainer.log_metrics(f"{dataset_name}_eval", metrics)
            trainer.save_metrics(f"{dataset_name}_eval", metrics)
            all_results['evaluate'][dataset_name] = metrics

    # Test
    all_results['test'] = {}
    if training_args.do_test:
        logger.info("*** Test ***")
        for dataset_name, test_dataset in eval_splits['test'].items():
            metrics = trainer.evaluate(eval_dataset=test_dataset,
            metric_key_prefix="test"
            )
            trainer.log_metrics(f"{dataset_name}_test", metrics)
            trainer.save_metrics(f"{dataset_name}_test", metrics)
        all_results['test'][dataset_name] = metrics

    # repo_name = create_hub_repo_name(root="DeltaHub",
    #                      dataset=data_args.task_name,
    #                      delta_type = delta_args.delta_type,
    #                      model_name_or_path= model_args.model_name_or_path)
    # results['repo_name'] = repo_name
    # if delta_args.delta_type.lower() != "none":
    #     if training_args.push_to_hub: # TODO add description here
    #         delta_model.save_finetuned(push_to_hub=True, save_directory=repo_name, use_auth_token=True)
    #         # trainer.push_to_hub(**kwargs)
    #     else:
    #         delta_model.save_finetuned(push_to_hub=False, save_directory=repo_name, use_auth_token=True)


    with open(f"{training_args.output_dir}/results.json", 'w') as fout:
        string = json.dumps(all_results, indent=4,sort_keys=True)
        fout.write(string+"\n")

    return all_results




if __name__ == "__main__":
    result = main()

