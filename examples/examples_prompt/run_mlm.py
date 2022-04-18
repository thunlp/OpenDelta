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
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
# from ..seq2seq.utils import get_adapter_config
from examples_prompt.data_processors import AutoTask #, #TaskDataCollatorForSeq2Seq, AutoPostProcessor, data_collator
from transformers import Seq2SeqTrainer
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

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    if training_args.predict_with_generate:
        model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model.resize_token_embeddings(len(tokenizer))





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






    column_names = ['source', 'target', 'label', 'extra_fields']
    performance_metrics = {}




    def get_prompts(task, tokenizer, predict_with_generate, template_id="0", verbalizer_id="0"):
        # tid = getattr(config, "template_id", "0")
        # vid = getattr(config, "verbalizer_id", "0")
        from openpromptu.prompts import GenerationVerbalizer, ManualVerbalizer
        from openpromptu.prompts import ManualTemplate
        template = ManualTemplate(text = task.templates_text[template_id])
        if predict_with_generate:
            verbalizer = GenerationVerbalizer(tokenizer=tokenizer, classes = task.labels_list, label_words=task.verbalizers[verbalizer_id])
        else:
            verbalizer = ManualVerbalizer(tokenizer=tokenizer, classes = task.labels_list, label_words=task.verbalizers[verbalizer_id])
            # max_target_length = self.get_max_target_length(self.default_max_length)

        from openpromptu import TokenizerWrapper
        tokenizer_wrapper = TokenizerWrapper(max_seq_length=data_args.max_source_length, tokenizer=tokenizer, truncate_method="balanced", mask_token_func=mask_token_func)
        return template, verbalizer, tokenizer_wrapper


    from openpromptu.data_utils import InputExample

    max_target_length = 32

    if os.path.basename(model_args.model_name_or_path).startswith("t5"):
        mask_token_func = lambda i: tokenizer.additional_special_tokens[i]
        def preprocess_function(raw_example, **kwargs):
            # max_target_length += 1
            tokenizer = kwargs['tokenizer']
            data_args = kwargs['data_args']
            template = kwargs['template']
            verbalizer = kwargs['verbalizer']
            tokenizer_wrapper = kwargs['tokenizer_wrapper']
            split = kwargs['split']
            # extra_fileds = example['extra_fields']

            example = InputExample(**raw_example)

            # from collections import namedtuple
            # example['tgt_text'] = ""
            # example = namedtuple("ObjectName", example.keys())(*example.values())
            try:
                example = verbalizer.wrap_one_example(example)
                example, other = template.wrap_one_example(example)
                input_sentence = tokenizer_wrapper.merge_wrapped_example(example)
                model_inputs = tokenizer(input_sentence, max_length=256,
                                    padding="max_length", truncation=True)
            except:
                from IPython import embed; embed(header="Therer")


            # if split == "train":
            with tokenizer.as_target_tokenizer():
                label = tokenizer(other['tgt_text']).input_ids
                # label = [l if l != tokenizer.pad_token_id else -100 for l in label]

            # from IPython import embed; embed(header="Therer")
            model_inputs["labels"] = label
            # else:
            #     # from IPython import embed; embed(header="Therer")
            #     model_inputs["tgt_text"] = other['tgt_text']
            #     model_inputs['labels'] = None            # model_inputs["extra_fields"] = extra_fileds
            # from IPython import embed; embed(header="Therer2")
            return model_inputs

        def compute_metrics(eval_preds, tokenizer, dataset_name, eval_metric):
            # from IPython import embed; embed(header="In compute metrics")
            preds, labels = eval_preds
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # post_processor = .get(data_args.dataset_name[0], tokenizer,
            #                                     data_args.ignore_pad_token_for_loss)
            # decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)
            result = {}
            for metric in eval_metric:
                result.update(metric(decoded_preds, decoded_labels))

            average_metric = sum(result.values())/len(result)
            result.update({"average_metrics":average_metric})
            return result



    elif os.path.basename(model_args.model_name_or_path).startswith("roberta") \
        or os.path.basename(model_args.model_name_or_path).startswith("bert"):
        mask_token_func = lambda i: tokenizer.mask_token
        def preprocess_function(raw_example, **kwargs):
            # max_target_length += 1

            # from IPython import embed; embed(header="Therer")
            tokenizer = kwargs['tokenizer']

            data_args = kwargs['data_args']
            template = kwargs['template']
            verbalizer = kwargs['verbalizer']
            tokenizer_wrapper = kwargs['tokenizer_wrapper']

            example = InputExample(**raw_example)

            # from collections import namedtuple
            # example['tgt_text'] = ""
            # example = namedtuple("ObjectName", example.keys())(*example.values())
            # try:
                # example = verbalizer.wrap_one_example(example)
            example, other = template.wrap_one_example(example)
            input_sentence = tokenizer_wrapper.merge_wrapped_example(example)
            model_inputs = tokenizer(input_sentence, max_length=256,
                                padding="max_length", truncation=True)



            # print("max_length", data_args.max_source_length)
            # model_inputs = tokenizer(examples['source'], max_length=data_args.max_source_length,
            #                         padding="max_length", truncation=True)

            # mask_position = [(id, input_id.index(tokenizer.mask_token_id)) for id, input_id in enumerate(model_inputs.input_ids)]# [[-100 if i != tokenizer.mask_token_id else tokenizer.convert_tokens_to_ids(target) for i in input_id] for input_id, target in zip(model_inputs.input_ids, examples['target'])]
            # model_inputs["mask_position"] = mask_position
            # model_inputs["extra_fields"] = examples['extra_fields']
            # from IPython import embed; embed(header="Therer")
            return model_inputs

        def compute_metrics(eval_preds, dataset_name):
            # from IPython import embed; embed(header="In compute metrics")

            preds, labels = eval_preds.predictions, eval_preds.label_ids

            preds = np.argmax(preds, axis=-1)

            result = {}
            average_metrics = []
            for metric in eval_metric:
                metric_item = metric(preds, labels)
                metric_value =  list(metric_item.values())
                result.update(metric_item)
                average_metrics.extend(metric_value)
            print("average:",average_metrics)
            average_metric = sum(average_metrics)/len(average_metrics)
            result.update({"average_metrics":average_metric})
            return result





    if training_args.do_train:

        train_task = AutoTask.get(data_args.task_name,
                                       data_args.dataset_config_name,
                                       data_args=data_args,
                                    #    tokenizer=tokenizer,
                                    #    predict_with_generate=training_args.predict_with_generate,
                                       seed=data_args.data_seed)

        train_dataset = train_task.get(split='train',
                                   split_validation_test=training_args.split_validation_test,
                                   n_obs=data_args.max_train_samples)

        template, verbalizer, tokenizer_wrapper = get_prompts(train_task, tokenizer, training_args.predict_with_generate)



        train_dataset = train_dataset.map(
                            functools.partial(preprocess_function,
                            data_args=data_args,
                            tokenizer=tokenizer,
                            template=template,
                            verbalizer=verbalizer,
                            tokenizer_wrapper=tokenizer_wrapper,
                            split="train"),
                            batched=False,
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
        eval_task = AutoTask.get(data_args.task_name,
                                data_args.dataset_config_name,
                                data_args=data_args,
                                # tokenizer=tokenizer,
                                # predict_with_generate=training_args.predict_with_generate,
                                seed=data_args.data_seed)
            # for dataset_name, dataset_config_name\
            # in zip(getattr(data_args,f"{split_name}_dataset_name"), getattr(data_args, f"{split_name}_dataset_config_name"))}

        eval_dataset = eval_task.get(split=split_name,
                                  split_validation_test=training_args.split_validation_test,
                                  n_obs=data_args.max_train_samples)



        template, _verbalizer, tokenizer_wrapper = get_prompts(eval_task, tokenizer, training_args.predict_with_generate)

        eval_dataset = eval_dataset.map(
                            functools.partial(preprocess_function,
                            data_args=data_args,
                            tokenizer=tokenizer,
                            template=template,
                            verbalizer=_verbalizer,
                            tokenizer_wrapper=tokenizer_wrapper,
                            split=split_name),
                            batched=False,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=[x for x in eval_dataset.features if x not in ("label",)], # if train_dataset != "superglue-record" else column_names+["answers"],
                            load_from_cache_file=not data_args.overwrite_cache,
                        )


        eval_splits[split_name] = eval_dataset
        if split_name == "test":
            eval_metric = eval_task.metric
            verbalizer = _verbalizer



    class MLMTrainer(Trainer):
        def __init__(self, verbalizer=None, **kwargs):
            super().__init__(**kwargs)
            self.verbalizer=verbalizer

        # def training_step(self, model, inputs):
        #     from IPython import embed; embed(header="in trainstep")
        #     return super().training_step(model, inputs)
        def compute_loss(self, model, inputs, return_outputs=False):

            labels = inputs.pop('labels')
            # extra_fields = inputs.pop("extra_fields")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            input_ids = inputs['input_ids']



            # from IPython import embed; embed(header="382")
            verbalizer = self.verbalizer.cuda()
            logits_at_mask = logits[torch.where(input_ids == verbalizer.tokenizer.mask_token_id)]
            label_logits = verbalizer.process_logits(logits_at_mask)
            loss_fct = torch.nn.CrossEntropyLoss()
            # from IPython import embed; embed(header="In compute loss")
            loss = loss_fct(label_logits, labels)
            outputs.logits = label_logits
            return (loss, outputs) if return_outputs else loss


    class MySeq2SeqTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # from IPython import embed; embed(header="agag")

            intlabel = inputs.pop('label')
            # extra_fields = inputs.pop("extra_fields")
            outputs = model(**inputs)
            # logits = outputs.get("logits")
            # input_ids = inputs['input_ids']



            # # from IPython import embed; embed(header="382")
            # verbalizer = self._verbalizers.cuda()
            # logits_at_mask = logits[torch.where(input_ids == verbalizer.tokenizer.mask_token_id)]
            # label_logits = verbalizer.process_logits(logits_at_mask)
            # loss_fct = torch.nn.CrossEntropyLoss()
            # # from IPython import embed; embed(header="In compute loss")
            # loss = loss_fct(label_logits, labels)
            # outputs.logits = label_logits
            if return_outputs:
                return (outputs.loss, outputs)
            else:
                return outputs.loss


        # def evaluate(
        #     self,
        #     eval_dataset: Optional[Dict[str, Dataset]] = None,
        #     ignore_keys: Optional[List[str]] = None,
        #     metric_key_prefix: str = "eval",
        #     max_length: Optional[int] = None,
        #     num_beams: Optional[int] = None,
        # ) -> Dict[str, float]:
        #     # TODO: this also needs to be set per dataset
        #     self._max_length = max_length
        #     self._num_beams = num_beams
        #     return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)


        def prediction_step(
            self,
            model, #nn.Module,
            inputs, #Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only, #: bool,
            ignore_keys, #: Optional[List[str]] = None,
        ): #-> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on :obj:`model` using obj:`inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (:obj:`nn.Module`):
                    The model to evaluate.
                inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument :obj:`labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (:obj:`bool`):
                    Whether or not to return the loss only.

            Return:
                Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
                labels (each being optional).
            """
            if not self.args.predict_with_generate or prediction_loss_only:
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
                )


            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)
            intlabel = inputs.pop('label')
            gen_kwargs = {
                "max_length": 10, # self._max_length if s is not None else self.model.config.max_length,
                "num_beams": 1 #self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            }
            generated_tokens = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

            with torch.no_grad():

                outputs = model(**inputs)
                if has_labels:
                    if self.label_smoother is not None:
                        loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    else:
                        loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
                else:
                    loss = None

            if self.args.prediction_loss_only:
                return (loss, None, None)

            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

            # from IPython import embed; embed(header="In seqseqtrainer")
            return (loss, generated_tokens, labels)





        # def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        #     aa = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        #     # from IPython import embed; embed()
        #     return aa
    # from transformers.data.data_collator import torch_default_data_collator , DataCollatorMixin
    # class DataCollatorWithExtraFields(DataCollatorMixin):
    #     return_tensors: str = "pt"
    #     def torch_call(self, features):
    #         # print(len(features))
    #         # extra_fields = [f.pop('extra_fields') for f in features]
    #         batch = torch_default_data_collator(features)
    #         batch['extra_fields'] =extra_fields
    #         # print(batch['input_ids'].size())
    #         # print(batch['labels'].size())
    #         return batch


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

    if os.path.basename(model_args.model_name_or_path).startswith("roberta") or \
        os.path.basename(model_args.model_name_or_path).startswith("bert"):
        trainer = MLMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_splits['eval'] if training_args.do_eval else None,
            compute_metrics=functools.partial(compute_metrics, dataset_name=data_args.task_name),
            tokenizer=tokenizer,
            # data_collator=DataCollatorWithExtraFields(),
            verbalizer=verbalizer,
        )
    elif os.path.basename(model_args.model_name_or_path).startswith("t5"):
        trainer = MySeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_splits['eval'] if training_args.do_eval else None,
            compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer, dataset_name=data_args.task_name, eval_metric=eval_metric),
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
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

        metrics = trainer.evaluate(eval_dataset=eval_splits['eval'],
        )
        trainer.log_metrics(f"{data_args.task_name}_eval", metrics)
        trainer.save_metrics(f"{data_args.task_name}_eval", metrics)
        all_results['evaluate'][data_args.task_name] = metrics

    # Test
    all_results['test'] = {}
    if training_args.do_test:
        logger.info("*** Test ***")
        metrics = trainer.evaluate(eval_dataset=eval_splits['test'],
        metric_key_prefix="test"
        )
        trainer.log_metrics(f"{data_args.task_name}_test", metrics)
        trainer.save_metrics(f"{data_args.task_name}_test", metrics)
        all_results['test'][data_args.task_name] = metrics

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

