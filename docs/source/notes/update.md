# Update Logs and Known Issues


## Version 0.3.0
### Updates:
- Add this changelog for a granular record of updates.
- The default configuration of delta models can be applied to more wrapped models.
  - There is less need to configure 'modified_modules' for wrapped models like [BertForSequenceClassification](https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertForSequenceClassification) or even [OpenMatch.DRModel](https://github.com/OpenMatch/OpenMatch/blob/master/src/openmatch/modeling/dense_retrieval_model.py#L37), as long as it has a model we support default configuration inside. **Note that if you customize `modified_modules` by yourself, most pytorch models are supported.**
- LoRA and BitFit models now does not need pseudo data to instantiate the model.
- BitFit models can now support [Conv1D](https://huggingface.co/docs/transformers/v4.23.1/en/internal/modeling_utils#transformers.Conv1D) using default configuration.
- Improve type hint for AutoDeltaModel.
- Fix bugs in documentation.
- Fix small bugs when saving a model without a config attributes.
- Make the default modified modules of adapter-like methods more accurate: attach the adapter-like modules after the output of attention layer and second feed-forward layer, both before the layernorm layers. 
- A simple unit test folder containing development-time tests has been added for interested users.


### Known Issues
- SoftPrompt is still not supported for wrapped model if the model has no attribute `get_input_embeddings`.
- Prefix Tuning is still limited to T5, GPT2, Bart, Bert, Roberta.

