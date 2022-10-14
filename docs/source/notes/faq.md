# FAQs

1. **Why I encounder NotImplementedError in Prefix Tuning?**

    This is because we find no easy way to get a unified Prefix Tuning implementation for different attention classes. If you really want to use Prefix Tuning for the models we have not supported, you can implement the ``PrefixLayerYOURMODEL`` on your own or raise a issue to request the feature for your model. 

2. **Available Models with default configurations are ..., Please manually add the delta models by speicifying 'modified_modules' based on the visualization of your model structure**

    Although most pre-trained models (PTMs) use the transformers archtecture, they are implemented differently. For example, the attention module in GPT2 and BERT is not only named differently, but also implemented in different ways. Common structure mapping mapps the different name conventions of different PTMs into a unified name convention. But there are many PTMs that we do not currently cover. But don't worry! For these models, you can figure out which modules should you modify by simply [visualizing the PTMs](visualization), and then specify the `modified modules` manually (See [name-based addressing](namebasedaddr)). 
