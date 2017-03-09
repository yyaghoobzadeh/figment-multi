# FIGMENT-MULTI
## Multi-level Representations for Fine-Grained Typing of Knowledge Base Entities

This is an extension to the old [FIGMENT](https://github.com/yyaghoobzadeh/figment/).  

There are implementations for learning 
character, word, and entity level representations for entity typing.


### Installation

### Prepare datasets
To download embeddings and preparing datasets, first run this command: 

```
sh prepare.sh
```
### Training
To train, test and evaluate one of the models, you can use this script ("swlr" is the example model in this case):

```
sh run_one_model.sh swlr
```
There should be a subdirectory with the name "swlr" in "configs" directory. 
All the models in the paper already have one config directory. 
Please see (https://github.com/yyaghoobzadeh/figment-multi/tree/master/configs) for the list. 

