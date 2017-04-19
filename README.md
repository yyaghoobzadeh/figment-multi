# FIGMENT-MULTI
## Multi-level Representations for Fine-Grained Typing of Knowledge Base Entities

This is an extension to the old [FIGMENT](https://github.com/yyaghoobzadeh/figment/).  

There are implementations for learning 
character, word, and entity level representations for entity typing.


### Installation
```
Python: 2.7.11 
Numpy: 1.10.4 
Theano: 0.8.2 
Blocks: 0.1.1 
Fuel: 0.1.1 
```


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


### References
More information about the models is in this paper:

<a href="https://github.com/yyaghoobzadeh/figment-multi/blob/master/eacl-multi-level.pdf">
Multi-level Representations for Fine-Grained Typing of Knowledge Base Entities</a>,
Yadollah Yaghoobzadeh, Hinrich schütze. (EACL2017).
