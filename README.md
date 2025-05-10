## TSSGCF:Textual Similarity-Supervised Graph Collaborative Filtering for Web API Recommendation

### Introduction
> To tackle the challenges of noise propagation, over-smoothing, and data sparsity, A novel Textual Similarity-Supervised Graph Collaborative Filtering (TSSGCF) model was proposed for Web API Recommendation. This model constructs denoised graphs by fusing textual semantics with structural topology, introduces a textual similarity-supervised loss to enhance node embedding discriminability, and implements multimodal graph fusion via weighted averaging.

### Environment Requirment
> This code has been tested running undeer Python 3.9.0
> The Required packages are as follows:
> - torch == 2.1.0+cu121
> - numpy == 1.26.4
> - sentence-transformers == 4.1.0
> - pandas == 2.2.3
> - tqdm == 4.67.1
> - scipy == 1.13.1
> - scikit-learn == 1.6.1 

#### NOTE:The S-BERT model uses the version of `all-MiniLM-L6-v2`, which you can download from`https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2`


### Example to ruan TSSGCF
 - Command `python main.py`
 - Train log:
```
   shape of mashup_desc  (2289,)
   shape of api_desc  (956,)
   torch.Size([2289, 384])
   torch.Size([956, 384])
   ...
   56%|█████▌    | 5/9 [00:05<00:04,  1.03s/it]
```
### File Introduction
1. processing.py
> This file contains the code of data processing.
2. tssgcf.py
> This file contains the code of TSSGCF and LightGCN.
3. utils.py
> This file contains the founction used in the item.
4. main.py
> This file contains the code of model training and evaluation.
5. topkl.py
> This file contains the dataset loading code.
6. data
> data is available in the file of Data

