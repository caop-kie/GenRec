# GenRec: Generative Sequential Recommendation with Large Language Models

Source code for our paper `GenRec: Generative Sequential Recommendation with Large Language Models`.

## Environment
- Python==3.8
- PyTorch==1.12.1
- transformers==4.22.2
- tensorboardx==2.5.1
- lxml==4.9.1
- beautifulsoup4==4.11.1
- bs4==0.0.1
- stanza==1.4.2
- sentencepiece==0.1.97
- ipdb==0.13.9

## Model

### Framework
![framework](./images/GenRec-Framework.png)

### Masking Strategy
![mask](./images/GenRec-Mask.png)

## Datasets
Finetuning datasets are available in the datasets folder. We will publish the pretrain datasets soon.

## Pretraining

By default, we pretrain 10 epochs on the pretrain datasets and finetune after that. Increasing the pretraining epochs can potentially improve the performance further. 
However, our work focuses on real-life low resource scenarios and thus does not pretrain more epochs to exhaustly find the best model performance.  
```Bash
python genrec/train.py -c config/pretrain_amazon_sports.json
```

## Training/Finetuning

Finetuning with a pretrained model. The script runs evaluation on the test dataset whenever the model has a better test score on the validation dataset.
```Bash
python genrec/train.py -c config/finetune_amazon_sports.json -pmp output/20240828_204904/epoch_10.mdl
```

Finetuning without a pretrained model. The model is trained from scratch.
```Bash
python genrec/train.py -c config/finetune_amazon_sports.json
```

## Evaluate

```Bash
python genrec/evaluate.py -c config/evaluate_amazon_sports.json
```


## Future Work
While our model is proposed for sequential recommendation, it can be easily converted to handle direct recommendation and other recommendation tasks. We leave this as future work and welcome collaboration from external contributors.

Please raise your questions or comments in the issues.
