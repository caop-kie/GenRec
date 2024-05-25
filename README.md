# GenRec: Generative Personalized Sequential Recommendation

Source code for our paper `GenRec: Generative Personalized Sequential Recommendation` submitted to 2024 RecSys.

## Environment
- Python==3.8
- PyTorch==1.8.0
- transformers==3.1.0
- tensorboardx==2.4
- lxml==4.6.3
- beautifulsoup4==4.9.3
- bs4==0.0.1
- stanza==1.2
- sentencepiece==0.1.95
- ipdb==0.13.9

## Model

### Framework
![framework](./images/GenRec-Framework.png)

### Masking Strategy
![mask](./images/GenRec-Mask.png)

## Pretraining

```Bash
python genrec/train.py -c config/config_genrec_pretrain_amazon_sports.json
```

## Finetuning

```Bash
python genrec/train.py -c config/config_genrec_finetune_amazon_sports.json -pmp path_to_pretrained_model
```

## Evaluate

```Bash
python genrec/evaluate.py -c config/config_genrec_finetune_amazon_sports.json
```
