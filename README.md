# [PoemBERT: A Dynamic Masking Content and Ratio Based Semantic Language Model For Chinese Poem Generation](https://aclanthology.org/2025.coling-main.5/) (COLING 2025)
By Chihan Huang, Xiaobo Shen

Ancient Chinese poetry stands as a crucial treasure in Chinese culture. To address the absence of pre-trained models for ancient poetry, we introduced PoemBERT, a BERT-based model utilizing a corpus of classical Chinese poetry. Recognizing the unique emotional depth and linguistic precision of poetry, we incorporated sentiment and pinyin embeddings into the model, enhancing its sensitivity to emotional information and addressing challenges posed by the phenomenon of multiple pronunciations for the same Chinese character. Additionally, we proposed Character Importance-based masking and dynamic masking strategies, significantly augmenting the model’s capability to extract imagery-related features and handle poetry-specific information. Fine-tuning our PoemBERT model on various downstream tasks, including poem generation and sentiment classification, resulted in state-of-the-art performance in both automatic and manual evaluations. We provided explanations for the selection of the dynamic masking rate strategy and proposed a solution to the issue of a small dataset size.


## Code Organization
```
project-root/
├── sentiment_model/              # sentiment model code folder
│   ├── aug_poem_data.py          # perform data augmentation to a small dataset
│   └── train_sentiment_bert.py   # code for training the sentiment model
├── calculate_pmi.py              # calculate pmi values between characters
├── dataset.py                    # dataset
├── fusion_embedding.py           # model utilizing the model components
├── README.md                     # readme
├── train.py                      # training with dynamic masking rate decay
└── trainer.py                    # trainer
```

## Use

### Augment sentiment data

```Python
python sentiment_model/aug_poem_data.py
```

### Train sentiment model

```Python
python sentiment_model/train_sentiment_bert.py
```

### Calculate PMI and save

```Python
python calculate_pmi.py
```

### Train PoemBERT

```Python
python train.py --masking_type 'CI-mask'  --mask_decay 'elliptical'
```

## Citation
```
@inproceedings{huang-shen-2025-poembert,
    title = {{P}oem{BERT}: A Dynamic Masking Content and Ratio Based Semantic Language Model For {C}hinese Poem Generation},
    author = {Huang, Chihan  and
      Shen, Xiaobo},
    booktitle = {Proceedings of the 31st International Conference on Computational Linguistics},
    month = jan,
    year = {2025},
    address = {Abu Dhabi, UAE},
    publisher = {Association for Computational Linguistics},
    url = {https://aclanthology.org/2025.coling-main.5/},
    pages = {50-60}
}
```
