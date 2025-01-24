import torch
from torch import nn
import json
import os
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel


class PinyinEmbedding(nn.Module):
    def __init__(self, embedding_size: int, pinyin_out_dim: int, config_path):
        super(PinyinEmbedding, self).__init__()
        with open(os.path.join('./pretrained/bert-base-chinese/config', 'pinyin_map.json')) as fin:
            pinyin_dict = json.load(fin)
        self.pinyin_out_dim = pinyin_out_dim
        self.embedding = nn.Embedding(len(pinyin_dict['idx2char']), embedding_size)
        self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.pinyin_out_dim, kernel_size=2,
                              stride=1, padding=0)

    def forward(self, pinyin_ids):

        embed = self.embedding(pinyin_ids)
        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        view_embed = embed.view(-1, pinyin_locs, embed_size)
        input_embed = view_embed.permute(0, 2, 1)
        pinyin_conv = self.conv(input_embed)
        pinyin_embed = F.max_pool1d(pinyin_conv, pinyin_conv.shape[-1])
        return pinyin_embed.view(bs, sentence_length, self.pinyin_out_dim)


class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('./pretrained/bert-base-chinese',
                                              num_labels=num_classes)
        self.classifier = nn.Linear(768, 5)
        self.dropout = nn.Dropout(0.4)  # 添加 Dropout

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        logits = self.classifier(pooler_output)
        return last_hidden_state, pooler_output, logits


class SentimentEmbedding(nn.Module):
    def __init__(self, embedding_size: int):
        super(SentimentEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.bert = BertClassifier(num_classes=5)
        self.tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-chinese')

    def forward(self, input_text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.bert
        model.load_state_dict(torch.load('./sentiment_model/sentiment_bert.pt'))
        inputs = input_text
        attention_mask = torch.where(input_text != 0, torch.tensor(1, device=input_text.device),
                                    torch.tensor(0, device=input_text.device))

        model.to(device)
        last_hidden_state, pooler_output, logits = self.bert(inputs, attention_mask=attention_mask)
        sentiment_embeddings = last_hidden_state
        return sentiment_embeddings


class FusionBertEmbeddings(nn.Module):

    def __init__(self, config):
        super(FusionBertEmbeddings, self).__init__()
        config_path = os.path.join(config.name_or_path, 'config')
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.pinyin_embeddings = PinyinEmbedding(embedding_size=128, pinyin_out_dim=config.hidden_size,
                                                 config_path=config_path)
        self.sentiment_embeddings = SentimentEmbedding(embedding_size=768)

        self.map_fc = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, pinyin_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        word_embeddings = inputs_embeds
        pinyin_embeddings = self.pinyin_embeddings(pinyin_ids)
        sentiment_embeddings = self.sentiment_embeddings(input_ids)

        concat_embeddings = torch.cat((word_embeddings, pinyin_embeddings, sentiment_embeddings), 2)
        inputs_embeds = self.map_fc(concat_embeddings)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
