import os
from pathlib import Path
import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch.utils import data
from transformers import BertTokenizerFast
import random
import json, gzip
from pypinyin import pinyin, Style


class Dataset(data.Dataset):

    def __init__(self, file_path, vocab_folder="./", doc_len=512, mask_perc=0.15,
                 masking_type="CI-mask"):

        super().__init__()

        self.file_info = []
        self.encodings = {}
        self.curr_file_index = -1
        self.tokenizer = BertTokenizerFast.from_pretrained('./pretrained/bert-base-chinese')
        self.CLS_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.PAD_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.SEP_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.MASK_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        self.masking_type = masking_type
        self.mask_perc = mask_perc
        self.doc_len = doc_len

        self.pmi_dict = None
        self.id_to_prob_dict = [0.0] * 31000
        self.num_pmi_span = 45

        self.num_mask_span = 17
        self.p_geometric = 0.2

        self.vocab_file = os.path.join('./pretrained/bert-base-chinese', 'vocab.txt')
        self.config_path = os.path.join('./pretrained/bert-base-chinese', 'config')
        with open(os.path.join(self.config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        with open(os.path.join(self.config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        with open(os.path.join(self.config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

        p = Path(file_path)

        if (self.masking_type == "CI-mask"):
            with open(vocab_folder + "/char_importance.json", 'r') as file:
                id_to_prob_dict = json.load(file)
                for key, value in id_to_prob_dict.items():
                    self.id_to_prob_dict[int(key)] = float(value)

        print("Masking Type: ", self.masking_type)

        assert (p.is_dir())

        files = sorted(p.glob('*.txt'))
        if len(files) < 1:
            raise RuntimeError('No dataset file found')

        for dataset_fp in files:
            self.file_info.append(dataset_fp)

        random.shuffle(self.file_info)

        self._load_data(file_index=0)

    def __getitem__(self, i):

        file_index = i // self.encodings['input_ids'].shape[0]
        item_index = i % self.encodings['input_ids'].shape[0]

        if (file_index != self.curr_file_index):
            self._load_data(file_index=file_index)

        encoded_dict = {key: tensor[item_index] for key, tensor in self.encodings.items()}

        if (self.masking_type == "CI-mask"):

            input_ids = encoded_dict["labels"].clone()
            rand = torch.rand(input_ids.shape)
            indices = input_ids.tolist()
            values = [0.0] * input_ids.shape[0]

            for i in range(len(indices)):
                wid = indices[i]
                values[i] = self.id_to_prob_dict[wid]

            probs = torch.tensor(values, dtype=torch.float)

            mask_arr = (rand < probs) * (input_ids != self.PAD_id) * (input_ids != self.CLS_id) * (
                    input_ids != self.SEP_id)

            selection = torch.flatten(mask_arr.nonzero()).tolist()
            input_ids[selection] = self.MASK_id

            encoded_dict["input_ids"] = input_ids

        else:

            input_ids = encoded_dict["labels"].clone()
            rand = torch.rand(input_ids.shape)
            mask_arr = (rand < self.mask_perc) * (input_ids != self.PAD_id) * (input_ids != self.CLS_id) * (
                    input_ids != self.SEP_id)

            selection = torch.flatten(mask_arr.nonzero()).tolist()
            input_ids[selection] = self.MASK_id

            encoded_dict["input_ids"] = input_ids

        sentence = self.tokenizer.decode(encoded_dict["labels"], skip_special_tokens=True)
        tokenizer_p = BertWordPieceTokenizer('./pretrained/bert-base-chinese/vocab.txt')
        tokenizer_output = tokenizer_p.encode(sentence)
        pinyin_ids = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        encoded_dict["pinyin_ids"] = pinyin_ids

        return encoded_dict

    def __len__(self):
        return self.encodings['input_ids'].shape[0] * len(self.file_info)

    def _load_data(self, file_index):

        lines = []

        with open(self.file_info[file_index], 'r', encoding='utf-8') as file:
            for line in file:
                lines.append(line.strip())

        encoding_list = self.tokenizer(lines, max_length=self.doc_len, padding='max_length', truncation=True)

        labels = torch.tensor([x for x in encoding_list["input_ids"]])
        mask = torch.tensor([x for x in encoding_list["attention_mask"]])
        self.encodings["input_ids"] = labels
        self.encodings["labels"] = labels.detach().clone()
        self.encodings["attention_mask"] = mask

        self.curr_file_index = file_index

    def convert_sentence_to_pinyin_ids(self, sentence, tokenizer_output):
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids
