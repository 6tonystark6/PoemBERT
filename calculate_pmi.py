import math
from collections import Counter
import json
from transformers import AutoTokenizer
from tqdm import tqdm


def read_sentences_from_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentences.append(line.strip())
    return sentences


def calculate_pmi_for_pairs(sentences):
    cooccurrence_counts = Counter()
    char_counts = Counter()
    total_sentences = len(sentences)

    for sentence in tqdm(sentences, desc="Counting co-occurrences"):
        unique_chars = set(sentence)
        char_counts.update(unique_chars)
        cooccurrence_counts.update([(char_a, char_b) for char_a in unique_chars for char_b in unique_chars if char_a != char_b])

    pmi_values = {}
    for (char_a, char_b), cooccurrence_count in tqdm(cooccurrence_counts.items(), desc="Calculating PMI"):
        p_a = char_counts[char_a] / total_sentences
        p_b = char_counts[char_b] / total_sentences
        p_ab = cooccurrence_count / total_sentences
        pmi = math.log(p_ab / (p_a * p_b))
        pmi_values[(char_a, char_b)] = pmi

    return pmi_values


def calculate_character_importance(pmi_values):
    char_importance = Counter()
    for (char_a, char_b), pmi in tqdm(pmi_values.items(), desc="Calculating character importance"):
        if pmi > 0:
            char_importance[char_a] += pmi
            char_importance[char_b] += pmi
    return char_importance


def save_character_importance_to_file(char_importance, file_path, tokenizer):
    tokenized_char_importance = {str(tokenizer(char)["input_ids"][1]): importance for char, importance in char_importance.items()}
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(tokenized_char_importance, file, ensure_ascii=False, indent=2)


def load_character_importance_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        char_importance = json.load(file)
    return char_importance


tokenizer = AutoTokenizer.from_pretrained("./pretrained/bert-base-chinese")
file_path = './dataset/merged_dataset.txt'
sentences = read_sentences_from_file(file_path)

pmi_values = calculate_pmi_for_pairs(sentences)

char_importance = calculate_character_importance(pmi_values)

save_character_importance_to_file(char_importance, 'char_importance.json', tokenizer)
