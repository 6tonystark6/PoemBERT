import pandas as pd
from nlpcda import Homophone
from sklearn.utils import shuffle
from tqdm import tqdm


def read_file(file_name):
    data = pd.read_csv(file_name, sep='\t')
    data = shuffle(data)
    return data

comments_data = read_file('./poems.csv')

for _, row in tqdm(comments_data.iterrows(), total=len(comments_data)):
    text, label = getattr(row, 'Text'), getattr(row, 'Sentiment')
    augment = Homophone(create_num=5, change_rate=0.3)
    aug_list = augment.replace(text)
    for sentence in aug_list[1:]:
        comments_data.loc[len(comments_data.index)] = [sentence, label]

comments_data.to_csv('./sentiment_data.csv')