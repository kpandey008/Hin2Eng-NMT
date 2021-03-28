import contractions
import csv
import os
import pandas as pd
import re
import string

from indicnlp.tokenize import indic_tokenize
from indicnlp.script import  indic_scripts as isc
from indicnlp.normalize import indic_normalize
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tqdm import tqdm


def remove_special_characters_hindi(text):
    # Remove all special characters except hindi unicode range (including hindi punctuations).
    # This means that in the english sentence we need to keep `.` and remove
    # other punctuations.
    # Refer to https://unicode.org/charts/PDF/U0900.pdf for Hindi unicode representation
    pat = r'[^\u0900-\u097F,]'
    return re.sub(pat, '', text)


def remove_special_characters(text):
    # Remove all special characters except punctuations (only . and ,)
    # is kept as dheerga virama in hindi is preserved.
    pat = r'[^a-zA-z0-9., ]'
    return re.sub(pat, '', text)


def remove_contractions(text):
    return contractions.fix(text)


def preprocess_hindi_text(hindi_text):
    normalizer = indic_normalize.DevanagariNormalizer()
    normalized_text = normalizer.normalize(hindi_text)
    tokens = []
    for t in indic_tokenize.trivial_tokenize(normalized_text):
        # Remove unwanted symbols from the hindi text
        t_ = remove_special_characters_hindi(t)
        tokens.append(t_)
    tokens = [token for token in tokens if token != '']
    recons_text = ' '.join(tokens)
    if recons_text == "":
        return -1
    return recons_text


def preprocess_english_text(english_text):
    normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
    filtered_text = remove_contractions(english_text)
    filtered_text = normalizer.normalize_str(filtered_text)
    filtered_text = remove_special_characters(filtered_text)
    if filtered_text == "":
        return -1
    return filtered_text


def generate_filtered_datasets(raw_file_path, de_save_path, en_save_path):
    # Generate clean datasets for both hindi and english
    df = pd.read_csv(raw_file_path)

    count = 0
    with open(de_save_path, 'w') as dfp, open(en_save_path, 'w') as efp:
        de_headers = ['Hindi']
        en_headers = ['English']

        de_writer = csv.DictWriter(dfp, fieldnames=de_headers)
        en_writer = csv.DictWriter(efp, fieldnames=en_headers)
        
        de_writer.writeheader()
        en_writer.writeheader()

        for de_text, en_text in tqdm(zip(df['hindi'], df['english'])):
            filtered_de_text = preprocess_hindi_text(de_text)
            filtered_en_text = preprocess_english_text(en_text)

            if filtered_de_text == -1 or filtered_en_text == -1:
                continue
            
            de_writer.writerow({'Hindi': filtered_de_text})
            en_writer.writerow({'English': filtered_en_text})


def train_val_split(de_path, en_path, save_dir, val_prop=0.05):
    de_text = pd.read_csv(de_path)
    en_text = pd.read_csv(en_path)

    n_total = de_text.shape[0]
    n_val = int(val_prop * n_total)

    # Take the last most samples as val set
    de_text_val = de_text[-n_val:]
    en_text_val = en_text[-n_val:]

    de_text_train = de_text[:n_total - n_val]
    en_text_train = en_text[:n_total - n_val]

    # Write files
    os.makedirs(save_dir, exist_ok=True)
    de_text_val.to_csv(os.path.join(save_dir, 'val_hindi.csv'), header=True, index=False)
    en_text_val.to_csv(os.path.join(save_dir, 'val_english.csv'), header=True, index=False)

    de_text_train.to_csv(os.path.join(save_dir, 'train_hindi.csv'), header=True, index=False)
    en_text_train.to_csv(os.path.join(save_dir, 'train_english.csv'), header=True, index=False)


if __name__ == '__main__':
    # raw_file_path = '/home/lexent/Hin2Eng-NMT/nmt/data/raw/train.csv'
    # de_save_path = '/home/lexent/de_cleaned.csv'
    # en_save_path = '/home/lexent/en_cleaned.csv'
    # generate_filtered_datasets(raw_file_path, de_save_path, en_save_path)

    # de_path = '/home/lexent/Hin2Eng-NMT/nmt/data/cleaned/train_hindi.csv'
    # en_path = '/home/lexent/Hin2Eng-NMT/nmt/data/cleaned/train_english.csv'
    # train_val_split(de_path, en_path, '/home/lexent/')
