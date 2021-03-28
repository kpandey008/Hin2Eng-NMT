import contractions
import csv
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


if __name__ == '__main__':
    raw_file_path = '/home/lexent/Hin2Eng-NMT/nmt/data/raw/train.csv'
    de_save_path = '/home/lexent/de_cleaned.csv'
    en_save_path = '/home/lexent/en_cleaned.csv'
    generate_filtered_datasets(raw_file_path, de_save_path, en_save_path)
