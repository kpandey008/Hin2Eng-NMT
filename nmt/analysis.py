import contractions
import csv
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns

from indicnlp.tokenize import indic_tokenize
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from preprocess import remove_accents


sns.set()


def plot_sentence_length(raw_corpus_file_path, save_path=None, hist_kwargs={}, **kwargs):
    hindi_token_lengths = []
    english_token_lengths = []
    print('Reading Parallel corpus!')
    with open(raw_corpus_file_path, 'r') as fp:
        reader = csv.DictReader(fp)
        for line in tqdm(reader):
            hindi_tokens = get_tokens(line['hindi'], lang='hi')
            hindi_token_lengths.append(len(hindi_tokens))

            english_tokens = get_tokens(line['english'], lang='en')
            english_token_lengths.append(len(english_tokens))

    # Plot histogram of the lengths of the sentences
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, **kwargs)
    sns.histplot(hindi_token_lengths, kde=True, ax=ax1, **hist_kwargs)
    ax1.title.set_text('Hindi Sentence Lengths')

    sns.histplot(english_token_lengths, kde=True, ax=ax2, **hist_kwargs)
    ax2.title.set_text('English Sentence Lengths')

    # Save the figure
    if save_path is not None:
        fig.savefig(save_path)

    # Display plot
    plt.show()
    
    # Return some essential statistics (mean, median etc.)
    return {
        'hi_mean': np.mean(hindi_token_lengths),
        'hi_max': np.max(hindi_token_lengths),
        'en_mean': np.mean(english_token_lengths),
        'en_max': np.max(english_token_lengths),
        'hi_median': np.median(hindi_token_lengths),
        'en_median': np.median(english_token_lengths)
    }


def get_noisy_sentence_count(raw_corpus_file_path):
    hindi_pattern = re.compile(r'[^\u0900-\u097F,;?!-" |]')
    english_pattern = re.compile(r'[^a-zA-z0-9.,;?!-" ]')

    # Count statistics placeholders
    noisy_hindi_count = 0
    noisy_english_count = 0
    total_count = 0
    with open(raw_corpus_file_path, 'r') as fp:
        reader = csv.DictReader(fp)
        for line in tqdm(reader):
            total_count += 1
            res1 = hindi_pattern.search(line['hindi'])
            res2 = english_pattern.search(line['english'])
            if res1 is not None:
                noisy_hindi_count += 1
            if res2 is not None:
                noisy_english_count += 1
    print(f'Analyzed {total_count} sentences. Found {noisy_hindi_count} noisy Hindi and {noisy_english_count} noisy English')
    return total_count, noisy_hindi_count, noisy_english_count


def get_contractions_counts(raw_corpus_file_path):
    n_contractions = 0
    total_count = 0
    with open(raw_corpus_file_path, 'r') as fp:
        reader = csv.DictReader(fp)
        for idx, line in tqdm(enumerate(reader)):
            fixed_text = contractions.fix(line['english'])
            if fixed_text != line['english']:
                n_contractions += 1
            total_count += 1
    print(f'Analyzed {total_count} sentences. Found {n_contractions} english sentences with contractions!')
    return total_count, n_contractions


def get_accents_counts(raw_corpus_file_path):
    n_accents = 0
    total_count = 0
    with open(raw_corpus_file_path, 'r') as fp:
        reader = csv.DictReader(fp)
        for idx, line in tqdm(enumerate(reader)):
            fixed_text = remove_accents(line['english'])
            if fixed_text != line['english']:
                n_accents += 1
            total_count += 1
    print(f'Analyzed {total_count} sentences. Found {n_accents} english sentences with Accents!')
    return total_count, n_accents


def get_tokens(sentence, lang='hi'):
    if lang == 'hi':
        return indic_tokenize.trivial_tokenize(sentence)
    if lang == 'en':
        return word_tokenize(sentence)


if __name__ == '__main__':
    raw_corpus_file_path = '/home/lexent/Hin2Eng-NMT/nmt/data/raw/train.csv'
    save_path = '/home/lexent/fig.png'
    # res = plot_sentence_length(raw_corpus_file_path, save_path=save_path, figsize=(16, 8))
    # res = get_contractions_counts(raw_corpus_file_path)
    res = get_accents_counts(raw_corpus_file_path)
    print(res)
