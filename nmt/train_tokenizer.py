import os

from tokenizers import BertWordPieceTokenizer


def train_tokenizer(files, vocab_size=30522, save_path=None):
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(files, vocab_size=vocab_size, special_tokens = ["[S]","[PAD]","[/S]","[UNK]","[MASK]", "[SEP]","[CLS]"])
    print(f'Trained tokenizer')

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_model(save_path)


if __name__ == '__main__':
    de_train = '/home/lexent/Hin2Eng-NMT/nmt/data/cleaned/train_hindi.csv'
    en_train = '/home/lexent/Hin2Eng-NMT/nmt/data/cleaned/train_english.csv'
    train_tokenizer([de_train, en_train], save_path='/home/lexent/Hin2Eng-NMT/nmt/data/tokenizers')
