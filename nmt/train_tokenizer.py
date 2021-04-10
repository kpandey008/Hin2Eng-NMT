import os

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer


def train_tokenizer(
    files, vocab_size=30522, backend='wordpiece', unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]',
    cls_token='[CLS]', mask_token='[MASK]', save_path=None, **kwargs
):
    special_tokens = [unk_token, sep_token, pad_token, cls_token, mask_token]
    if backend == 'wordpiece':
        tokenizer = BertWordPieceTokenizer()
    elif backend == 'bytebpe':
        tokenizer = ByteLevelBPETokenizer()
    else:
        raise NotImplementedError(f'The tokenizer scheme {backend} is not yet supported!')
    tokenizer.train(files, vocab_size=vocab_size, special_tokens=special_tokens, **kwargs)
    print(f'Trained tokenizer')

    # Save the tokenizer
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_model(save_path)


if __name__ == '__main__':
    de_train = '/home/lexent/Hin2Eng-NMT/nmt/data/cleaned/train_hindi.csv'
    en_train = '/home/lexent/Hin2Eng-NMT/nmt/data/cleaned/train_english.csv'
    train_tokenizer([de_train, en_train], save_path='/home/lexent/Hin2Eng-NMT/nmt/data/tokenizers')
