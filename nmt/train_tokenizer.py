from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def train_de_tokenizer(files, vocab_size=30522, save_path=None):
    de_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    de_tokenizer.pre_tokenizer = Whitespace()
    de_trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    de_tokenizer.train(files, de_trainer)
    de_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", de_tokenizer.token_to_id("[CLS]")),
            ("[SEP]", de_tokenizer.token_to_id("[SEP]")),
        ],
    )
    print(f'Trained Hindi BPE tokenizer')

    if save_path is not None:
        de_tokenizer.save(save_path)


def train_en_tokenizer(files, vocab_size=30522, save_path=None):
    en_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    en_tokenizer.pre_tokenizer = Whitespace()
    en_trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    en_tokenizer.train(files, en_trainer)
    en_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", en_tokenizer.token_to_id("[CLS]")),
            ("[SEP]", en_tokenizer.token_to_id("[SEP]")),
        ],
    )
    print(f'Trained English BPE tokenizer')

    if save_path is not None:
        en_tokenizer.save(save_path)


if __name__ == '__main__':
    de_train = '/home/lexent/Hin2Eng-NMT/nmt/data/cleaned/train_hindi.csv'
    en_train = '/home/lexent/Hin2Eng-NMT/nmt/data/cleaned/train_english.csv'
    train_de_tokenizer([de_train], save_path='/home/lexent/Hin2Eng-NMT/nmt/data/tokenizers/de_tokenizer.json')
    train_en_tokenizer([en_train], save_path='/home/lexent/Hin2Eng-NMT/nmt/data/tokenizers/en_tokenizer.json')
