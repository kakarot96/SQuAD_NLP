import os
import numpy as np
import re
from tqdm import tqdm
import mmap

def binary_search(lst, target):
    low = 0
    high = len(lst) - 1
    while low <= high:
        avg = (low + high) / 2
        if lst[avg][0] == target:
            return lst[avg][1]
        elif lst[avg][0] < target:
            low = avg + 1
        else:
            high = avg - 1

    return -1

def tokenizer(sentence):
    tokens = []
    for word in sentence.strip().split():
        tokens.extend(re.split(" ", word))

    return [w for w in tokens if w]

def get_num_lines(path):
    
    fp = open(path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1

    return lines

def initialize_vocab(vocab_path):
    if os.path.exists(vocab_path):
        rev_vocab = []
        with open(vocab_path, "r") as f:
            for line in f:
                rev_vocab.append(line.strip('\n'))
        vocab = dict([(x,y) for (y,x) in enumerate(rev_vocab)])
        return vocab, rev_vocab

    else:
        return ValueError("File {} does not found".format(vocab_path))

def create_vocabulary(vocab_dir, data_paths):
    if not os.path.exists(os.path.join(vocab_dir, 'vocab.dat')):
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)
        vocab = {}
        for path in data_paths:
            with open(path, 'r') as f:
                counter = 0
                for line in f:
                    words = tokenizer(line)
                    for word in words:
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1

        vocab_list = sorted(vocab, key=vocab.get, reverse=True)
        with open(os.path.join(vocab_dir, 'vocab.dat'), 'w') as fh:
            for w in vocab_list:
                fh.write(w + b"\n")
    else:
        print "Vocabulary is already created!"
                    
def process_glove(glove_dir, vocab, glove_dim):
    vocab_with_idx = [(value, counter) for counter, value in enumerate(vocab)]
    vocab_with_idx.sort()
    
    if not os.path.exists(os.path.join(glove_dir, "glove.trimmered_{}.npz".format(glove_dim))):
        glove_path = os.path.join(glove_dir, 'glove.6B.{}d.txt'.format(glove_dim))
        
        glove = np.zeros((len(vocab), glove_dim))
        found = 0

        with open(glove_path, 'r') as f:
            for line in tqdm(f, total=4e5):
                arr  = line.strip().split()
                word = arr[0]
                vector = list(map(float, arr[1:]))
                
                find = False
                idx = binary_search(vocab_with_idx, word)
                
                if idx != -1:
                    glove[idx, :] = vector
                    found += 1
                
                idx = binary_search(vocab_with_idx, word.capitalize())
                if idx != -1:
                    glove[idx, :] = vector 
                    found += 1
                
                idx = binary_search(vocab_with_idx, word.upper())
                if idx != -1: 
                    glove[idx, :] = vector 
                    found += 1

        print glove[0]
        print "{} word out of {} word found in glove data".format(found, len(vocab))
        np.savez_compressed(os.path.join(glove_dir, "glove.trimmered_{}".format(glove_dim)), glove=glove)
    
    else:
        print "Trimmed glove file is already presented.."

def sentence_to_ids(sentence, vocab):
    words = tokenizer(sentence)

    return [vocab.get(w, 'UNK_ID') for w in words]

def data_to_ids(data_path, vocab, save_path):

    if not os.path.exists(save_path):
        print "Tokenizing data in {}".format(data_path)

        with open(data_path, 'r') as data_file:
            with open(save_path, 'w') as save_file:
                
                for line in tqdm(data_file, total = get_num_lines(data_path)):

                    tokens = sentence_to_ids(line, vocab)
                    save_file.write(" ".join([str(tok) for tok in tokens]) + "\n")
    
    else:
        print "{} already exists..".format(save_path)

if __name__=='__main__':

    train_dir = os.path.join('data/processed/squad', 'train')
    dev_dir = os.path.join('data/processed/squad', 'dev')
    glove_dir = os.path.join('data/download', 'glove')
    vocab_dir = os.path.join('data', 'processed', 'vocab')

    create_vocabulary(vocab_dir, [train_dir + '/contexts',
                                  train_dir + '/questions',
                                  dev_dir + '/contexts',
                                  dev_dir + '/questions'])

    vocab, rev_vocab = initialize_vocab(vocab_dir + '/vocab.dat')

    process_glove(glove_dir, rev_vocab, 100)
    
    context_train_ids_path = os.path.join(train_dir, "ids.contexts")
    question_train_ids_path = os.path.join(train_dir, "ids.questions")

    data_to_ids(train_dir + '/contexts', vocab, context_train_ids_path)
    data_to_ids(train_dir + '/questions', vocab, question_train_ids_path)

    context_dev_ids_path = os.path.join(dev_dir, "ids.contexts")
    question_dev_ids_path = os.path.join(dev_dir, "ids.questions")
    
    data_to_ids(dev_dir + '/contexts', vocab, context_dev_ids_path)
    data_to_ids(dev_dir + '/questions', vocab, question_dev_ids_path)


