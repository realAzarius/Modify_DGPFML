#Modified from: https://github.com/litian96/FedProx/blob/master/flearn/utils/language_utils.py
#credit goes to: Tian Li (litian96 @ GitHub)

"""Utils for language models."""

import re

import numpy as np
import torch

# ------------------------
# utils for shakespeare.embed.pt dataset
ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    根据给定的索引和大小，返回一个独热向量（one-hot vector）
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    将给定的单个字符转换成其对应的独热向量表示
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    '''returns a list of character indices
    Args:
        word: string or Tensor
    将单词转换成字符索引的列表
    Return:
        indices: int list with length len(word)
    '''
    if isinstance(word, torch.Tensor):
        # 如果word是Tensor，我们需要逐个处理其中的每个字符
        # 假设word Tensor的形状是 (n,)，其中n是字符的数量
        word = word.squeeze()  # 去除单维度，确保word是一维的
        indices = [ALL_LETTERS.find(char.item()) for char in word]
    else:
        # 如果word是字符串，我们按照原来的逻辑处理
        indices = [ALL_LETTERS.find(char) for char in word]
    return indices


def process_x(raw_x_batch):
    """
    Processes a batch of text data by converting each element (word or character) to its index representation.
    处理一批文本数据，将每个单词或字符转换成其索引表示
    Args:
        raw_x_batch (str, list, or tensor): A string representing a word or phrase, a list of words or characters, or a tensor representing a sequence of characters.

    Returns:
        list: A list of lists, where each inner list contains the index representation of a word or character.
    """
    if isinstance(raw_x_batch, str):
        # If raw_x_batch is a string, process it character by character
        return [ALL_LETTERS.find(char) for char in raw_x_batch]
    elif isinstance(raw_x_batch, list):
        # If raw_x_batch is a list, process each element individually
        return [process_x(element) for element in raw_x_batch]
    elif isinstance(raw_x_batch, torch.Tensor):
        # If raw_x_batch is a tensor, process it based on its content
        if raw_x_batch.dim() == 1:
            # Convert the tensor to a list of character indices
            char_indices = raw_x_batch.tolist()
            # Convert the list of indices to a list of characters
            char_list = [ALL_LETTERS[index] for index in char_indices]
            # Process the list of characters as a single string
            return process_x(''.join(char_list))
        else:
            # If the tensor has more than one dimension, raise an error
            raise TypeError(f"Unsupported tensor shape: {raw_x_batch.shape}")
    else:
        # If the input is neither a string, list, nor a tensor, raise an error
        raise TypeError(f"Expected a string, list, or a tensor, but got {type(raw_x_batch)}")

def process_y(raw_y_batch):
    #将目标数据（通常是单词中的单个字符）转换成独热向量表示
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return np.array(y_batch).squeeze()

# ------------------------
# utils for sent140 dataset


def split_line(line):
    '''split given line/phrase into list of words
    Args:
        line: string representing phrase to be split

    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    '''returns index of given word based on given lookup dictionary
    returns the length of the lookup dictionary if word not found
    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    '''
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices

    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer
    representing unknown index to returned list until the list's length is
    max_words
    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list
    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line)  # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id] * (max_words - len(indl))
    return indl


def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab
    Args:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values
    Return:
        integer list
    '''
    bag = [0] * len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


#def process_x(raw_x_batch):
    #x_batch = [word_to_indices(word) for word in raw_x_batch]
    #x_batch = np.array(x_batch).T
    #x_batch = x_batch.squeeze()
    #return x_batch


#def process_y(raw_y_batch):
    #y_batch = [letter_to_vec(c) for c in raw_y_batch]
    #return np.array(y_batch).squeeze()


def patch_h_weights(weights, L_next, assignments):
    # e.g. (1024, 256) comes from (256,256)|(256,256)|(256,256)|(256,256)
    def __permutate(weight, assignments, L_next):
        new_w_j = np.zeros((L_next, L_next), dtype=np.float32)
        new_w_j[np.ix_(assignments, assignments)] = weight  # TODO(hwang): make sure if this is correct
        return new_w_j

    split_range = np.split(np.arange(weights.shape[0]), 4)
    h_weights = []
    for indices in split_range:
        # logger.info("assignments: {}".format(assignments))
        tempt_h_w = __permutate(weights[indices, :], assignments, L_next)
        h_weights.append(tempt_h_w)
        # logger.info("equal: {}".format(np.array_equal(tempt_h_w, weights[indices, :])))
    return np.vstack(h_weights)


def patch_biases(biases, L_next, assignments):
    # e.g. (1024, 256) comes from (256,256)|(256,256)|(256,256)|(256,256)
    def __permutate(bias, assignments, L_next):
        new_w_j = np.zeros(L_next)
        new_w_j[assignments] = bias
        return new_w_j

    splitted_bias = np.split(biases, 4)
    h_bias = [__permutate(sb, assignments, L_next) for sb in splitted_bias]
    return np.hstack(h_bias)


def perm_i_weights(w_j, L_next, assignment_j_c):
    split_range = np.split(np.arange(w_j.shape[0]), 4)
    res = []
    for i in range(4):
        cand_w_j = w_j[split_range[i], :]
        temp_new_w_j = np.zeros((L_next, w_j.shape[1]))
        temp_new_w_j[assignment_j_c, :] = cand_w_j
        res.append(temp_new_w_j)
    return np.vstack(res)


def patch_i_weights(weights, L_next, assignments):
    # e.g. (1024, 256) comes from (256,256)|(256,256)|(256,256)|(256,256)
    def __permutate(weight, assignments, L_next):
        new_w_j = np.zeros((L_next, L_next), dtype=np.float32)
        new_w_j[np.ix_(assignments, assignments)] = weight  # TODO(hwang): make sure if this is correct
        return new_w_j

    split_range = np.split(np.arange(weights.shape[0]), 4)
    h_weights = [__permutate(weights[indices, :], assignments, L_next) for indices in split_range]
    return np.hstack(h_weights).T


def patch_i_biases(biases, L_next, assignments):
    # e.g. (1024, 256) comes from (256,256)|(256,256)|(256,256)|(256,256)
    def __permutate(bias, assignments, L_next):
        new_w_j = np.zeros(L_next, dtype=np.float32)
        new_w_j[assignments] = bias
        return new_w_j

    splitted_bias = np.split(biases, 4)
    h_bias = [__permutate(sb, assignments, L_next) for sb in splitted_bias]
    return np.hstack(h_bias)


def perm_i_weights(w_j, L_next, assignment_j_c):
    split_range = np.split(np.arange(w_j.shape[0]), 4)
    res = []
    for i in range(4):
        cand_w_j = w_j[split_range[i], :]
        temp_new_w_j = np.zeros((L_next, w_j.shape[1]))
        temp_new_w_j[assignment_j_c, :] = cand_w_j
        res.append(temp_new_w_j)
    return np.vstack(res)
