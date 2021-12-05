from collections import Counter

import numpy as np
import torch
from scipy.spatial.distance import cosine
from torch import optim
from torch.utils.data import DataLoader

from WordEmbeddingDataset import WordEmbeddingDataset

from torchsummary import summary


def find_train_nearest(word):
    if word not in train_vocab_dict:
        word = '<UNK>'
    index = train_word2idx[word]
    embedding = embedding_weights[index]

    cos_dis = np.array([cosine(e, embedding) for e in embedding_weights])
    return [train_idx2word[i] for i in cos_dis.argsort()[:10]]


def find(word):
    if word not in dev_vocab_dict:
        word = '<UNK>'
    index = dev_word2idx[word]
    out_embedding = net.out_embeddings[index]

    cos_dis = np.array([cosine(e, out_embedding) for e in embedding_weights])
    return [dev_idx2word[i] for i in cos_dis.argsort()[:10]]


# 查找最近向量

if __name__ == '__main__':
    print("main")
    net = torch.load('model/embedding_epoch_new_25_.pt')
    embedding_weights = net.input_embeddings().detach().numpy()

    with open('data/text8.train.txt') as f:
        train_text = f.read()  # 得到文本内容
    with open('data/text8.dev.txt') as f:
        dev_text = f.read()  # 得到文本内容
    train_text = train_text.lower().split()  # 分割成单词列表
    dev_text = dev_text.lower().split()  # 分割成单词列表
    train_vocab_dict = dict(Counter(train_text).most_common(1000 - 1))  # 统计并筛选词频最高的的999个，变更成key=word，value=times
    dev_vocab_dict = dict(Counter(dev_text).most_common(1000 - 1))  # 统计并筛选词频最高的的999个，变更成key=word，value=times

    train_vocab_dict['<UNK>'] = len(train_text) - np.sum(
        list(train_vocab_dict.values()))  # 把不常用的单词都编码为"<UNK>"，也就是999排名以后的，所有的累加
    dev_vocab_dict['<UNK>'] = len(dev_text) - np.sum(
        list(dev_vocab_dict.values()))  # 把不常用的单词都编码为"<UNK>"，也就是999排名以后的，所有的累加
    train_idx2word = [word for word in train_vocab_dict.keys()]  # 关键字
    dev_idx2word = [word for word in dev_vocab_dict.keys()]  # 关键字
    train_word2idx = {word: i for i, word in enumerate(train_idx2word)}  # 按照词频序去设置kv的idx
    dev_word2idx = {word: i for i, word in enumerate(dev_idx2word)}  # 按照词频序去设置kv的idx
    print("train")
    print("*************")
    for word in ["apple", "america", "computer"]:
        print(word, find_train_nearest(word))
    print("*************")
    print("dev")
    for word in list(dev_vocab_dict.keys())[101:105]:
        print(word, find_train_nearest(word))
    print("*************")
    train_word_counts = np.array([count for count in train_vocab_dict.values()],
                                 dtype=np.float32)  # 科学计数法，转为numpy.ndarray格式的词频统计
    train_word_freqs = train_word_counts / np.sum(train_word_counts)  # 词频百分比
    train_dataset = WordEmbeddingDataset(train_text, train_word2idx, train_idx2word, train_word_freqs,
                                         train_word_counts)

    dev_word_counts = np.array([count for count in dev_vocab_dict.values()],
                               dtype=np.float32)  # 科学计数法，转为numpy.ndarray格式的词频统计
    dev_word_freqs = train_word_counts / np.sum(dev_word_counts)  # 词频百分比
    dev_dataset = WordEmbeddingDataset(dev_text, dev_word2idx, dev_idx2word, dev_word_freqs,
                                       dev_word_counts)
    train_dataloader = DataLoader(train_dataset, 100, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, 100, shuffle=True)

    train_loss = list()

    for i, (input_labels, pos_labels, neg_labels) in enumerate(train_dataloader):
        input_labels = torch.LongTensor(input_labels.long()).cuda()
        pos_labels = torch.LongTensor(pos_labels.long()).cuda()
        neg_labels = torch.LongTensor(neg_labels.long()).cuda()

        loss = net.forward(input_labels, pos_labels, neg_labels).mean()
        train_loss.append(loss)
        if i == 100:
            break
    print("train_mean_loss")
    print(sum(train_loss) / 100)
    print("end")

    dev_loss = list()

    for i, (input_labels, pos_labels, neg_labels) in enumerate(dev_dataloader):
        input_labels = torch.LongTensor(input_labels.long()).cuda()
        pos_labels = torch.LongTensor(pos_labels.long()).cuda()
        neg_labels = torch.LongTensor(neg_labels.long()).cuda()

        loss = net.forward(input_labels, pos_labels, neg_labels).mean()
        dev_loss.append(loss)
        if i == 100:
            break
    print("dev_mean_loss")
    print(sum(dev_loss) / 100)
    print("end")

    input_labels, pos_labels, neg_labels = next(iter(train_dataloader))
    input_labels = torch.LongTensor(input_labels.long()).cuda()
    pos_labels = torch.LongTensor(pos_labels.long()).cuda()
    neg_labels = torch.LongTensor(neg_labels.long()).cuda()
    print(input_labels.shape)
    print(pos_labels.shape)
    print(neg_labels.shape)

    print(dict(net.__dict__.items()).get('_modules'))
