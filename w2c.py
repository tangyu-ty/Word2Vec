import torch
from scipy.spatial.distance import cosine

from torch import optim
from torch.utils.data import DataLoader

from collections import Counter
import numpy as np
import random

import scipy

from Modul import EmbeddingModel
from WordEmbeddingDataset import WordEmbeddingDataset

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

epochs = 30
MAX_VOCAB_SIZE = 1000  # 包含一个<UNK>
EMBEDDING_SIZE = 100
batch_size = 15000
lr = 0.2
with open('data/text8.train.txt') as f:
    text = f.read()  # 得到文本内容
text = text.lower().split()  # 分割成单词列表
vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))  # 统计并筛选词频最高的的999个，变更成key=word，value=times
vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))  # 把不常用的单词都编码为"<UNK>"，也就是999排名以后的，所有的累加

idx2word = [word for word in vocab_dict.keys()]  # 关键字排序
word2idx = {word: i for i, word in enumerate(idx2word)}  # 逆序

word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)  # 科学计数法，转为numpy.ndarray格式的词频统计

word_freqs = word_counts / np.sum(word_counts)  # 词频百分比
# word_freqs = word_freqs ** (3. / 4.)  # 这是因为word2vec论文里面推荐这么做


dataset = WordEmbeddingDataset(text, word2idx, idx2word, word_freqs, word_counts)


dataloader = DataLoader(dataset, batch_size, shuffle=True)
net = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
#net=torch.load('model/embedding_epoch_new_9_.pt')
net.cuda()
#loss_min=0;

print('*****************')
for e in range(epochs):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = torch.LongTensor(input_labels.long()).cuda()
        pos_labels = torch.LongTensor(pos_labels.long()).cuda()
        neg_labels = torch.LongTensor(neg_labels.long()).cuda()

        optimizer = optim.Adam(params=net.parameters(), lr=lr, )
        optimizer.zero_grad()  # net.zero_grad()
        loss = net.forward(input_labels, pos_labels, neg_labels).mean()
        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print('epoch', e, 'iteration', i, loss.item())
    if e==0:
        loss_min = loss.item()
    if e != 0:
        torch.save(net, './model/embedding_epoch_new_{}_.pt'.format(e))
        loss_min=loss.item()
        lr = lr * 0.1
embedding_weights = net.input_embeddings()