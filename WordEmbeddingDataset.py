from torch.utils.data.dataset import Dataset
import torch

C = 3  # context window
K = 15  # number of negative samples


class WordEmbeddingDataset(Dataset):
    def __init__(self, text, word2idx, idx2word, word_freqs, word_counts):
        """ text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            idx2word: index to word mapping
            word_freqs: the frequency of each word
            word_counts: the word counts
        """

        super(WordEmbeddingDataset, self).__init__()  # #通过父类初始化模型，然后重写两个方法

        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]
        # 返回单词在词典中的数字下表，把单词数字化表示。如果不在词典中，也表示为unk
        self.text_encoded = torch.LongTensor(self.text_encoded)  # nn.Embedding需要传入LongTensor类型
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.word_freqs = torch.Tensor(word_freqs)#词频
        self.word_counts = torch.Tensor(word_counts)#总数

    def __len__(self):
        return len(self.text_encoded)  # 返回所有单词的总数，即item的总数

    def __getitem__(self, idx):
        """ 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        """
        center_words = self.text_encoded[idx]  # 取得中心词
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))  # 先取得中心左右各C个词的索引,C在前面定义了
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]  # 为了避免索引越界，所以进行取余处理
        pos_words = self.text_encoded[pos_indices]  # tensor(list)，获得indices下标的tensor的word,这里是2*C个

        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)  # K 是负采样
        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量

        return center_words, pos_words, neg_words
