from torch import nn
import torch
import torch.nn.functional as F


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)  # 词点的尺寸和嵌入向量的维度
        # 1k,100
        self.out_embed = nn.Embedding(self.embed_size, self.vocab_size)
        # ，给一个编号，嵌入层就能返回这个编号对应的嵌入向量，嵌入向量反映了各个编号代表的符号之间的语义关系

    def forward(self, input_labels, pos_labels, neg_labels):  # 前向传播
        """ input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]

            return: loss, [batch_size]
        """
        # 得到对应标签的向量
        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.in_embed(pos_labels)  # [batch_size, (window * 2), embed_size]
        neg_embedding = self.in_embed(neg_labels)  # [batch_size, (window * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1]
        # 在第2维度那里加维度

        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        # 矩阵乘法 强制规定维度和大小相同
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]
        # 删除维度

        neg_dot = torch.bmm(neg_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]
        # 负样本的目的是拉开距离，所以距离越小损失越大
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]
        # 删除维度
        log_pos = F.logsigmoid(pos_dot).sum(1)  # .sum()结果只为一个数，.sum(1)结果是一维的张量，按照1维度计算
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg
        # 共同损失

        return -loss

    def input_embeddings(self):
        return self.in_embed.weight.cpu()

    def out_embeddings(self):
        return self.out_embed.weight.cpu()
