import torch

from Modul import EmbeddingModel

if __name__ == '__main__':
    net = EmbeddingModel(1000, 100)
    net.cuda()
    for i in range(10):
        torch.save(net, './model/embedding_epoch{}_.pt'.format(i))