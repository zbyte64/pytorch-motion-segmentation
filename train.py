import torchvision
import torch
from torch.nn import functional as F
import os
from torch import optim
from model import ObjectMask, ProximalEmbedding, masked_transit_loss, resnet50
from dataset import sample_folder
from torch.utils.data import DataLoader


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 10
    data_dir = os.environ['DATA_DIR']
    model = resnet50()
    model = model.to(device).train()
    model.embeddings.to(device)
    model_optimizer = optim.Adam(model.parameters(), lr=3e-4)
    e_optimizer = optim.Adam(model.embeddings.parameters(), lr=3e-4)
    ds = DataLoader(sample_folder(data_dir), batch_size=batch_size, shuffle=True)
    for i, data in enumerate(ds):
        a_img, b_img = data
        a_img = a_img.to(device)
        b_img = b_img.to(device)
        model.zero_grad()
        model.embeddings.zero_grad()
        a_mask, a_embed_p, a_vector_l = model(a_img, train_embed=True)
        b_mask, b_embed_p, b_vector_l = model(b_img, train_embed=True)
        l_p = 1 / ( -(a_embed_p + b_embed_p) / 2 + 1e-6)
        m_l = list()
        for a_vector, b_vector in zip(a_vector_l, b_vector_l):
            h, w = a_vector.shape[2:4]
            _a_mask = F.interpolate(a_mask, size=(h, w))
            _b_mask = F.interpolate(b_mask, size=(h, w))
            #print(a_vector.shape, b_vector.shape, _a_mask.shape, _b_mask.shape)
            mask_loss = masked_transit_loss(a_vector, b_vector, _a_mask, _b_mask)
            #print(mask_loss.shape)
            m_l.append(mask_loss)
        loss = torch.cat(m_l, dim=1).mean(dim=1)
        loss = loss.sum()
        l_p = l_p.sum()
        if i % 10 == 0:
            print('Losses:', loss.detach().item(), l_p.detach().item())
        if i > 50:
            loss.backward()
            model_optimizer.step()
        l_p.backward()
        e_optimizer.step()


if __name__ == '__main__':
    train()
