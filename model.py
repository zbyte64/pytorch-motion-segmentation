import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
from collections import OrderedDict


class ProximalEmbedding(nn.Module):
    def __init__(self, n_channels, k=3):
        super(ProximalEmbedding, self).__init__()
        self.n_channels = n_channels
        self.k = k
        self.proximity_filter = torch.zeros(n_channels, n_channels, k, k)
        for i in range(self.n_channels):
            self.proximity_filter[i, i] = torch.ones(k, k) / (k-1)
            self.proximity_filter[i, i, k//2, k//2] = 0
        self.transit = nn.Parameter(torch.ones(n_channels, n_channels))

    def forward(self, x):
        return F.softmax(
            x * self.transit
        )

    def log_prob(self, x):
        p = F.conv2d(x, self.proximity_filter)
        m = D.Categorical(self(x))
        d = m.log_prob(p)
        return d


class ObjectMask(nn.Module):
    lookup_depth = lambda m: list(filter(lambda n: hasattr(n, 'out_channels'), m.children()))[-1]
    def __init__(self, decoder, conv_layers, conv_sizes, n_objects):
        super(ObjectMask, self).__init__()
        self._decoder = decoder
        for l in conv_layers:
            l.register_forward_hook(self.receive_observation)
        self.embeddings = nn.ModuleList([ProximalEmbedding(s) for s in conv_sizes])
        self.layer_index = {l:i for i, l in enumerate(conv_layers)}
        self.masking_layers = nn.ModuleList([
            nn.Conv2d(s, n_objects, 1) for l, s in zip(conv_layers, conv_sizes)])
        self.global_masking_layer = nn.Conv2d(n_objects*len(conv_layers), n_objects, 1)
        self._collector = dict()

    def receive_observation(self, module, input_, output_):
        self._collector[self.layerindex[module]] = output_

    def get_layer_values(self, x):
        with torch.no_grad():
            y = self._decoder(x)
        d = self._collector
        self._collector = dict()
        return d, y

    def forward(self, x, train_embed=False):
        h, w = x.shape[2:3]
        d, y = self.get_layer_values(x)
        o_m, o_lm = self._object_masks(h, w, d)
        if train_embed:
            return o_m, self._embeding_log_prob(d), o_lm
        return o_m

    def _object_masks(self, h, w, d):
        object_layer_masks = []
        object_layer_mass = []
        for i, (masking_layer, y_i) in enumerate(zip(self.masking_layers, d)):
            layer_embedding = self.embeddings[i]
            with torh.no_grad():
                layer_mass = layer_embedding(y_i)
            object_mask = F.relu(masking_layer(layer_mass))
            if object_mask.shape[2:3] != (h, w):
                object_mask = F.interpolate(object_mask, size=(h, w))
            object_layer_masks.append(object_mask)
            if layer_mass.shape[2:3] != (h, w):
                layer_mass = F.interpolate(layer_mass, size=(h, w))
            object_layer_mass.append(layer_mass)
        object_layers = torch.cat(object_layer_masks, dim=1)
        object_masks = self.global_masking_layer(object_layers)
        return object_masks, object_layer_mass

    def embedding_log_prob(self, x):
        d, y = self.get_layer_values(x)
        return self._embeding_log_prob(d)

    def _embeding_log_prob(self, d):
        l = []
        for i, (masking_layer, y_i) in enumerate(zip(self.masking_layers, d)):
            y_i = d[i]
            layer_embedding = self.embeddings[i]
            layer_prob = layer_embedding.log_prob(y_i)
            l.append(layer_prob.sum(dim=1, keepdims=True))
        batch_probs = torch.cat(l, dim=1)
        return batch_probs.mean(dim=1)


def masked_transit_loss(a_vector, b_vector, a_mask, b_mask):
    '''
    a_vector: frame a encoded mass [b,v,h,w]
    b_vector: frame b encoded mass [b,v,h,w]
    a_mask: [b,n,h,w]
    b_mask: [b,n,h,w]
    '''
    delta_vector = a_vector - b_vector
    object_mass = a_mask & b_mask * (a_vector - b_vector)
    object_transit = (
        a_mask & (~b_mask) * a_vector -
        b_mask & (~a_mask) * b_vector
    )
    loss = torch.sum((object_transit**2 + object_mass**2) ** .5, dim=1, keepdim=True)


def resnet50(n_objects=5):
    import torchvision
    decoder = torchvision.models.resnet50().eval()
    conv_layers = [decoder.layer1, decoder.layer2, decoder.layer3, decoder.layer4]
    conv_sizes = [
        decoder.layer1[-1].conv3.out_channels,
        decoder.layer2[-1].conv3.out_channels,
        decoder.layer3[-1].conv3.out_channels,
        decoder.layer4[-1].conv3.out_channels,
    ]
    return ObjectMask(decoder, conv_layers, conv_sizes, n_objects)
