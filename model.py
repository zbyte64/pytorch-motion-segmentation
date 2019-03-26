import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
from collections import OrderedDict


class ProximalEmbedding(nn.Module):
    def __init__(self, n_channels, k=3, padding=1):
        super(ProximalEmbedding, self).__init__()
        self.n_channels = n_channels
        self.k = k
        self.padding = padding
        proximity_filter = torch.zeros(n_channels, n_channels, k, k)
        for i in range(self.n_channels):
            proximity_filter[i, i] = torch.ones(k, k) / (k-1)
            proximity_filter[i, i, k//2, k//2] = 0
        self.register_buffer('proximity_filter', proximity_filter)
        self.transit = nn.Parameter(torch.ones(n_channels, n_channels))

    def forward(self, x):
        return F.softmax(
            #x * self.transit,
            torch.einsum('ijkl,mn->inkl', x, self.transit),
            dim=1
        )

    def log_prob(self, x):
        p = F.conv2d(x, self.proximity_filter, padding=self.padding)
        y = self(x)
        m = D.Bernoulli(y)
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
        self.softmax = nn.LogSoftmax(dim=1)
        self._collector = dict()

    def receive_observation(self, module, input_, output_):
        self._collector[self.layer_index[module]] = output_

    def get_layer_values(self, x):
        with torch.no_grad():
            y = self._decoder(x)
        d = self._collector
        self._collector = dict()
        return d, y

    def forward(self, x, train_embed=False):
        h, w = x.shape[2:4]
        d, y = self.get_layer_values(x)
        o_m, o_lm = self._object_masks(h, w, d)
        if train_embed:
            return o_m, self._embeding_log_prob(d), o_lm
        return o_m

    def _object_masks(self, h, w, d):
        object_layer_masks = []
        object_layer_mass = []
        for i, masking_layer in enumerate(self.masking_layers):
            y_i = d[i]
            layer_embedding = self.embeddings[i]
            with torch.no_grad():
                layer_mass = layer_embedding(y_i)
            object_mask = F.relu(masking_layer(layer_mass))
            if object_mask.shape[2:4] != (h, w):
                object_mask = F.interpolate(object_mask, size=(h, w))
            object_layer_masks.append(object_mask)
            object_layer_mass.append(layer_mass)
        object_layers = torch.cat(object_layer_masks, dim=1)
        object_masks = torch.abs(self.softmax(self.global_masking_layer(object_layers).clamp(1e-6, 1e6)))
        return object_masks, object_layer_mass

    def embedding_log_prob(self, x):
        d, y = self.get_layer_values(x)
        return self._embeding_log_prob(d)

    def _embeding_log_prob(self, d):
        l = []
        for i, masking_layer in enumerate(self.masking_layers):
            y_i = d[i]
            layer_embedding = self.embeddings[i]
            layer_prob = layer_embedding.log_prob(y_i)
            p = layer_prob.view(layer_prob.shape[0], -1).sum(dim=1, keepdim=True)
            l.append(p)
        batch_probs = torch.cat(l, dim=1)
        return batch_probs.mean(dim=1, keepdim=True)


def masked_transit_loss(a_vector, b_vector, a_mask, b_mask):
    '''
    a_vector: frame a encoded mass [b,v,h,w]
    b_vector: frame b encoded mass [b,v,h,w]
    a_mask: [b,n,h,w]
    b_mask: [b,n,h,w]
    '''
    batch_size = a_vector.shape[0]
    #apply_mask = lambda m,v: m.view(batch_size, m.shape[1], -1) * v.view(batch_size, v.shape[1], -1)
    apply_mask = lambda m,v: torch.einsum('bvhw,bmhw->bvmhw',v,m)
    object_mass = apply_mask(a_mask * b_mask, a_vector - b_vector)
    # frame & not other_frame
    object_transit = (
        apply_mask(a_mask * (1-b_mask), a_vector) -
        apply_mask(b_mask * (1-a_mask), b_vector)
    )
    object_distance = (object_transit**2 + object_mass**2) ** .5
    loss = torch.sum(object_distance.view(batch_size, -1), dim=1, keepdim=True)
    #print(torch.sum(object_mass), torch.sum(object_transit))
    #print(torch.sum(a_vector - b_vector))
    #print(torch.sum(b_mask), torch.sum(a_mask))
    return loss


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
