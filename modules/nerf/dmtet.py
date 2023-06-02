import torch
import torch.nn as nn
import tqdm

class Embedder:
    def __init__(self,
                 input_dims,
                 include_input,
                 max_freq_log2,
                 num_freqs,
                 log_sampling,
                 periodic_fns,
                 **kwargs):
        self.kwargs = kwargs
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns

        self.create_embedding()
    def create_embedding(self):
        embed_fns = []
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0., self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0, 2. ** self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires):
    embedder_obj = Embedder(
        include_input=True,
        input_dims=3,
        max_freq_log2=multires - 1,
        num_freqs=multires,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos]
    )

    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class Decoder(nn.Module):
    def __init__(self, input_dims = 3, internal_dims = 128, output_dims = 4, hidden = 5, multires = 2):
        super().__init__()
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            input_dims = input_ch

        net = (torch.nn.Linear(input_dims, internal_dims, bias=False), torch.nn.ReLU())
        for i in range(hidden-1):
            net = net + (torch.nn.Linear(internal_dims, internal_dims, bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),)
        self.net = torch.nn.Sequential(*net)

    def forward(self, p):
        if self.embed_fn is not None:
            p = self.embed_fn(p)
        out = self.net(p)
        return out

    def pre_train_sphere(self, iter):
        print ("Initialize SDF to sphere")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)

        for i in tqdm(range(iter)):
            p = torch.rand((1024,3), device='cuda') - 0.5
            ref_value  = torch.sqrt((p**2).sum(-1)) - 0.3
            output = self(p)
            loss = loss_fn(output[...,0], ref_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Pre-trained MLP", loss.item())
class DMTet(nn.Module):
    def __init__(self):
        super().__init__()