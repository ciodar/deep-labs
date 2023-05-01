class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for (in_dim,out_dim) in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim,out_dim))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = x.view(x.shape[0],-1)
        out = self.layers(out)
        return out