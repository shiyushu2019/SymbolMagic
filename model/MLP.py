import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ResidualBlock(nn.Module):
    def __init__(self, 
                size,
                *, dropout,
                action="gelu"):
        super(ResidualBlock, self).__init__()
        if action not in ["relu", "gelu","tanh","sin","sigmoid"]:
            raise ValueError
        if action=="gelu":
            self.activation=nn.GELU()
        elif action=="relu":
            self.activation=nn.ReLU()
        elif action=="tanh":
            self.activation=nn.Tanh()
        elif action=="sin":
            self.activation=torch.sin
        elif action=="sigmoid":
            self.activation=torch.sigmoid
        self.linear1 = nn.Linear(size, size)
        self.norm=nn.LayerNorm(size)
        self.linear2 = nn.Linear(size, size)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return out+residual

class MyModel(nn.Module):
    def __init__(self, 
                *,input_size,
                hidden_size,
                output_size,
                num_layers,
                dropout,
                activation):
        assert num_layers >= 2 and num_layers %2==0
        super(MyModel, self).__init__()
        layers=nn.ModuleList()
        layers.append(nn.Linear(input_size, hidden_size))
        for _ in range((num_layers-2)//2):
            layers.append(ResidualBlock(hidden_size, dropout=dropout, action=activation))
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers=layers
    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
        return x

if __name__ == "__main__":
    model=MyModel(
        input_size=10,
        hidden_size=20,
        output_size=5,
        num_layers=4,
        dropout=0.1,
        activation="gelu"
    ).to(DEVICE)
    with torch.no_grad():
        model.eval()
        x=torch.randn(2,10)
        output=model(x)
        print(output)