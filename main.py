import argparse
from sympy import arg
import torch
from model.MLP import MyModel
from utils.resolution import LazyDataset, ActDataset, MathExpr
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ =="__main__":

    PATH="./out"

    parser = argparse.ArgumentParser()
    parser.add_argument("formula")
    parser.add_argument("--name",type=str,default="new_project")
    #---------model parameters---------
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu", "tanh", "sin", "sigmoid"])
    #---------training parameters---------
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1","smooth_l1","huber"])
    #---------data parameters---------
    parser.add_argument("--lower_bound", type=float, default=0.0)
    parser.add_argument("--upper_bound", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=1e-5)
    parser.add_argument("--lazy_or_act", type=str,default="lazy", choices=["lazy", "act"])
    #---------other parameters---------
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    exp=MathExpr(args.formula)
    if args.lazy_or_act=="lazy":
        dataset=LazyDataset(
            formula=args.formula,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound,
            step=args.step
        )
    else:
        dataset=ActDataset(
            formula=args.formula,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound,
            step=args.step
        )
    dataloader=DataLoader(dataset, batch_size=args.batchsize, shuffle=True)

    if args.loss=="mse":
        loss_fn=torch.nn.MSELoss()
    elif args.loss=="l1":
        loss_fn=torch.nn.L1Loss()
    elif args.loss=="smooth_l1":
        loss_fn=torch.nn.SmoothL1Loss()
    elif args.loss=="huber":
        loss_fn=torch.nn.HuberLoss()

    if args.optimizer=="adam":
        optimizer=torch.optim.Adam
    elif args.optimizer=="sgd":
        optimizer=torch.optim.SGD

    model=MyModel(
            input_size=len(exp.vars),
            output_size=1,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
    ).to(DEVICE)

    for epoch in range(args.epochs):
        model.train()
        for i, (x, y) in enumerate(dataloader):
            x=x.to(DEVICE)
            y=y.to(DEVICE)
            optimizer=optimizer(model.parameters(), lr=args.lr)
            optimizer.zero_grad()
            y_pred=model(x)
            loss=loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        

            



