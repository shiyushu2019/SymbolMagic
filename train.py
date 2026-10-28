import argparse
import os
import tqdm
import torch
from model.MLP import MyModel
from utils.resolution import LazyDataset, ActDataset, MathExpr
import torch.nn.utils as U
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ =="__main__":

    PATH="./out"
    os.makedirs(PATH, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("formula")
    parser.add_argument("--name",type=str,default="new_project")
    #---------model parameters---------
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu", "tanh", "sin", "sigmoid"])
    #---------training parameters---------
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1","smooth_l1","huber"])
    #---------data parameters---------
    parser.add_argument("--lower_bound", type=float, default=0.0)
    parser.add_argument("--upper_bound", type=float, default=10.0)
    parser.add_argument("--step", type=float, default=1e-4)
    parser.add_argument("--lazy_or_act", type=str,default="lazy", choices=["lazy", "act"])
    #---------other parameters---------
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

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

    model=MyModel(
                input_size=len(dataset.expr.vars),
                output_size=1,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                activation=args.activation,
        ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6
    print(f"Model Params: {params_m:.6f} M")

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
    optimizer=optimizer(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        bar=tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss=0.0
        for i, (x, y) in enumerate(bar):
            x=x.to(DEVICE)
            y=y.to(DEVICE)
            y_pred=model(x)
            loss=loss_fn(y_pred, y)
            total_loss+=loss.item()
            loss.backward()
            grad_norm=U.clip_grad_norm_(model.parameters(),max_norm=1.0)
            bar.set_postfix({"loss": total_loss/(i+1),"grad_norm":grad_norm.item()})
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(dataloader)}")

    checkpoint={
        "name": args.name,
        "formula": args.formula,
        "model_state_dict": model.state_dict(),
        "args": args
    }
    torch.save(checkpoint, f"{PATH}/{args.name}.pt")
