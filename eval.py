import argparse
import os
import tqdm
import torch
from model.MLP import MyModel
from utils.resolution import LazyDataset, ActDataset, MathExpr
from torch.utils.data import DataLoader
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ =="__main__":

    PATH="./out"
    os.makedirs(PATH, exist_ok=True)
    os.makedirs(os.path.join(PATH, "plots"), exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name",type=str,default="new_project")
    #---------data parameters---------
    parser.add_argument("--lower_bound", type=float, default=0.0)
    parser.add_argument("--upper_bound", type=float, default=10.0)
    parser.add_argument("--step", type=float, default=1e-4)
    parser.add_argument("--lazy_or_act", type=str,default="lazy", choices=["lazy", "act"])
    parser.add_argument("--test_bia", type=float, default=5e-5)
    parser.add_argument("--batchsize", type=int, default=32)
    #---------other parameters---------
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    checkpoint=torch.load(os.path.join(PATH, f"{args.name}.pt"), map_location=DEVICE,weights_only=False)

    lower_bound=args.lower_bound+args.test_bia
    if abs(args.lower_bound-checkpoint["args"].lower_bound)>=1e-12:
        print(f"Warning: lower_bound in checkpoint ({checkpoint['args'].lower_bound}) is different from the one provided ({args.lower_bound}). It's ok if you know what you are doing.")
    if abs(args.upper_bound-checkpoint["args"].upper_bound)>=1e-12:
        print(f"Warning: upper_bound in checkpoint ({checkpoint['args'].upper_bound}) is different from the one provided ({args.upper_bound}). It's ok if you know what you are doing.")
    if args.lazy_or_act=="lazy":
        dataset=LazyDataset(
            formula=checkpoint["formula"],
            lower_bound=lower_bound,
            upper_bound=args.upper_bound,
            step=args.step
        )
    else:
        dataset=ActDataset(
            formula=checkpoint["formula"],
            lower_bound=lower_bound,
            upper_bound=args.upper_bound,
            step=args.step
        )
    dataloader=DataLoader(dataset, batch_size=args.batchsize, shuffle=False)

    formula=checkpoint["formula"]
    print(f"Loaded formula: {formula}")
    exp=MathExpr(checkpoint["formula"])
    model=MyModel(input_size=len(exp.vars),
                  output_size=1,
                  hidden_size=checkpoint["args"].hidden_size,
                  num_layers=checkpoint["args"].num_layers,
                  dropout=checkpoint["args"].dropout,
                  activation=checkpoint["args"].activation
                 ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    xs, y_preds, ys = [], [], []
    with torch.no_grad(): 
        total_error=0.0
        for i, (x, y) in enumerate(tqdm.tqdm(dataloader)):
            x=x.to(DEVICE)
            y=y.to(DEVICE)
            y_pred=model(x)
            error=torch.abs(y_pred-y)
            total_error+=error.sum().item()

            if len(exp.vars) == 1:
                xs.append(x.view(-1).cpu())
                y_preds.append(y_pred.view(-1).cpu())
                ys.append(y.view(-1).cpu())

        avg_error=total_error/len(dataset)
        print(f"Average error: {avg_error}")

    if len(exp.vars) == 1 and len(xs) > 0:
        xs = torch.cat(xs).numpy()
        y_preds = torch.cat(y_preds).numpy()
        ys = torch.cat(ys).numpy()
        order = xs.argsort()
        xs, y_preds, ys = xs[order], y_preds[order], ys[order]
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys, label="correct", linewidth=1.5)
        plt.plot(xs, y_preds, label="predict", linewidth=1.5, linestyle="--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{formula}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(PATH,"plots", f"{args.name}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved to {save_path}")