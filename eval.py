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
from matplotlib.ticker import FormatStrFormatter

if __name__ =="__main__":

    PATH="./out"
    os.makedirs(PATH, exist_ok=True)
    os.makedirs(os.path.join(PATH, "plots"), exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name",type=str,default="new_project")
    #---------data parameters---------
    parser.add_argument("--lower_bound", type=float, default=0.0)
    parser.add_argument("--upper_bound", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=1e-5)
    parser.add_argument("--lazy_or_act", type=str,default="lazy", choices=["lazy", "act"])
    parser.add_argument("--test_bia", type=float, default=5e-6)
    parser.add_argument("--batchsize", type=int, default=32768)
    parser.add_argument("--num_workers", type=int, default=16)
    #---------other parameters---------
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    checkpoint=torch.load(os.path.join(PATH, f"{args.name}.pt"), map_location=DEVICE,weights_only=False)

    lower_bound=args.lower_bound+args.test_bia
    if abs(args.lower_bound-checkpoint["args"].lower_bound)>=1e-12:
        print(f"Warning: lower_bound in checkpoint ({checkpoint['args'].lower_bound}) is different from the one provided ({args.lower_bound}). It's ok if you know what you are doing.")
    if abs(args.upper_bound-checkpoint["args"].upper_bound)>=1e-12:
        print(f"Warning: upper_bound in checkpoint ({checkpoint['args'].upper_bound}) is different from the one provided ({args.upper_bound}). It's ok if you know what you are doing.")
    if abs(args.step-checkpoint["args"].step)>=1e-12 :
        print(f"Warning: step in checkpoint ({checkpoint['args'].step}) is different from the one provided ({args.step}). It's ok if you know what you are doing.")
    q = args.test_bia / args.step
    if abs(q - round(q)) < 1e-6:
        print(f"test_bia is a multiple of step, it may cause data leak.")

    if args.debug:
        print("solving data......")
    
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
    dataloader=DataLoader(dataset, batch_size=args.batchsize, shuffle=False,num_workers=args.num_workers)

    # ensure dataset is all legal
    if args.lazy_or_act=="lazy":
        temp=0
        for x in dataloader:
            temp=id(x)
    print(f"samples: {len(dataset)}")

    if args.debug:
        print("data solved!")

    formula=checkpoint["formula"]
    print(f"Loaded formula: {formula}")
    exp=MathExpr(checkpoint["formula"])

    if args.debug:
        print("building model......")

    model=MyModel(input_size=len(exp.vars),
                  output_size=1,
                  hidden_size=checkpoint["args"].hidden_size,
                  num_layers=checkpoint["args"].num_layers,
                  dropout=checkpoint["args"].dropout,
                  activation=checkpoint["args"].activation
                 ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6
    print(f"Model Params: {params_m:.6f} M")

    if args.debug:
        print("model built!")

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
        ax = plt.gca()
        ax.plot(xs, ys, label="correct", linewidth=1.5)
        ax.plot(xs, y_preds, label="predict", linewidth=1.5, linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{formula}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if len(xs) > 1:
            step_ax = ax.inset_axes([0.58, 0.08, 0.36, 0.32])
            step_ax.plot(xs[:5], ys[:5], linewidth=1.2)
            step_ax.plot(xs[:5], y_preds[:5], linewidth=1.2, linestyle="--")
            x_start, x_end = xs[0], xs[1]
            x_left = x_start - args.step * 0.5
            x_right = x_start + args.step * 4.5
            step_ax.set_xlim(x_left, x_right)
            step_ax.set_xticks([x_left, (x_left + x_right) / 2, x_right])
            step_ax.xaxis.set_major_formatter(FormatStrFormatter("%.1e"))
            local_values = list(ys[:5]) + list(y_preds[:5])
            y_min, y_max = min(local_values), max(local_values)
            y_span = max(y_max - y_min, max(abs(y_min), abs(y_max), 1.0) * 0.1)
            arrow_y = y_max + y_span * 0.3
            step_ax.set_ylim(y_min - y_span * 0.15, y_max + y_span * 0.65)
            step_ax.annotate(
                "",
                xy=(x_end, arrow_y),
                xytext=(x_start, arrow_y),
                arrowprops={"arrowstyle": "<->", "color": "tab:red"},
            )
            step_ax.text(
                (x_start + x_end) / 2,
                arrow_y + y_span * 0.05,
                f"1 step = {args.step:g}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="tab:red",
            )
            step_ax.set_title("One-step interval", fontsize=8)
            step_ax.set_xlabel("x", fontsize=8, labelpad=0)
            step_ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(PATH,"plots", f"{args.name}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved to {save_path}")
