import argparse
import torch
from model.MLP import MyModel

if __name__ =="__main__":

    PATH="./out"

    parser = argparse.ArgumentParser()
    parser.add_argument("formula")
    parser.add_argument("--name",type=str,default="new_project")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--dropout", type=int, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--device", type=str, default="cuda")
    #parser.add_argument("--verbose", action="store_true", help="显示详细计算过程")

    args = parser.parse_args()

    model=MyModel(
            input_size=10, #
            output_size=5, #
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
    )

