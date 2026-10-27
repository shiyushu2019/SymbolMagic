import argparse
import os
import torch
from model.MLP import MyModel
from utils.resolution import MathExpr
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ =="__main__":

    PATH="./out"
    os.makedirs(PATH, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name",type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    checkpoint=torch.load(os.path.join(PATH, f"{args.name}.pt"), map_location=DEVICE,weights_only=False)
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
    with torch.no_grad(): 
        while(True):
            x_str=input("x: ")
            x=[float(i) for i in x_str.split()]
            if(len(x)!=len(exp.vars)):
                print(f"Expected {len(exp.vars)} variables {exp.vars}, got {len(x)}")
                continue
            y_correct=exp(*x)
            x=torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            y_pred=model(x)
            print(f"Predict: {y_pred.item()}, Correct: {y_correct}, Error: {abs(y_pred.item()-y_correct)}")

        

            



