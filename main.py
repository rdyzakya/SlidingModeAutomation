import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml
import re
import json
import os
import shutil
import imageio.v2 as imageio
# import numpy as np
from tqdm import tqdm

from argparse import ArgumentParser

def init_args():
    parser = ArgumentParser()
    parser.add_argument("--time", type=int, help="Time limit", default=10)
    parser.add_argument("--delta", type=float, help="Delta", default=0.1)
    parser.add_argument("--epoch", type=int, help="Epoch", default=1000)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.2)
    parser.add_argument("--config", type=str, help="Input config", default="input.yaml")
    parser.add_argument("--video", type=str, help="Video path (ex: video.mp4)")

    args = parser.parse_args()
    return args

def parse_equation(equation_string, system_variables, control_variables, params):
    result = equation_string

    # 2. Change system variables
    for i, sv in enumerate(system_variables):
        result = re.sub(sv, f"sv[{i}]", result)
    
    # 3. Change control variables
    for i, cv in enumerate(control_variables):
        result = re.sub(cv, f"cv[{i}]", result)
    
    # 4. Change params
    for i, par in enumerate(params):
        result = re.sub(par, f"par[{i}]", result)
    def func(sv, cv, par, t):
        eq = eval(result)
        return eq

    return func

def make_video(args, history, system_functions, control_functions, desired_functions, initial):

    os.makedirs('figures', exist_ok=True)

    time_steps = torch.arange(0.0, args.time, args.delta)
    for h in tqdm(history):
        parameter_values = torch.tensor(h["param_values"])
        system_variables = torch.zeros((time_steps.shape[0], len(initial)))
        system_variables[0,:] = torch.tensor([el for el in initial.values()], dtype=torch.float32)

        for i, t in enumerate(time_steps[1:]):
            # calculate control variables
            control_variables = [cf(system_variables[i], None, parameter_values, t) for cf in control_functions]
            control_variables = torch.stack(control_variables)

            # calculate system variables
            dsystem_variables = [sf(system_variables[i], control_variables, parameter_values, t) for sf in system_functions]
            dsystem_variables = torch.stack(dsystem_variables)

            system_variables[i+1] = system_variables[i] + args.delta * dsystem_variables

        desired_values = torch.tensor([[df(None, None, None, t) for df in desired_functions] for t in time_steps])
        
        fig, ax = plt.subplots(1, len(initial), figsize=(16,9))
        sv_name = list(initial.keys())
        for i in range(len(ax)):
            ax[i].plot(time_steps.numpy(), system_variables[:,i].numpy(), label="X Output", color="blue")
            ax[i].plot(time_steps.numpy(), desired_values[:,i].numpy(), label="X Desired", color="red", linestyle="--")
            ax[i].title.set_text(f"{sv_name[i]} - t")
            ax[i].legend()
        super_title = [f"{k} : {v}" for k, v in zip(h["param_names"], h["param_values"])]
        plt.suptitle(" | ".join(super_title))

        filename = f'figures/epoch_{h["epoch"]}.png'
        plt.savefig(filename)
        plt.close(fig)

    # Compile images into a video
    images = []
    for h in history:
        filename = f'figures/epoch_{h["epoch"]}.png'
        images.append(imageio.imread(filename))

    # Save the video
    imageio.mimsave(args.video, images, fps=10)

    print(f"Video saved as {args.video}")

    shutil.rmtree("figures")

if __name__ == "__main__":
    args = init_args()

    torch.manual_seed(42)

    with open(args.config, 'r') as fp:
        config = yaml.safe_load(fp)
    
    system = config["system"]
    control = config["control"]
    desired = config["desired"]
    initial = config["initial"]
    params = config["params"]

    if not(len(system.keys()) == len(desired.keys()) == len(initial.keys())):
        raise ValueError
    else:
        for sk, dk, ik in zip(system.keys(), desired.keys(), initial.keys()):
            if not('d' + dk == 'd' + ik == sk):
                raise ValueError
    
    system_functions = [
        parse_equation(el, system_variables=initial.keys(), control_variables=control.keys(), params=params)
        for el in system.values()
    ]

    control_functions = [
        parse_equation(el, system_variables=initial.keys(), control_variables=control.keys(), params=params)
        for el in control.values()
    ]

    desired_functions = [
        parse_equation(el, system_variables=initial.keys(), control_variables=control.keys(), params=params)
        for el in desired.values()
    ]

    parameter_values = torch.randn(len(params), requires_grad=True).float()

    history = []

    # Optimizer
    optimizer = optim.Adam([parameter_values], lr=args.lr)
    criterion = torch.nn.MSELoss(reduction="sum")

    for e in range(args.epoch):

        system_variables = torch.tensor([el for el in initial.values()], dtype=torch.float32)

        loss = torch.tensor(0.0)

        time_steps = torch.arange(args.delta, args.time, args.delta)
        for t in time_steps:
            # calculate control variables
            control_variables = [cf(system_variables, None, parameter_values, t) for cf in control_functions]
            control_variables = torch.stack(control_variables)

            # calculate system variables
            dsystem_variables = [sf(system_variables, control_variables, parameter_values, t) for sf in system_functions]
            dsystem_variables = torch.stack(dsystem_variables)

            system_variables = system_variables + args.delta * dsystem_variables

            # calculate desired
            desired_values = [df(None, None, None, t) for df in desired_functions]
            desired_values = torch.tensor(desired_values)

            # calculate loss, SE
            loss = loss + criterion(system_variables, desired_values)

        # loss = criterion(preds.view(-1), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        entry = {
            "epoch" : e+1,
            "loss" : loss.item(),
            "param_names" : params,
            "param_values" : parameter_values.tolist()
        }

        history.append(entry)

        if e % 10 == 0:
            msg = [
                f"epoch : {entry['epoch']}",
                f"loss : {entry['loss']}"
            ]
            for k, v in zip(entry["param_names"], entry["param_values"]):
                msg.append(f"{k} : {v:.4f}")
            msg = " | ".join(msg)
            print(msg)
    
    with open("history.json", 'w') as fp:
        json.dump(history, fp)
    
    if args.video:
        make_video(args, history, system_functions, control_functions, desired_functions, initial)