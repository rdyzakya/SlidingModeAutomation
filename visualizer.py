import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import shutil
from tqdm import tqdm
import warnings

class Visualizer:
    def __init__(self, trainer, fps=10):
        self.trainer = trainer
        self.fps = fps
    
    def figs_to_video(self, dir_path, out_path, delete_dir=False):
        # Compile images into a video
        images = []
        for h in self.trainer.history:
            filename = os.path.join(dir_path, f"epoch_{h['epoch']}.png")
            images.append(imageio.imread(filename))

        # Save the video
        imageio.mimsave(out_path, images, fps=self.fps)

        print(f"Video saved as {out_path}")

        if delete_dir:
            shutil.rmtree(dir_path)

    def system_graphic_figs(self):

        os.makedirs('system_graphic', exist_ok=True)

        time_steps = np.arange(0.0, self.trainer.time + self.trainer.delta, self.trainer.delta) # time + delta so the time window would be [0..time] and not [0..time)

        for h in tqdm(self.trainer.history, desc="Create system graphics figures"):
            system_variables, desired_values, loss = self.trainer.model(self.trainer.time, self.trainer.delta)

            fig, ax = plt.subplots(1, len(self.trainer.model.param_names), figsize=(16,9))
            sv_name = list(self.trainer.model.system_variable_names)
            for i in range(len(ax)):
                ax[i].plot(time_steps, system_variables[:,i].numpy(), label="X Output", color="blue")
                ax[i].plot(time_steps, desired_values[:,i].numpy(), label="X Desired", color="red", linestyle="--")
                ax[i].title.set_text(f"{sv_name[i]} - t")
                ax[i].legend()
            super_title = [f"{k} : {v}" for k, v in zip(h["param_names"], h["param_values"])]
            plt.suptitle(" | ".join(super_title))

            filename = f'system_graphic/epoch_{h["epoch"]}.png'
            plt.savefig(filename)
            plt.close(fig)
    
    def loss_landscape(self, num_points=100):
        if self.trainer.model.params.shape[0] != 2:
            warnings.wanr(f"Loss landscape only can be made for 2 parameter systems, your system contains {self.trainer.model.params.shape[0]} parameters")
            return None, None, None
        param_values = [h["param_values"] for h in self.trainer.history]

        max_param = np.ceil(np.max(param_values).item())
        min_param = np.ceil(np.min(param_values).item())

        # Generate x and y values
        x_values = np.linspace(min_param, max_param, num_points)
        y_values = np.linspace(min_param, max_param, num_points)

        # Create a meshgrid from x and y values
        X, Y = np.meshgrid(x_values, y_values)

        Z = np.zeros_like(X) # loss

        bar = tqdm(total=num_points**2, desc="Create loss landscape")
        for i in range(num_points):
            for j in range(num_points):
                Z[i,j] = self.trainer.model.forward(self.trainer.time, self.trainer.delta, param_values=torch.tensor([X[i,j], Y[i,j]]))[2].numpy()
                bar.update(1)
        
        return X, Y, Z
    
    def loss_landscape_figs(self, num_points=100):
        os.makedirs('loss_landscape', exist_ok=True)

        X, Y, Z = self.loss_landscape(num_points)

        if (X, Y, Z) == (None, None, None):
            return
        
        # Without tracking point
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

        # Add labels and title
        ax.set_xlabel(self.trainer.model.param_names[0]) # X
        ax.set_ylabel(self.trainer.model.param_names[1]) # Y
        ax.set_zlabel('Loss') # Z
        ax.set_title('Optimization Visualization')

        filename = './loss_landscape.png'
        plt.savefig(filename)
        plt.close(fig)
        
        # With tracking point
        trackline = {
            "loss" : [],
            self.trainer.model.param_names[0] : [],
            self.trainer.model.param_names[1] : []
        }
        for i, h in tqdm(enumerate(self.trainer.history), desc="Create loss landscape figures"):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot the surface
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
            trackline["loss"].append(h["loss"])
            trackline[h["param_names"][0]].append(h["param_values"][0])
            trackline[h["param_names"][1]].append(h["param_values"][1])
            # Scatter
            if i > 0:
                ax.plot(trackline[h["param_names"][0]], trackline[h["param_names"][1]], trackline["loss"], alpha=0.5, color="red")
            ax.scatter(trackline[h["param_names"][0]][-1], trackline[h["param_names"][1]][-1], trackline["loss"][-1], color="red", edgecolor="black", s=50, label="SGD")

            # Add labels and title
            ax.set_xlabel(h["param_names"][0]) # X
            ax.set_ylabel(h["param_names"][1]) # Y
            ax.set_zlabel('Loss') # Z
            ax.set_title('Optimization Visualization')

            filename = f'loss_landscape/epoch_{h["epoch"]}.png'
            plt.savefig(filename)
            plt.close(fig)