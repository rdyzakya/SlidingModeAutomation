import torch
from tqdm import tqdm
import json

ACCEPTABLE_OPTIMIZER = [
    'ASGD',
    'Adadelta',
    'Adagrad',
    'Adam',
    'AdamW',
    'Adamax',
    'LBFGS',
    'NAdam',
    'Optimizer',
    'RAdam',
    'RMSprop',
    'Rprop',
    'SGD',
    'SparseAdam'
]

class Trainer:
    def __init__(self, model, optimizer="Adam", epoch=300, learning_rate=1e-2, time=10, delta=0.1, logging_steps=10, out_path="./out.json"):
        self.model = model
        self.epoch = epoch
        self.learning_rate = learning_rate

        assert optimizer in ACCEPTABLE_OPTIMIZER, f"Optimizer should be from {ACCEPTABLE_OPTIMIZER}"
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=self.learning_rate)

        self.time = time
        self.delta = delta

        self.logging_steps = logging_steps
        self.out_path = out_path

        self.history = []
    
    def train(self, verbose=True):
        self.history = []
        for e in tqdm(range(self.epoch), desc="Searching params"):
            system_variables, desired_values, loss = self.model(self.time, self.delta)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            entry = {
                "epoch" : e+1,
                "loss" : loss.item(),
                "param_names" : self.model.param_names,
                "param_values" : self.model.params.tolist()
            }

            self.history.append(entry)

            if e % self.logging_steps == 0 and verbose:
                msg = [
                    f"epoch : {entry['epoch']}",
                    f"loss : {entry['loss']}"
                ]
                for k, v in zip(entry["param_names"], entry["param_values"]):
                    msg.append(f"{k} : {v:.4f}")
                msg = " | ".join(msg)
                print(msg)
        self.history = self.history
    
    def save(self, out_path=None):
        out_path = out_path or self.out_path
        with open("history.json", 'w') as fp:
            json.dump(self.history, fp)