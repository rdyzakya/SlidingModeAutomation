from argparse import ArgumentParser
import torch
from model import SlidingMode
from trainer import Trainer
from visualizer import Visualizer

def init_args():
    parser = ArgumentParser()
    parser.add_argument("--time", type=int, help="Time limit", default=10)
    parser.add_argument("--delta", type=float, help="Delta", default=0.1)
    parser.add_argument("--epoch", type=int, help="Epoch", default=1000)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.2)
    parser.add_argument("--config", type=str, help="Input config", default="input.yaml")
    parser.add_argument("--video-system", type=str, help="Video path (ex: video.mp4)")
    parser.add_argument("--video-loss", type=str, help="Video path (ex: video.mp4)")
    parser.add_argument("--seed", type=int, help="Seed random", default=42)
    parser.add_argument("--init", type=str, help="Initial param mode", default="ones")
    parser.add_argument("--reduction", type=str, help="Reduction mode for mse (mean or sum)", default="sum")
    parser.add_argument("--optim", type=str, help="Optimizer name, refer to torch documentation", default="Adam")
    parser.add_argument("--log", type=int, help="Logging steps", default=10)
    parser.add_argument("--out", type=str, help="Out path for history json file", default="./history.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode on")
    parser.add_argument("--fps", type=int, help="FPS", default=10)
    parser.add_argument("--n-points", type=int, help="Num points for loss visualization", default=100)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = init_args()

    torch.manual_seed(42)

    model = SlidingMode(args.config, init=args.init, mse_reduction=args.reduction)

    trainer = Trainer(model, optimizer=args.optim, epoch=args.epoch, learning_rate=args.lr, time=args.time, delta=args.delta, logging_steps=args.log, out_path=args.out)

    trainer.train(verbose=bool(args.verbose))

    trainer.save()

    visualizer = Visualizer(trainer, fps=args.fps)

    if args.video_system:
        visualizer.system_graphic_figs()
        visualizer.figs_to_video(dir_path="system_graphic", out_path=args.video_system, delete_dir=True)
    if args.video_loss:
        visualizer.loss_landscape_figs(num_points=args.n_points)
        visualizer.figs_to_video(dir_path="loss_landscape", out_path=args.video_loss, delete_dir=True)