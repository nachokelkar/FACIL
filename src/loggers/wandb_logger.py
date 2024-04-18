import wandb

from loggers.exp_logger import ExperimentLogger
import numpy as np


class Logger(ExperimentLogger):
    """Characterizes a WandB logger"""

    def __init__(self, log_path, exp_name, begin_time=None):
        super(Logger, self).__init__(log_path, exp_name, begin_time)
        wandb.init(
            project="ideas-tlc",
            config={
                "exp_name": exp_name
            },
            reinit=False
        )

    def log_scalar(self, task, iter, name, value, group=None, curtime=None):
        wandb.log(
            {
                "task": task,
                "iter": iter,
                name: value
            }
        )

    def log_figure(self, name, iter, figure, curtime=None):
        wandb.log({name: figure, "iter": iter})

    def log_args(self, args):
        wandb.config = args.__dict__

    def log_result(self, array, name, step):
        if array.ndim == 1:
            # log as scalars
            wandb.run.summary[name + "_" + step] = array[step]

        elif array.ndim == 2:
            s = ""
            i = step
            # for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                s += '{:5.1f}% '.format(100 * array[i, j])
            if np.trace(array) == 0.0:
                if i > 0:
                    s += '\tAvg.:{:5.1f}% \n'.format(100 * array[i, :i].mean())
            else:
                s += '\tAvg.:{:5.1f}% \n'.format(100 * array[i, :i + 1].mean())
            wandb.run.summary[name + "_" + step] = s

    def __del__(self):
        wandb.finish()
