import flogging
import holoviews as hv
import panel as pn

from fragile.actions import RandomGaussianPolicy, UniformDtSampler
from fragile.benchmarks import Rastrigin
from fragile.core import FaiRunner
from fragile.functions import FunctionTree
from fragile.shaolin.stream_plots import RGB
from fragile.shaolin.streaming_fai import InteractiveFai


hv.extension("bokeh")
pn.extension("tabulator", theme="dark")


class PlanGymDisplay:
    def __init__(
        self,
    ):
        self.best_img = RGB()
        self._curr_best = -1

    def reset(self, fai):  # noqa: ARG002
        return

    def send(self, fai):
        best_ix = fai.cum_reward.argmax().cpu().item()
        best_img = fai.img[best_ix]
        if best_ix != self._curr_best:
            self.best_img.send(best_img)
            self._curr_best = best_ix

    def __panel__(self):
        return pn.Column(
            pn.Row(
                self.best_img.plot,
                # self.room_grey.plot * self.tree_best_room,
            ),
        )


def main():
    flogging.setup()
    env = Rastrigin(2)

    n_walkers = 10000
    fai = FunctionTree(
        max_walkers=n_walkers,
        env=env,
        dt_sampler=UniformDtSampler(min_dt=1, max_dt=3),
        policy=RandomGaussianPolicy(std=0.05, min=-1.0, max=1.0),
        device="cpu",
        min_leafs=50,
        start_walkers=50,
        minimize=True,
    )
    plot = InteractiveFai(fai)
    runner = FaiRunner(fai, 1000000, plot=plot)
    return pn.panel(pn.Column(runner, plot)).servable()


# if __name__ == "__main__":
main()
