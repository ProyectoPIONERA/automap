from automap.utils import get_for_wandb
import wandb


class Eval2WB:
    def __init__(self, project: str = "foo"):
        self.project = project

    def __call__(self, eval_json: dict, run_name: str = None, group_name: str = None) -> dict:
        """Convert evaluation results to a format suitable for Weights & Biases logging.

        Args:
            eval_json (dict): The evaluation results in JSON format.

        Returns:
            dict: A dictionary formatted for Weights & Biases logging.
        """
        self._wb_init(project=self.project, run_name=run_name, group_name=group_name)

        wandb_metrics = get_for_wandb(eval_json)
        self.run.log(wandb_metrics)

    def _wb_login(self):
        pass

    def _wb_init(self, project: str, run_name: str, group_name: str):
        self.run = wandb.init(
            entity="carlos_g",
            project=project,
            name=run_name,
            group=group_name
        )


if __name__ == "__main__":
    import json
    import sys

    eval2wb = Eval2WB(project="foo")
    eval2wb(json.loads(sys.stdin.read()))
