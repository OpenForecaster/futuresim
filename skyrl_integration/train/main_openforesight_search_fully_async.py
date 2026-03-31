"""SkyRL entrypoint for fully async OpenForesight warmup-style search training."""

from __future__ import annotations

import asyncio
import traceback

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer

from skyrl_integration.train import main_openforesight_search as sync_main


class FullyAsyncOpenForesightPPOExp(BasePPOExp):
    """Reuse the sync OpenForesight setup, but swap in SkyRL's fully async trainer."""

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def run(self):
        trainer = self._setup_trainer()
        asyncio.run(trainer.train())


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    try:
        sync_main._run_openforesight_entrypoint(cfg, exp_cls=FullyAsyncOpenForesightPPOExp)
    except BaseException:
        sync_main._log_step("skyrl_entrypoint: unhandled exception")
        traceback.print_exc()
        raise


def main() -> None:
    sync_main._run_openforesight_main(skyrl_entrypoint)


if __name__ == "__main__":
    main()
