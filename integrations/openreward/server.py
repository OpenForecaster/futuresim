"""ORS server entrypoint for OpenReward-hosted Futuresim."""

from __future__ import annotations


def main() -> None:
    from openreward.environments import Server

    from integrations.openreward.futuresim_env import FuturesimOpenRewardEnv

    Server([FuturesimOpenRewardEnv]).run()


if __name__ == "__main__":
    main()
