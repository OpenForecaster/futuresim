"""ORS server entrypoint for OpenReward-hosted Futuresim."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local Futuresim OpenReward server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    from openreward.environments import Server

    from integrations.openreward.futuresim_env import FuturesimOpenRewardEnv

    Server([FuturesimOpenRewardEnv]).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
