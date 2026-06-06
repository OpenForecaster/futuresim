"""OpenReward deployment entrypoint.

OpenReward's default Docker examples run ``python server.py`` from the
repository root. Keep the real implementation in ``integrations.openreward`` so
the same code is also available as the ``futuresim-openreward-server`` console
script when installed from a wheel.
"""

from integrations.openreward.server import main


if __name__ == "__main__":
    main()
