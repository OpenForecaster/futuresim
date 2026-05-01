# Simulation Job Submission

Operational guidance moved to `.agents/skills/run-simulation/`.

Use that skill for local runs, HTCondor submission, explicit scaffold selection, output locations, and resume or restart workflows.

## Codex GPU Batch Runs

This is the launch shape we used for Codex GPT-5.5 resume runs on MPI HTCondor GPU nodes:

```bash
python mpi_scripts/run_sim/submit_sim.py \
  --config configs/minimalHarness/aljazeeraQ12026v37_codex_gpt55_resume.yaml \
  --runs 1 \
  --set sim_name=codex_aljazeeraQ12026v37_gpt55_low_resume \
  --set defaults.codex_path="$(command -v codex)" \
  --set defaults.reasoning_effort=low \
  --set defaults.sandbox=true \
  --set defaults.sandbox_proc_mode=host_ro \
  --set resources.memory_gb=240
```

Notes:

- Prefer `defaults.codex_path="$(command -v codex)"` for personal launches. If Codex is not on `PATH` in your batch environment, pass your own absolute path on the command line rather than committing it to a shared config.
- `defaults.sandbox_proc_mode=host_ro` keeps the bwrap filesystem sandbox but binds the Condor job's `/proc` read-only. Use this on nodes where `bwrap --proc /proc` fails with a permission error.
- `resources.memory_gb=240` avoids Condor image-size holds we saw with the default 80 GB request. The run itself did not appear to use that much RSS; this is a scheduler-side safety margin.
- `socat` and `bwrap` must be available on the execute node through the job environment or `PATH`. The sandboxed Codex process uses `socat` to reach host-side relay sockets for MCP/search services.

The generated per-run config is written under `${FSIM_SIM_LOG_BASE}/<sim_name>/<sim_name>_r00/config.yaml`, and simulation outputs go under `${FSIM_OUTPUT_BASE}/<sim_name>_r00/<timestamp>/`.

Useful monitoring commands:

```bash
condor_q <cluster_id> -af ClusterId ProcId JobStatus HoldReason RemoteHost ResidentSetSize ImageSize
tail -f "${FSIM_SIM_LOG_BASE}/codex_aljazeeraQ12026v37_gpt55_low_resume/<cluster_id>.out"
```

vLLM / `transformers` quirks (e.g. Qwen3.5 MoE config load): see [`inference/README.md`](../../inference/README.md).
