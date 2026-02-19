- Uses harmony format

- For launching 10 action per q on day 0, 50 each subsequent day:

python mpi_scripts/run_sim/submit_sim.py --config configs/allq_sim_oss20b.yaml --runs 1 \
  --set sim_name=allq_sim_oss20b_128k_a10_50_med \
  --set defaults.warmup_max_actions=10 \
  --set defaults.max_actions=50 \
  --set defaults.gptoss_reasoning_effort=medium \
  --set agent_max_model_len=131072

python mpi_scripts/run_sim/submit_sim.py --config configs/allq_sim_oss20b.yaml --runs 1 \
--set sim_name=allq_sim_oss20b_128k_a10_50_high \
--set defaults.warmup_max_actions=10 \
--set defaults.max_actions=50 \
--set defaults.gptoss_reasoning_effort=high \
--set agent_max_model_len=131072

python mpi_scripts/run_sim/submit_sim.py --config configs/allq_sim_oss120b.yaml --runs 1 \
  --set sim_name=allq_sim_oss120b_128k_a10_50_med \
  --set defaults.warmup_max_actions=10 \
  --set defaults.max_actions=50 \
  --set defaults.gptoss_reasoning_effort=medium \
  --set agent_max_model_len=131072

python mpi_scripts/run_sim/submit_sim.py --config configs/allq_sim_oss120b.yaml --runs 1 \
  --set sim_name=allq_sim_oss120b_128k_a10_50_high \
  --set defaults.warmup_max_actions=10 \
  --set defaults.max_actions=50 \
  --set defaults.gptoss_reasoning_effort=high \
  --set agent_max_model_len=131072