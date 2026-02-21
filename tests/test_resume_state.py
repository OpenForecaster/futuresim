from types import SimpleNamespace

from environment.env import SimulationEnvironment


def test_add_agent_preserves_restored_stats():
    """Adding agents after restore should not wipe reconstructed counters."""
    existing_aid = "agent_existing"
    new_aid = "agent_new"

    # Build a lightweight instance without running full __init__.
    env = SimulationEnvironment.__new__(SimulationEnvironment)
    env.agents = []
    env.agent_scores = {existing_aid: 12.5}
    env.agent_correct = {existing_aid: 7}
    env.agent_wrong = {existing_aid: 2}
    env.agent_questions = {existing_aid: 9}
    env.agent_raw_brier = {existing_aid: 4.2}
    env.agent_snapshot_peer = {existing_aid: 6.8}
    env.agent_exp_acc_sum = {existing_aid: 3.4}

    existing_agent = SimpleNamespace(agent_id=existing_aid)
    SimulationEnvironment.add_agent(env, existing_agent)

    # Existing restored values must remain unchanged.
    assert env.agent_scores[existing_aid] == 12.5
    assert env.agent_correct[existing_aid] == 7
    assert env.agent_wrong[existing_aid] == 2
    assert env.agent_questions[existing_aid] == 9
    assert env.agent_raw_brier[existing_aid] == 4.2
    assert env.agent_snapshot_peer[existing_aid] == 6.8
    assert env.agent_exp_acc_sum[existing_aid] == 3.4

    new_agent = SimpleNamespace(agent_id=new_aid)
    SimulationEnvironment.add_agent(env, new_agent)

    # New agents still get zero-initialized stats.
    assert env.agent_scores[new_aid] == 0.0
    assert env.agent_correct[new_aid] == 0
    assert env.agent_wrong[new_aid] == 0
    assert env.agent_questions[new_aid] == 0
    assert env.agent_raw_brier[new_aid] == 0.0
    assert env.agent_snapshot_peer[new_aid] == 0.0
    assert env.agent_exp_acc_sum[new_aid] == 0.0
