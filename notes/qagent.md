I want to create a allQAgent, which is just like basicAgent except on day 0 of the sim, before the first day begins, it has a first phase where it is given each question one by one in different context windows,  and has to make a prediction on each question using up to 5 search interactions (with max date till that day ofc). it doesnt need any instructions on memory, how the simulation works etc. it just has to make a forecasting prediction on each q which gets registered to the env. then, day 1 onwards it is the same as the basicAgent, excpet the prompt can mention that the agent has already made initial predictions on all qs at the day 0 date and now has to focus on updating any predictions it feels necessary. 

i am doing this to ensure that the number of questions a model predicts on stops being a confounder for my evaluations. create an implementation plan for this that keeps thigns as modular as possible, adding minimal code bloat while implementing the functionality and tries to clarify exactly those things the language model needs to maximize its forecasting performance

Implementation Plan - AllQAgent
This plan describes the implementation of AllQAgent, an agent that makes initial predictions on all active questions at the start of the simulation (Day 0) before entering the regular daily consistency.

Proposed Changes
1. New Agent: AllQAgent
Location: agents/allQAgent/agent.py

Create a new class AllQAgent inheriting from 
BasicAgent
.

warmup method:

Iterates through all active questions provided by the environment.
For each question:
Constructs a focused prompt (no memory/sim instructions, just forecasting task).
Executes a mini-loop (max 5 interactions, typically search + submit).
Submits prediction to the environment.
Sets a flag warmed_up = True upon completion.
_build_instructions
 override:

Calls super()._build_instructions().
If warmed_up is True, injects a reminder: "You have already made initial predictions on all questions. Focus on updating predictions where new information is available."
2. Environment Update: 
SimulationEnvironment
Location: 
environment/env.py

Add a warmup() method to 
SimulationEnvironment
.

warmup():
Gets all active questions from q_pool.
Iterates through registered agents.
Checks if agent has a warmup method.
If yes, creates a 
SimForecastInterface
 (initialized with current_date) and calls agent.warmup(active_questions, interface, current_date).
3. Script Update: 
test_basic_agent.py
Location: 
scripts/test_basic_agent.py

Support scaffold="allQ" (or allq) in 
create_agents_from_config
.
In 
main()
, call env.warmup() before env.run().
Detailed Logic
AllQAgent.warmup
def warmup(self, questions, forecast_interface, current_date):
    for q in questions:
        # 1. Setup specific prompt for this question
        # 2. Run interaction loop (max 5 actions)
        # 3. Submit prediction
Prompting
The warmup prompt will be stripped down: "You are a forecaster. Current date: {date}. Question: {q.title} ... Background: ... Criteria: ... Answer Type: ... Constraint: Use up to 5 actions. Search is available. Submit a prediction."

Verification Plan
Automated Tests
Run 
scripts/test_basic_agent.py
 with a config specifying scaffold: allQ.
Verify in logs that "Warmup" phase occurs before Day 1.
Verify predictions are recorded in actions.jsonl with timestamp of Day 0.
Verify efficient execution (agent shouldn't spend forever on random queries).
Manual Verification
Check model_outputs.jsonl to see the warmup prompts and responses.
Check day 1 prompt contains the inserted text.

Comment
⌥⌘M
