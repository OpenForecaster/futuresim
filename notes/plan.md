## Context

- We are now building on top of our first released paper notes/v1_paper.pdf which has associated blog which might be easier to read for you, notes/v1_blog.md. 

- This is a research project. I will eventually train a forecasting agent in the environment we setup, so I want to make design decisions such that when the agent optimizes for this environment, the behavior it learns generalizes to good real-world predictions, and potentially profit on prediction markets.

- You can write any salient memories, things you'd want to remember in future contexts in notes/memories.md, but don't write about your new changes or recommendations, only past state of the repo, as I might reject your proposed changes. 

In the previous version, the forecaster had to do single-turn QA, it was provided a forecasting question, optionally some retrieved context, and then output a prediction and probability. We then measured progress in brier score and accuracy with RL training on large-scale forecasting question answer pairs created from news articles.

I now want to explore a more realistic simulation. We have an overall set of documents stored in <overall_doc_dir>. Each day, documents with that publish date get added to <visible_doc_dir>, the documents available to a forecasting agent in the sim. There is a <doc_interface> through which agents can explore documents. We have to decide its design later, but this is how an agent can choose which articles to "Read", and navigate the large number of documents in <visible_doc_dir>, which won't fit in context at once.

The other structural element I want is <forecast_question_pool>. This is a pool of forecasting questions with resolution dates, final outcomes, and optionally market aggregate history if its from a prediction market. Agents can interact with the <current_question_pool> using <forecast_interface> but the final outcome should always be hidden from them for questions whose resolution date has not passed. Instead, agents can see a "current aggregate", which will either be fetched from prediction markets for questions from there, or computed as aggregate across all agents that have registered a prediction on the question.

Each day, after interacting with the new documents that have emerged, agents can update their existing predictions, or make new predictions, on a subset of questions they choose, through the <forecast_interface>.

<overall_doc_dir> = data/overall_sim/context
<visible_doc_dir> = data/current_sim/context
<forecast_question_pool> = data/overall_sim/market
<current_question_pool> = data/current_sim/market

All the main environment logic should be in environment/

Some questions:

Ideally we want to start with a single-agent acting in the environment for simplicity. But then what is the incentive for the agent to update its predictions on a question, and not just directly predict one day before the resolution date?

If I do generative (non-binary, non multiple choice) forecasting where the agent gives a short text answer and probability, or a list of possible outcomes and a distribution of probability over them, how do I aggregate these?

##