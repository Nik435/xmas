"""Group C: the Orchestrator. Optional (a stage-2 factor); the hub of the system when active.

After every agent's turn the orchestrator reads what happened and decides what happens next:
which agent works, with what instruction, or that the pipeline is submitted. The loop that
runs this is :mod:`tabular_preprocessing.inner_loop.orchestrated`.
"""

from tabular_preprocessing.inner_loop.agents.base import AgentType

ORCHESTRATOR = AgentType(
    key="orchestrator",
    title="Orchestrator",
    group="orchestrator",
    role="C",
    task="delegate the work to the agents, manage the budget, decide when to stop and submit the pipeline",
    tool_sets=("orchestration", "validation"),
    instructions=(
        "You lead the team: after every agent's turn you decide who works next. Read the agents' latest "
        "answers below and the decisions so far. orchestrator_status shows the draft pipeline, who changed "
        "what and the budget; orchestrator_check_pipeline, orchestrator_score_pipeline and "
        "orchestrator_feature_report measure the draft (scores use the evaluation budget). A good order is "
        "usually plan, then execute, then validate, for one group and then the other; send an agent back with "
        "a concrete instruction when its work must change. Every turn costs budget: submit when further turns "
        "are unlikely to improve the validation score, or when agents repeat themselves."
    ),
    answer=(
        'End your answer with exactly one JSON object: {"next": "<agent name>", "instruction": "<what it should '
        'do now>"} to delegate, or {"next": "submit", "summary": "<why this pipeline>"} to finish.'
    ),
)
