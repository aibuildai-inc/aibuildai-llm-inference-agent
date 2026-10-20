"""Input and Output records for the agentic package's own Worker role."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from pydantic import ConfigDict, Field, ValidationInfo, model_validator

from engine.base import WorkflowBaseModel
from engine.execution_output import SuccessfulOutput
from engine.work_unit.agent.base import AgentInput
from engine.builtin.agentic.agents.designer.io import Citation, WorkerDesignPlan
from engine.builtin.aibuildai.agents.submitter.io import ExternalScore


class RecipeChange(WorkflowBaseModel):
    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "A narrative edit to one component of the revised "
            "design's design plan."
        },
    )
    target: str = Field(
        description="Which area is being changed. Use one of: 'architecture', "
        "'data_pipeline', 'loss_function', 'regularization', 'training_config', "
        "'validation_strategy'."
    )
    from_state: str = Field(
        description="One sentence describing the revised design's current state for "
        "this target (your understanding of what's there now)."
    )
    to_state: str = Field(
        description="One sentence describing the new state — concrete and "
        "actionable, the recipient uses this as-is."
    )
    rationale: str = Field(
        description="Why this change addresses the diagnosis. Must tie to evidence "
        "in `diagnosis`."
    )


class CodePatch(WorkflowBaseModel):
    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": inspect.cleandoc(
                """An exact find/replace patch applied to inherited design source.

        Use this when the failure is code-shaped (NaN logits, single-class
        collapse, off-by-one indexing, AMP misuse) rather than recipe-shaped.
        The recipient applies these verbatim BEFORE applying recipe_changes."""
            )
        },
    )
    file: str = Field(
        description="Path relative to the recipient's `owner_dir` (e.g. 'run.py')."
    )
    find: str = Field(
        description="EXACT substring to locate in the file. Multi-line is fine; "
        "whitespace must match. If the recipient cannot find this string, the "
        "patch FAILS and execution stops — do not paraphrase or guess."
    )
    replace: str = Field(
        description="Replacement substring. Use empty string to delete the `find` "
        "block."
    )
    rationale: str = Field(
        description="One sentence: what bug this patch fixes and how you identified "
        "it."
    )


class WorkerRevision(WorkflowBaseModel):
    """One direction the current Worker proposes for a child Worker.

    AgenticSearch always forces zero revisions (its Manager has no tool that
    continues one), so this shape never actually reaches a child Worker in
    this package; it stays here only because it is part of the Worker role's
    own submission contract -- ``WorkerOutput`` still owes the framework an
    empty ``revisions`` list every time."""

    model_config = ConfigDict(frozen=True)

    name: str
    diagnosis: str
    revision_intent: str
    recipe_changes: tuple[RecipeChange, ...]
    code_patches: tuple[CodePatch, ...]
    expected_outcome: str
    requested_gpus: int = Field(ge=0)
    references: tuple[Citation, ...]


@dataclass(frozen=True)
class WorkerInput(AgentInput):
    """One complete design and the optional revision it applies."""

    plan: WorkerDesignPlan
    revision: WorkerRevision | None
    # The design this one continues, or None for a fresh plan. Its
    # directory holds the source, the logs and the output that ran.
    parent_path: str | None
    parent_directory: str | None
    parent_metric: float | None
    task_name: str
    min_revisions: int
    num_revisions: int | None
    gpu_count: int
    progress_enabled: bool
    # Advisory external results recorded so far, newest last. Empty when sharing
    # is off, when no submission has been made yet, or when none has returned --
    # three states the prompt must not conflate.
    external_scores: tuple[ExternalScore, ...]


class WorkerOutput(SuccessfulOutput):
    revisions: tuple[WorkerRevision, ...]
    feedback: str

    @model_validator(mode="after")
    def _revision_count_fits_the_request(self, info: ValidationInfo) -> "WorkerOutput":
        """Refuse a submission that did not do the revisions it was asked for.

        The count is a property of this answer against the request that asked
        for it, so nothing outside the two is needed to decide it.

        A persistence restore re-reads an Output the run already accepted and
        carries no input, so it stops at the context check."""
        if info.context is None:
            return self
        worker_input = info.context.get("input")
        if not isinstance(worker_input, WorkerInput):
            raise ValueError("the Worker request is missing from validation context")
        count = len(self.revisions)
        minimum = worker_input.min_revisions
        maximum = worker_input.num_revisions
        if count >= minimum and (maximum is None or count <= maximum):
            return self
        if maximum == minimum:
            requirement = f"exactly {minimum} revisions"
        elif maximum is None:
            requirement = f"at least {minimum} revisions"
        else:
            requirement = f"between {minimum} and {maximum} revisions"
        raise ValueError(
            f"This Worker requires {requirement}; you submitted {count}."
        )
