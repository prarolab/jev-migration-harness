from datetime import date
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Identifier = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")]
Probability = Annotated[float, Field(ge=0, le=1)]


class Evidence(StrictModel):
    id: Identifier
    source: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    observed_at: date
    content: str = Field(min_length=1)


class Question(StrictModel):
    id: Identifier
    text: str = Field(min_length=1)
    rubric: str = Field(min_length=1)
    kind: Literal["choice", "narrative"]
    choices: dict[str, str] = Field(default_factory=dict)
    evidence_ids: list[Identifier]
    requires_owner: bool = False
    applicable: bool = True
    applicability_reason: str | None = None
    max_age_days: int = Field(default=90, ge=0, le=3650)

    @model_validator(mode="after")
    def valid_choices(self) -> Self:
        if self.kind == "choice" and not 2 <= len(self.choices) <= 32:
            raise ValueError("Choice questions require 2-32 explicitly defined options.")
        if self.kind == "narrative" and self.choices:
            raise ValueError("Narrative questions cannot define choices.")
        if any(not key or key.startswith("__") for key in self.choices):
            raise ValueError("Empty and '__'-prefixed choices are reserved.")
        if not self.applicable and not self.applicability_reason:
            raise ValueError("Non-applicability requires an explicit reason.")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Duplicate evidence reference.")
        return self


class AssessmentInput(StrictModel):
    application: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    as_of: date
    evidence: list[Evidence]
    questions: list[Question] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def valid_references(self) -> Self:
        ids = [item.id for item in self.evidence]
        question_ids = [question.id for question in self.questions]
        if len(ids) != len(set(ids)) or len(question_ids) != len(set(question_ids)):
            raise ValueError("Evidence and question IDs must be unique within their lists.")
        for question in self.questions:
            if set(question.evidence_ids) - set(ids):
                raise ValueError(f"{question.id}: unknown evidence reference.")
        return self


class ChoiceAnswer(StrictModel):
    type: Literal["choice"]
    choice: str
    confidence: Probability
    probabilities: dict[str, Probability]

    @model_validator(mode="after")
    def valid_distribution(self) -> Self:
        if self.choice not in self.probabilities:
            raise ValueError("Chosen option is absent from probability distribution.")
        if abs(sum(self.probabilities.values()) - 1.0) > 0.01:
            raise ValueError("Probabilities do not sum to one.")
        if self.probabilities[self.choice] < max(self.probabilities.values()) - 0.001:
            raise ValueError("Choice is not a highest-probability option.")
        return self


class Usage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class JevResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    answers: dict[str, ChoiceAnswer]
    usage: Usage


class Citation(StrictModel):
    evidence_id: str
    quote: str = Field(min_length=1)


class LLMAnswer(StrictModel):
    question_id: str
    status: Literal["supported", "insufficient", "conflicting"]
    answer: str | None
    citations: list[Citation]
    missing_evidence: list[str]

    @model_validator(mode="after")
    def consistent_answer(self) -> Self:
        if self.status == "supported":
            if not self.answer or not self.answer.strip() or not self.citations:
                raise ValueError("A supported answer requires text and citations.")
            if self.missing_evidence:
                raise ValueError("A supported answer cannot have missing required evidence.")
        elif self.answer is not None:
            raise ValueError("Unresolved or conflicting evidence cannot produce an answer.")
        if self.status == "insufficient" and not self.missing_evidence:
            raise ValueError("Insufficient answers must identify missing evidence.")
        return self
