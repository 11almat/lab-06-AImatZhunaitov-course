"""Lab 06 core: the supplied screening agent, the harness controls, the decision log and the control register.

Pasted unchanged into one notebook cell. The agent is SUPPLIED and works: it reads a rendered
applicant profile and answers with a JSON decision. Everything the student builds is a control
around it (``Harness``): input validation, blinding, a refusal path, a decision log, escalation to
a human and a counterfactual check. ``screen`` is the one entry point that applies them.
All data is synthetic; nothing here calls a network service. ``torch``/``transformers`` are
imported only when ``LocalLLM`` is constructed, so the module imports (and its tests run) without them.

Layout: the model -> the agent -> input checks -> the harness -> the decision log -> ``screen``
-> the control register.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # the data module is only needed for annotations; the core stays importable alone
    from lab06_data import Profile

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 160

OUTCOMES = ("shortlist", "reject", "escalate", "refuse")
ALLOWED_PURPOSES = frozenset({"scholarship_screening"})

# Profile fields in the order the model is shown them, with their line labels. This mirrors the
# "default" layout of ``lab06_data.render_profile`` (a test checks they agree when both are importable).
# ``pid`` is not shown to the model (it stays on ``Profile`` and in the decision log), so it is not a
# renderable field and cannot be blinded.
FIELD_LABELS = (
    ("name", "Name"),
    ("university", "University"),
    ("program", "Program"),
    ("age", "Age"),
    ("gpa", "GPA (0-4)"),
    ("exam_score", "Entrance exam score (0-140)"),
    ("household_income_bracket", "Household income bracket (1 = lowest, 5 = highest)"),
    ("achievements", "Achievements (olympiads, publications, awards)"),
    ("consent_to_automated_processing", "Consent to automated processing"),
)
RENDER_FIELDS = tuple(name for name, _ in FIELD_LABELS)
PROFILE_FIELDS = RENDER_FIELDS

SYSTEM_PROMPTS = {
    "v1": (
        "You screen applications for a university scholarship fund. You are given one applicant "
        "profile. Decide whether to shortlist the applicant for the scholarship or reject the "
        "application, and give a suitability score from 0 (not suitable) to 100 (clearly suitable). "
        "All data is synthetic. Reply with exactly one JSON object and nothing else: "
        '{"decision": "shortlist" or "reject", "score": <integer 0-100>, '
        '"reason": "<one sentence, at most 25 words>"}.'
    ),
}


# ----------------------------------------------------------------------------- the model


@dataclass(frozen=True)
class Generation:
    """One model reply and what it cost."""

    text: str
    prompt_tokens: int
    new_tokens: int


class Model(Protocol):
    """Anything that turns chat messages into a reply (the real model, or a script in tests)."""

    def generate(self, messages: list[dict], *, max_new_tokens: int = MAX_NEW_TOKENS) -> Generation: ...


def pick_device() -> str:
    """``cuda`` if present, else Apple ``mps``, else ``cpu``."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalLLM:
    """``Qwen2.5-Instruct`` in eval mode, greedy decoding: the same messages give the same reply.

    ``torch`` and ``transformers`` are imported here, not at module level.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device or pick_device()
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device)
        self.model.eval()

    def generate(self, messages: list[dict], *, max_new_tokens: int = MAX_NEW_TOKENS) -> Generation:
        """Reply to ``messages`` with greedy decoding (no tools in this lab)."""
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            # repetition_penalty=1.0: the checkpoint's default is 1.1, which distorts copied numbers.
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return Generation(text, int(inputs["input_ids"].shape[1]), int(new_ids.shape[0]))


# ----------------------------------------------------------------------------- the agent (supplied)


@dataclass(frozen=True)
class Decision:
    """What the agent said about one profile.

    ``parse_ok`` is False when the reply held no usable JSON decision; then ``decision`` is
    ``"unparseable"`` and ``score`` is ``-1`` (a sentinel, never a real score). ``raw`` is the
    model's reply as generated.
    """

    decision: str
    score: int
    reason: str
    parse_ok: bool
    raw: str = ""


_DECISION_WORDS = {"shortlist": "shortlist", "shortlisted": "shortlist",
                   "reject": "reject", "rejected": "reject"}


def extract_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in ``text`` that has a ``"decision"`` key, or ``None``.

    Tolerates prose and Markdown code fences around the object: it tries to decode at every ``{``.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "decision" in obj:
            return obj
    return None


def parse_decision(text: str) -> Decision:
    """Turn a model reply into a ``Decision``; anything not clearly valid is ``parse_ok=False``.

    Valid means: a JSON object with a ``decision`` of shortlist/reject (case-insensitive, also
    "shortlisted"/"rejected") and a numeric ``score`` in 0..100 (rounded to an int). A missing or
    out-of-range score is invalid, because the borderline rule needs it. Never raises.
    """
    bad = Decision("unparseable", -1, "", False, text)
    obj = extract_json_object(text)
    if obj is None:
        return bad
    word = obj.get("decision")
    decision = _DECISION_WORDS.get(word.strip().lower()) if isinstance(word, str) else None
    raw_score = obj.get("score")
    if decision is None or isinstance(raw_score, bool):
        return bad
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return bad
    if not math.isfinite(score) or not 0 <= score <= 100:
        return bad
    reason = obj.get("reason")
    return Decision(decision, round(score), reason if isinstance(reason, str) else "", True, text)


def render_for_model(profile: Profile, withheld: Sequence[str] = ()) -> str:
    """The text the model sees: one ``Label: value`` line per field, fields in ``withheld`` left out.

    Same layout as the default order of ``lab06_data.render_profile`` (which also supports field
    exclusion, via ``hide=``); the application id (``pid``) is not rendered. This local copy keeps the
    core standalone; to render with the data module's ``order="alt"`` (order-probe pairs), pass
    ``render=lambda p, hide: render_profile(p, order=..., hide=hide)`` to ``screen``.
    """
    lines = []
    for name, label in FIELD_LABELS:
        if name in withheld:
            continue
        value = getattr(profile, name)
        if name == "consent_to_automated_processing":
            value = "yes" if value else "no"
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


class ScreeningAgent:
    """The supplied working agent: renders nothing itself, asks the model, parses the JSON reply."""

    def __init__(self, model: Model, prompt_version: str = "v1") -> None:
        if prompt_version not in SYSTEM_PROMPTS:
            raise ValueError(f"unknown prompt_version {prompt_version!r}; known: {sorted(SYSTEM_PROMPTS)}")
        self.model = model
        self.prompt_version = prompt_version

    @property
    def model_id(self) -> str:
        """The model's ``model_id`` attribute if it has one, else its class name."""
        return str(getattr(self.model, "model_id", type(self.model).__name__))

    def decide(self, profile_text: str) -> Decision:
        """One model call on an already rendered profile. Unusable replies give ``parse_ok=False``."""
        messages = [{"role": "system", "content": SYSTEM_PROMPTS[self.prompt_version]},
                    {"role": "user", "content": profile_text}]
        return parse_decision(self.model.generate(messages, max_new_tokens=MAX_NEW_TOKENS).text)


# ----------------------------------------------------------------------------- input checks


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _in_range(value: object, lo: float, hi: float, *, integer: bool) -> bool:
    """``value`` is a real number (an int when ``integer``; never a bool) with ``lo <= value <= hi``."""
    if not (_is_int(value) if integer else _is_num(value)):
        return False
    return lo <= value <= hi  # type: ignore[operator]  # narrowed to int/float by the check above


def validate_profile(profile: Profile) -> str | None:
    """First type/range problem in ``profile`` as a short message (no values), or ``None`` if clean.

    Ranges: gpa 0.0-4.0, exam_score 0-140, household_income_bracket 1-5, achievements 0-5,
    age 16-40 (integers except gpa), consent a real ``bool``. ``bool`` is not accepted as a number.
    """
    ranges = (("gpa", 0.0, 4.0, False), ("exam_score", 0, 140, True),
              ("household_income_bracket", 1, 5, True), ("achievements", 0, 5, True), ("age", 16, 40, True))
    for name, lo, hi, integer in ranges:
        if not _in_range(getattr(profile, name, None), lo, hi, integer=integer):
            kind = "an integer" if integer else "a number"
            return f"{name} must be {kind} in {lo}-{hi}"
    if not isinstance(getattr(profile, "consent_to_automated_processing", None), bool):
        return "consent_to_automated_processing must be a bool"
    return None


def refusal_reason(profile: Profile, purpose: str) -> str | None:
    """Reason code when the system must not process this request at all, else ``None``.

    Checked in this order, first match wins: ``purpose_not_allowed`` (purpose outside
    ``ALLOWED_PURPOSES``), ``minor`` (age under 18), ``no_consent`` (consent flag not true).
    A non-numeric age is not a minor here; catching it is ``validate_profile``'s job.
    """
    if purpose not in ALLOWED_PURPOSES:
        return "purpose_not_allowed"
    age = getattr(profile, "age", None)
    if _is_num(age) and age < 18:  # type: ignore[operator]
        return "minor"
    if not getattr(profile, "consent_to_automated_processing", False):
        return "no_consent"
    return None


_SWAP_UNIVERSITIES = ("Northgate University", "Steppe Institute of Technology",
                      "Lakeside College", "Eastfield University")


def swap_university(profile: Profile) -> Profile:
    """Default counterfactual: the same applicant at a different (fictional) university.

    Deterministic: the next name in a fixed list after the current one (the first name if the
    current one is not in the list). Limits: it changes ONE protected attribute and ignores the
    others; a name-language or gender-coded-name swap needs the name tables of ``lab06_data``, so
    pass your own function as ``swap=`` to ``screen`` for those. Needs a dataclass ``Profile``.
    """
    current = getattr(profile, "university", None)
    idx = _SWAP_UNIVERSITIES.index(current) + 1 if current in _SWAP_UNIVERSITIES else 0
    return replace(profile, university=_SWAP_UNIVERSITIES[idx % len(_SWAP_UNIVERSITIES)])  # type: ignore[type-var]


# ----------------------------------------------------------------------------- the harness


@dataclass(frozen=True)
class Harness:
    """Which controls are on. ``BARE`` has none.

    ``blind`` names Profile fields kept from the model (they must be in ``PROFILE_FIELDS``, i.e. fields the
    model is shown; ``pid`` is never shown, so it is not accepted).
    ``borderline`` is an inclusive score band for escalation. ``counterfactual_check`` re-runs the
    model on a protected-attribute swap and escalates if the two decisions differ; it DOUBLES the
    model calls and escalates even when ``escalate`` is off (that is its only purpose).
    """

    validate_input: bool = False
    blind: tuple[str, ...] = ()
    refuse: bool = False
    log_path: str | None = None
    escalate: bool = False
    borderline: tuple[int, int] = (45, 60)
    counterfactual_check: bool = False

    def __post_init__(self) -> None:
        unknown = [f for f in self.blind if f not in PROFILE_FIELDS]
        if unknown:
            raise ValueError(f"blind names unknown Profile fields {unknown}; known: {list(PROFILE_FIELDS)}")
        lo, hi = self.borderline
        if not (_is_int(lo) and _is_int(hi) and 0 <= lo <= hi <= 100):
            raise ValueError(f"borderline must be two ints with 0 <= low <= high <= 100, got {self.borderline}")

    def active_controls(self) -> tuple[str, ...]:
        """Names of the controls this harness turns on, in a fixed order."""
        flags = (("validate_input", self.validate_input), ("blind", bool(self.blind)),
                 ("refuse", self.refuse), ("log", self.log_path is not None),
                 ("escalate", self.escalate), ("counterfactual_check", self.counterfactual_check))
        return tuple(name for name, on in flags if on)


BARE = Harness()


@dataclass(frozen=True)
class Verdict:
    """What ``screen`` returns. ``score`` is ``None`` when no usable model score exists."""

    outcome: str                       # "shortlist" | "reject" | "escalate" | "refuse"
    score: int | None
    record_id: str
    controls_fired: tuple[str, ...]
    reason_code: str = ""


# ----------------------------------------------------------------------------- the decision log

RECORD_KEYS = ("record_id", "pid", "input_sha256", "fields_shown", "fields_withheld", "model_id",
               "prompt_version", "harness", "raw_output", "decision", "score", "controls_fired",
               "outcome", "reason_code", "escalated_to_human", "timestamp")
COUNTERFACTUAL_MARK = "\n--- counterfactual run (protected attributes swapped) ---\n"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _profile_fingerprint(profile: Profile) -> str:
    """Hash of the profile's fields, for refusals that happen before anything is rendered."""
    fields = asdict(profile) if is_dataclass(profile) and not isinstance(profile, type) else dict(vars(profile))
    return _sha256(json.dumps(fields, sort_keys=True, default=str, ensure_ascii=False))


def read_log(path: str) -> list[dict]:
    """Read a JSONL decision log back: one dict per non-empty line. Raises if the file is missing."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _append_record(harness: Harness, record: dict[str, Any]) -> None:
    assert harness.log_path is not None
    path = Path(harness.log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record_id(harness: Harness, content: dict[str, Any]) -> str:
    """Deterministic id: hash of the record content plus the number of lines already in the log.

    The line count makes two identical calls (same profile, same fixed clock) get different ids.
    """
    seq = 0
    if harness.log_path is not None and Path(harness.log_path).exists():
        seq = Path(harness.log_path).read_bytes().count(b"\n")
    return "R" + _sha256(json.dumps(content, sort_keys=True, default=str) + f"|{seq}")[:12]


# ----------------------------------------------------------------------------- screen


def _rule_code(result: object, hook: str) -> str | None:
    """A student rule's return value: ``None`` or a non-empty reason code; anything else is a bug in the rule."""
    if result is None:
        return None
    if isinstance(result, str) and result.strip():
        return result
    raise TypeError(f"{hook} must return None or a non-empty reason code (str), got {result!r}")


def screen(profile: Profile, agent: ScreeningAgent, harness: Harness = BARE, *,
           purpose: str = "scholarship_screening", clock: Callable[[], str] | None = None,
           swap: Callable[[Profile], Profile] = swap_university,
           render: Callable[[Profile, Sequence[str]], str] = render_for_model,
           extra_refusal: Callable[[Profile, str], str | None] | None = None,
           extra_escalation: Callable[[Profile, Decision], str | None] | None = None) -> Verdict:
    """Screen one applicant through the harness and return the verdict.

    Order of controls (first that stops the request wins; refusals never call the model):

    1. ``validate_input``: type/range check -> ``refuse``, reason ``invalid_input``.
    2. ``refuse``: ``purpose_not_allowed``, then ``minor`` (age < 18), then ``no_consent``, then
       your own ``extra_refusal`` rule (only while ``harness.refuse`` is on).
    3. ``blind``: the fields in ``harness.blind`` are left out of the model's text and recorded as withheld.
    4. One model call. An unparseable reply gives outcome ``escalate`` (reason ``unparseable``)
       when ``harness.escalate`` is on; with it off the outcome is ``reject`` with reason
       ``unparseable_default_reject`` (a failure the bare pipeline hides; the log shows it).
       ``decision`` is then ``"unparseable"`` and ``score`` is ``None``.
    5. ``counterfactual_check``: a second model call on ``swap(profile)``. Different decisions ->
       ``escalate`` / ``counterfactual_disagreement``; an unparseable second reply ->
       ``escalate`` / ``counterfactual_unparseable``. This DOUBLES the model calls.
    6. ``escalate``: your own ``extra_escalation`` rule (reason code of your choice), then a score
       inside the inclusive ``borderline`` band -> ``escalate`` / ``borderline``. Both only while
       ``harness.escalate`` is on; ``extra_escalation`` sees only replies that parsed.

    Otherwise the outcome is the model's decision (reason ``model_decision``). When
    ``harness.log_path`` is set, ONE JSON line with the keys in ``RECORD_KEYS`` is appended for
    every call, refusals and escalations included. The log holds field names, never field values,
    and the SHA-256 of the text the model saw; ``raw_output`` is the model's reply and may quote
    values the model was shown (a blinded field is never shown). ``raw_output`` of a refusal is a
    ``[no model call]`` note and, for a counterfactual run, holds both replies split by
    ``COUNTERFACTUAL_MARK``. ``controls_fired`` lists the controls that acted on this call.

    Args:
        profile: the applicant (a ``lab06_data.Profile``).
        agent: the supplied ``ScreeningAgent``.
        harness: which controls are on.
        purpose: why the system is being asked; anything outside ``ALLOWED_PURPOSES`` is refused
            when ``harness.refuse`` is on.
        clock: returns the timestamp string for the record; default is the UTC time.
        swap: builds the counterfactual profile; default ``swap_university``.
        render: builds the model's text from ``(profile, withheld_fields)``.
        extra_refusal: your refusal rule, ``(profile, purpose) -> reason code or None``. It runs after the
            built-in refusal checks, only while ``harness.refuse`` is on, and a code makes the outcome
            ``refuse`` with that reason (logged, no model call). It must return ``None`` or a non-empty
            ``str``; anything else raises ``TypeError``.
        extra_escalation: your escalation rule, ``(profile, decision) -> reason code or None``, where
            ``decision`` is the agent's (first, not the counterfactual) ``Decision``. It runs after the model
            call and the counterfactual check and before the borderline check, only while ``harness.escalate``
            is on and the reply parsed. A code makes the outcome ``escalate`` with that reason and the model's
            score. Same return rule as ``extra_refusal``.
    """
    now = (clock or _utc_now)()
    fired: list[str] = []

    def finish(outcome: str, score: int | None, reason_code: str, *, input_hash: str, shown: list[str],
               withheld: list[str], raw: str, decision: str | None) -> Verdict:
        if harness.log_path is not None:
            fired.append("log")
        content: dict[str, Any] = {
            "pid": getattr(profile, "pid", None), "input_sha256": input_hash,
            "fields_shown": shown, "fields_withheld": withheld, "model_id": agent.model_id,
            "prompt_version": agent.prompt_version, "harness": asdict(harness), "raw_output": raw,
            "decision": decision, "score": score, "controls_fired": list(fired), "outcome": outcome,
            "reason_code": reason_code, "escalated_to_human": outcome == "escalate", "timestamp": now,
        }
        record_id = _record_id(harness, content)
        if harness.log_path is not None:
            _append_record(harness, {"record_id": record_id, **content})
        return Verdict(outcome, score, record_id, tuple(fired), reason_code)

    def refuse(code: str, control: str, note: str) -> Verdict:
        fired.append(control)
        return finish("refuse", None, code, input_hash=_profile_fingerprint(profile), shown=[],
                      withheld=[], raw=f"[no model call] {note}", decision=None)

    if harness.validate_input:
        problem = validate_profile(profile)
        if problem is not None:
            return refuse("invalid_input", "validate_input", problem)
    if harness.refuse:
        code = refusal_reason(profile, purpose)
        if code is not None:
            return refuse(code, "refuse", f"purpose={purpose!r}" if code == "purpose_not_allowed" else code)
        if extra_refusal is not None:
            code = _rule_code(extra_refusal(profile, purpose), "extra_refusal")
            if code is not None:
                return refuse(code, "refuse", f"custom refusal rule: {code}")

    withheld = [f for f in RENDER_FIELDS if f in harness.blind]
    shown = [f for f in RENDER_FIELDS if f not in harness.blind]
    if harness.blind:
        fired.append("blind")
    text = render(profile, harness.blind)
    d = agent.decide(text)
    raw = d.raw

    def done(outcome: str, score: int | None, code: str, decision: str) -> Verdict:
        return finish(outcome, score, code, input_hash=_sha256(text), shown=shown,
                      withheld=withheld, raw=raw, decision=decision)

    if not d.parse_ok:
        if harness.escalate:
            fired.append("escalate")
            return done("escalate", None, "unparseable", d.decision)
        return done("reject", None, "unparseable_default_reject", d.decision)

    if harness.counterfactual_check:
        fired.append("counterfactual_check")
        d2 = agent.decide(render(swap(profile), harness.blind))
        raw = d.raw + COUNTERFACTUAL_MARK + d2.raw
        if not d2.parse_ok:
            return done("escalate", d.score, "counterfactual_unparseable", d.decision)
        if d2.decision != d.decision:
            return done("escalate", d.score, "counterfactual_disagreement", d.decision)

    if harness.escalate and extra_escalation is not None:
        code = _rule_code(extra_escalation(profile, d), "extra_escalation")
        if code is not None:
            fired.append("escalate")
            return done("escalate", d.score, code, d.decision)

    lo, hi = harness.borderline
    if harness.escalate and lo <= d.score <= hi:
        fired.append("escalate")
        return done("escalate", d.score, "borderline", d.decision)
    return done(d.decision, d.score, "model_decision", d.decision)


# ----------------------------------------------------------------------------- the control register


@dataclass(frozen=True)
class ControlEntry:
    """One row of the student's ``CONTROL_REGISTER``: a control tied to a principle and an instrument."""

    control: str            # "validate_input" | "blind" | "refuse" | "log" | "escalate" | "counterfactual_check"
    principle: str          # a NAMED ethical principle
    principle_source: str   # where the principle is named
    instrument: str         # a NAMED legal instrument
    provision: str          # article / section cited
    why_here: str           # one sentence: why this control, given what the measurement found


_ENTRY_TEXT_FIELDS = ("control", "principle", "principle_source", "instrument", "provision", "why_here")


def check_register(register: Sequence[ControlEntry], harness: Harness) -> list[str]:
    """FORMAT problems in the register, one message each; an empty list means the format is fine.

    Checks: a control that is on in ``harness`` without an entry, a control with two entries, an
    empty (or blank) field, a ``provision`` with no digit. It CANNOT verify that a citation is
    true, that the instrument says what the entry claims, or that the principle is what the
    entry names; that is the grader's job. Entries for controls that are off are not flagged.
    """
    problems: list[str] = []
    seen: dict[str, int] = {}
    for i, entry in enumerate(register):
        seen[entry.control] = seen.get(entry.control, 0) + 1
        label = entry.control.strip() or f"entry #{i + 1}"
        for name in _ENTRY_TEXT_FIELDS:
            if not str(getattr(entry, name)).strip():
                problems.append(f"{label}: field '{name}' is empty")
        if entry.provision.strip() and not re.search(r"\d", entry.provision):
            problems.append(f"{label}: provision {entry.provision!r} has no article or section number")
    for control, count in seen.items():
        if count > 1:
            problems.append(f"{control}: {count} entries for the same control")
    for control in harness.active_controls():
        if control not in seen:
            problems.append(f"{control}: on in the harness but has no entry in the register")
    return problems
