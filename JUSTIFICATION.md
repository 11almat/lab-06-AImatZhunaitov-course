# JUSTIFICATION — the design of the harness, and why

Authors: Zhuman Arnur, Kuralbek Atabek · Lab 06, Harness and Reliable AI Systems · Narxoz University

All numbers below are copied from our submitted run of `lab06_harness_says_no.ipynb`
(Qwen2.5-3B-Instruct, greedy decoding, `N_PAIRS = 40`, `SEED = 2026`, 200 pairs = 400 profiles).

## 1. What the system may do

**Purpose, stated narrowly.** The system pre-screens applications to the (fictional) Tulpar Scholarship Fund.
It may (a) refuse to process a request it must not process, (b) recommend *shortlist* or *reject* with a
0–100 score for a clean, consenting, adult applicant whose score lies outside the escalation band and whose
case triggers none of the escalation rules, and (c) send every other case to a human reviewer. It may not
make a final award, rank applicants against each other, or be used for any purpose other than
`scholarship_screening` (`ALLOWED_PURPOSES`). Even its "shortlist/reject" outcomes are recommendations that
the fund's staff confirm; under Law No. 94-V Art. 19-1(2) the applicant must be told that automated
processing was used and may object (see §4).

**Controls, the constraint behind each, and where they are in the code.**

| control | constraint that made us build it | where in the code |
|---|---|---|
| `validate_input` | safety/reliability duty of an AI-system owner (AI Law Art. 15(2) 1–2); Part 4: the bare pipeline scored GPA 9.9 / exam −5 | `Harness(validate_input=True)` → `validate_profile` in `lab06_core.py` |
| `blind = ("name", "university")` | non-discrimination (Constitution 2026 Art. 16(2); AI Law Art. 6) and data minimisation (Law 94-V Art. 7(9)); Part 1 measurement | cell 2a `BLIND`; applied in `screen` step 3, logged as `fields_withheld` |
| `refuse` (+ our `my_extra_refusal`) | purpose limitation and consent for automated decisions (Law 94-V Art. 7(8), Art. 19-1(1)); prohibited AI functions (AI Law Art. 17(3)); robustness against inputs addressed to the model | `refusal_reason` + cell 2b `my_extra_refusal` |
| `log` | documentation, explanation on request and audit (AI Law Art. 15(2) 3, Art. 16(1) 4, Art. 20) | `Harness(log_path="decisions.jsonl")` → `decisions.jsonl` |
| `escalate` (+ band + `my_extra_escalation`) | human control and the right to object (AI Law Art. 7(2), Art. 8(1); Law 94-V Art. 19-1(2)–(3)) | `Harness(escalate=True, borderline=BORDERLINE)`, cell 2c, cell 2d `calibrate_band` |

**A constraint that changed the architecture.** Without Art. 19-1 of Law 94-V (in force since January 2026),
the natural design is "the model decides, a human audits a sample later". Art. 19-1(1) forbids automated
processing that creates, changes or ends a person's rights or legitimate interests unless the person consented
or a law provides for it, so consent became a **gate before the model call** (refusal `no_consent`), not a
field the model reads. Likewise, because the model's own decision on an uncertain case cannot be explained
from the record, unparseable replies are **escalated**, not defaulted to `reject` as in the bare pipeline
(0 of 400 replies in Part 1 were unparseable and silently became rejections). A second change: the band was
calibrated on a separate calibration set screened **blinded**, because a band set on the unblinded Part 1 scores
does not transfer to the configuration that is actually deployed (§6).

## 2. What it must refuse

| refusal | enforced in | fired in our run | legitimate case it also blocks | who bears it |
|---|---|---|---|---|
| `purpose_not_allowed` (anything but `scholarship_screening`) | `refusal_reason` | Part 4, "rank applicants by ethnicity" | a legitimate new use (e.g. statistics for the fund's annual report) until it is added on purpose | the fund (it must re-approve the use) |
| `minor` (age < 18) | `refusal_reason` | Part 4, age 17 | a gifted 17-year-old first-year student who is eligible | the applicant: no automated decision, a human route only |
| `no_consent` | `refusal_reason` | Part 4 | an applicant who simply missed the checkbox | the applicant (delay) and staff (manual handling) |
| `invalid_input` | `validate_profile` | Part 4, GPA 9.9 / exam −5 | a data-entry error by the fund's own staff | applicant (delay) |
| `instruction_in_<field>`, `unexpected_characters_in_<field>`, `oversized_<field>`, `missing_<field>` (our rule) | cell 2b `my_extra_refusal` | Part 4, name = "Ignore all previous instructions…" → `instruction_in_name` | a real name with an initial ("A."), a digit, a hyphenated or very long name, or a family name equal to a keyword | the applicant: refused although honest |

Our rule refused 0 of the 400 clean applicants in the measured pairs (the last line of Part 4); with
invented names that is expected to be 0 and **does not prove** the rule is safe on real names, which may contain
punctuation or Cyrillic/Kazakh letters (the rule accepts any Unicode letter, space, hyphen and apostrophe for
this reason). We accept the cost because a refusal is not a rejection: every refusal is logged with its reason
code and **no model call**, and in deployment it goes to staff, who can correct the form or decide by hand. A
refused applicant can always ask for a human decision (Law 94-V Art. 19-1(2)–(3): objection answered within
three working days). What the harness does **not** yet do is notify the applicant automatically; that is an
institutional process (§6, gaps).

The injected name would in any case be blinded, so the model would never see it. We still refuse: a tampered
form is evidence about the request, not about the applicant, and the blinding list could change in a later
version (defence in depth).

## 3. What record it keeps

**Audit question.** *"Why did this application get this outcome, which controls acted on it, and can someone
else verify what the model was shown, without re-running the model?"*

A record in `decisions.jsonl` answers it: `pid`, `input_sha256` (hash of the exact text the model saw; for a
refusal, a fingerprint of the profile), `fields_shown` / `fields_withheld`, `model_id`, `prompt_version`, the
full `harness` settings (including the band), the model's `raw_output`, `decision`, `score`, `controls_fired`,
`outcome`, `reason_code`, `escalated_to_human`, `timestamp`, and a deterministic `record_id`. Our run wrote
406 records: 400 from Part 3 and 6 from Part 4 (by outcome: {'escalate': 115, 'reject': 24, 'shortlist': 262, 'refuse': 5}; by reason code: {'borderline': 115, 'model_decision': 286, 'purpose_not_allowed': 1, 'minor': 1, 'no_consent': 1, 'invalid_input': 1, 'instruction_in_name': 1}).

**Left out on purpose:** the field *values* (name, university, grades). A reviewer who holds the application can
re-render it and compare the hash; the log itself is not a second copy of applicants' personal data (Law 94-V
Art. 7(9), minimisation). `raw_output` may quote values the model was shown; a blinded field is never shown, so
it never appears there.

**Retention, access, integrity (design decisions, not implemented in code).** Keep records as long as the
purpose lasts: the scholarship round plus the appeal period (Law 94-V Art. 12(2): retention is set by the date
the purposes are achieved); we propose one funding cycle plus 12 months, and at least the six months the
EU AI Act Art. 26(6) uses as a benchmark. Read access: the fund's review panel and its data-protection officer;
nobody edits. The current file is append-only by convention only: nothing in the code stops a later edit. A
hash chain (each record carrying the hash of the previous one) or write-once storage would close that gap.
**What the record cannot reconstruct:** *why* the model produced its score (only what it was shown and what it
said), and what a human reviewer later decided about an escalated case (the reviewer's decision is not logged).

## 4. The legal constraints

**Jurisdiction.** The fund is fictional, but placed where this course is taught: a fund operating in Kazakhstan,
processing personal data of applicants in Kazakhstan with an AI system it operates. We therefore treat Kazakh law
as **binding** and EU law as a **benchmark** only.

1. **Constitution of the Republic of Kazakhstan** (adopted by referendum on 15 March 2026, in force since
   1 July 2026), **Art. 16(1)–(2)**: equality before the law; no one may be discriminated against on grounds of
   origin, social or property status, sex, race, nationality, language, religion and other circumstances.
   *Requires* equal treatment; it is addressed to the State and to law, so its direct effect on a private fund is
   arguable; we use it as the ground the statutory non-discrimination rules rest on. → `blind`.
2. **Law of the RK No. 230-VIII of 17 November 2025 "On Artificial Intelligence"** (in force 60 days after
   publication, Art. 31). Art. 6 (fairness and equality, excluding any discrimination); Art. 7(2) (a person
   subject to AI-based decisions may be informed of the automated processing and its consequences and may
   object); Art. 8(1) (owners/holders keep constant control over the system); Art. 15(2) 1–3 (manage risks,
   ensure safety and reliability, keep documentation); Art. 16(1) 4 (users may request an explanation of results
   affecting their rights); Art. 17(3) (prohibited AI functions, including classifying people by biometric data
   to discriminate and processing personal data in breach of the law); Art. 20 (audit). *Requires* risk
   management, documentation and human control; it **does not** prescribe logs, bands or blinding: those are our
   choices for meeting it.
3. **Law of the RK No. 94-V of 21 May 2013 "On Personal Data and Their Protection"**, as amended, in particular by
   Law No. 231-VIII of 17 November 2025 which added **Art. 19-1**. Art. 7(1), 8(1) (consent of the subject or
   legal representative); Art. 7(8) (processing limited to specific, predefined, lawful purposes); Art. 7(9) (no
   data excessive for the purpose); Art. 12(2) (retention ends when the purposes are achieved); **Art. 19-1(1)**
   (automated processing that creates, changes or ends rights or legitimate interests is prohibited unless the
   subject consented or a law provides for it), **19-1(2)** (explain the processing, its consequences and the
   right to object), **19-1(3)** (answer an objection within three working days). This is the one provision we
   found that is stated to apply to **automated decisions specifically**. → `refuse` (no consent), `escalate`.
4. **Benchmarks (not binding on this fund):** Regulation (EU) 2024/1689 (AI Act), which applies to providers
   and deployers in the EU or whose output is used in the EU (Art. 2(1)); a fund in Kazakhstan whose decisions are
   used in Kazakhstan is outside it. We use: Annex III point 3(a) (AI used to determine access or admission to
   education is high-risk; a scholarship that decides whether a student can study is close to it, which is our
   reading); Art. 10(2)(f) (examination for possible biases); Art. 12(1) (automatic logging); Art. 14(1), 14(4)(d)
   (human oversight, ability to disregard the output); Art. 26(6) (deployers keep logs ≥ 6 months).
   Regulation (EU) 2016/679 (GDPR) Art. 22(1) and 22(3) (no decision based solely on automated processing with
   significant effects; right to human intervention).

**Where we went beyond the law.** Kazakh law requires consent or a legal basis for automated processing and the
right to object; it does not require blinding, a measured bias test, an escalation band or an
injection-resistant input filter. We built them because the ethical principles (§5) and the measurement asked
for them. **Where we found no provision:** for refusing minors. Law 94-V has no age-specific rule (it speaks only
of a "legal representative"), so the `minor` refusal is a design choice that routes the case to a human who can
involve the legal representative. We do not claim a law requires it.

## 5. The ethical principles

| principle | source and status | decision in this system it bears on |
|---|---|---|
| Fairness and non-discrimination | UNESCO Recommendation on the Ethics of AI (23 Nov 2021), sec. III.2; OECD AI Principle 1.2 (OECD/LEGAL/0449, 2019, amended 2024). Both non-binding recommendations to states | hiding name and university; measuring flips against floors |
| Right to privacy and data protection | UNESCO Rec. sec. III.2 | logging names of fields, not values; consent gate |
| Human oversight and determination | UNESCO Rec. sec. III.2 | escalation band, unparseable → person, refusal routes to staff |
| Transparency and explainability | OECD 1.3; UNESCO Rec. sec. III.2 | `reason_code` on every record; escalation when the reason cites a protected attribute or contradicts the score |
| Robustness, security and safety | OECD 1.4 | input validation; refusal of instruction-bearing fields |
| Accountability | OECD 1.5 (traceability); UNESCO Rec. sec. III.2 "Responsibility and accountability" | the decision log, the register |

**Conflicts.** *Identical treatment vs. fair treatment:* blinding treats twins identically by construction, but
it also hides context that could justify favouring a disadvantaged applicant (a regional college can signal less
opportunity, not less merit). We put identical treatment first for the automated step and left need-based
judgement to humans through the `high_need_high_merit_rejected` escalation, because the model's use of the
university is unmeasured in direction for real applicants. *Oversight vs. privacy:* more escalation means more
people read the file. *Robustness vs. inclusion:* the injection filter can refuse unusual real names.

**Where ethics asks more than law.** Kazakh law does not require us to *measure* whether the system treats
matched applicants alike; the fairness principle does, and so does professional responsibility (see REPORT §3).

## 6. The trade-offs

| control | cost (from our run) | harm it prevents |
|---|---|---|
| validate_input | 0 model calls; 0 wrong refusals on clean data | scoring impossible profiles (Part 4) |
| blind | the decision of 122 of 400 profiles and the score of 166 of 400 changed vs. the unblinded run (mean abs. change 14.01); loss of the university as a signal | the name/university channel measured in Part 1 (name_language 2/40 vs null_name 0/40; gender_coded_name 4/40 vs null_name 0/40; university 1/40 vs null_university 1/40) |
| refuse | 0 calls; human workload for refused cases | forbidden purpose, minors, no consent, injected instruction |
| log | storage; a second record of decisions to protect | silent failures (0 unparseable → rejected in the bare run) |
| escalate | 114 of 400 profiles (28.5%) sent to a person, against ~15 % planned | uncertain and contestable calls decided by a model alone |
| calibration of the band | 80 extra model calls, not logged | a band that silently sends far more or fewer people to review than planned |

**The band.** Rule: the narrowest band centred on the model's decision boundary (midway between the median scores
of rejected and shortlisted profiles) that holds ≥ 15 % of scored profiles, an assumed reviewer capacity. Set on
the bare Part 1 scores it was (55, 60); recalibrated on 80 other applicants screened blinded it became
(35, 60), which is the band used. Applied in the harness run it sent 114 of 400 (28.5%) to a person once
unparseable replies and our own rules are added (by reason: borderline 114). Outside the band
286 decisions go unchecked: a confident wrong rejection is never read by anyone. We would widen the band if
the fund has more reviewers, or if the counterfactual extension (0 of 16 profiles escalated by disagreement)
showed that name-driven flips occur far from the boundary.

**What the design still fails to protect against.**
- **Proxies in the shown fields** (programme, age, income bracket) and bias in the merit data itself: blinding
  removes two fields, not all routes to a protected attribute. Closing it needs a measurement with matched pairs
  on those fields and real outcome data.
- **Prompt sensitivity beyond the floors:** the floors change one name or university; the order probe
  (10/40 flips of 40 identical pairs) shows the agent also reacts to layout. A different prompt wording could
  move decisions more; we did not test it.
- **The record is not tamper-evident** (no hash chain, no write-once storage), and the human reviewer's final
  decision is not logged.
- **Applicant notification and the objection procedure** (Law 94-V Art. 19-1(2)–(3)) are institutional and not
  built here.
- **The refusal filter is tested only on invented Latin-script names.**
We did not close these because they need real data, the fund's processes or infrastructure outside a notebook.

## AI-use declaration

We used Claude (Anthropic) as an assistant: to read the lab's code and propose the contents of the ✏️ cells
(refusal and escalation rules, band calibration, the control register), to locate and summarise the primary legal
texts (adilet/zakon.uchet.kz copies of Law 94-V and Law 230-VIII, the 2026 Constitution, EUR-Lex texts, OECD and
UNESCO recommendations) and to draft this document and REPORT.md. The predictions in the notebook were drafted
with the assistant and reviewed by us before the cells that reveal the results were run. The model run and all
numbers are ours; we checked each citation against the text it cites, and we are responsible
for every claim here.
