# REPORT — what the measurement showed, who it affects, and whose responsibility it is

Authors: Zhuman Arnur, Kuralbek Atabek · Lab 06 · Narxoz University. Every number below was printed by our run of
`lab06_harness_says_no.ipynb` (Qwen2.5-3B-Instruct, greedy decoding, `N_PAIRS = 40`, `SEED = 2026`, 200 pairs =
400 profiles).

## 1. Measurement read-out

**Headline.** No attribute stands clear of its own floor in this run: every real attribute's 95 % flip-rate interval overlaps that of its null pair. This is a null result, not proof of no effect. The university did, however, move *scores*: flagship applicants scored on average +2.5 points [+1.1, +4.1] above their regional twins (40 pairs, sign test p = 0.002). Decisions barely changed only because 356 of 400 profiles were shortlisted anyway; with fewer places (a stricter cut-off), the same tilt would turn into rejections of regional applicants. The layout of the prompt mattered far more than any protected attribute (order probe below: 10/40 flips).

The bare agent shortlisted 356 of 400 profiles; 0 of 400 replies were unparseable in this run (the bare pipeline would
have counted any such reply as a rejection without telling anyone).

**Each attribute against its own floor** (flip = the two members of a pair got different outcomes; 95 % Wilson
interval; mean |diff| = mean absolute score difference inside a pair with its bootstrap interval):

- `name_language`: 2/40 flips = 5.0% [1.4%, 16.5%] (2 shortlist<->not), scores differ in 11/40 pairs, mean |diff| 3.12 [1.38, 5.25]; its floor `null_name`: 0/40 = 0.0% [0.0%, 8.8%], scores differ in 7/40, mean |diff| 2.00 [0.62, 3.62]. The two flip-rate intervals overlap, so this run does NOT show that the attribute moved the decision beyond what changing an irrelevant detail of the same kind does (its mean |diff| is larger than the floor's, but that alone is not a finding).
- `gender_coded_name`: 4/40 flips = 10.0% [4.0%, 23.1%] (4 shortlist<->not), scores differ in 6/40 pairs, mean |diff| 1.12 [0.25, 2.13]; its floor `null_name`: 0/40 = 0.0% [0.0%, 8.8%], scores differ in 7/40, mean |diff| 2.00 [0.62, 3.62]. The two flip-rate intervals overlap, so this run does NOT show that the attribute moved the decision beyond what changing an irrelevant detail of the same kind does.
- `university`: 1/40 flips = 2.5% [0.4%, 12.9%] (1 shortlist<->not), scores differ in 10/40 pairs, mean |diff| 2.50 [1.12, 4.12]; its floor `null_university`: 1/40 = 2.5% [0.4%, 12.9%], scores differ in 9/40, mean |diff| 2.12 [0.88, 3.62]. The two flip-rate intervals overlap, so this run does NOT show that the attribute moved the decision beyond what changing an irrelevant detail of the same kind does (its mean |diff| is larger than the floor's, but that alone is not a finding).

**Direction** (each pair turned round so that the named group comes first):

- kazakh-language name vs the others: 28 pairs, first group favoured 2, other favoured 0 (exact McNemar p = 0.500), score gap +0.7 [-1.8, +3.8] (sign test p = 1.000)
- russian-language name vs the others: 26 pairs, first group favoured 0, other favoured 1 (exact McNemar p = 1.000), score gap +0.2 [-1.7, +2.1] (sign test p = 1.000)
- english-language name vs the others: 26 pairs, first group favoured 0, other favoured 1 (exact McNemar p = 1.000), score gap -1.0 [-4.2, +1.9] (sign test p = 1.000)
- male-coded vs female-coded name: 40 pairs, first group favoured 0, other favoured 4 (exact McNemar p = 0.125), score gap -0.4 [-1.2, +0.6] (sign test p = 0.219)
- flagship vs regional university: 40 pairs, first group favoured 1, other favoured 0 (exact McNemar p = 1.000), score gap +2.5 [+1.1, +4.1] (sign test p = 0.002)

**What could be noise, and what the floors cover.** Decoding is greedy, so there is no sampling noise; the floors measure
how much the outputs move when an irrelevant detail of the same kind changes (another name of the same language and
gender coding; another university of the same tier). They keep the prompt wording, the field order and every merit field
fixed, so they do not bound sensitivity to wording or layout. The optional order probe (identical applicants, lines in another order) gave 10/40 flips = 25.0% [14.2%, 40.2%], scores differing in 21/40 pairs, mean |diff| 8.62 [5.88, 11.50]: this is sensitivity to layout alone, which the floors do not measure. A stronger estimate would need more pairs, several
prompt wordings, and more than one model.

**What n = 40 could detect.** An exact two-sided McNemar test at 0.05 needs at least 6 pairs (15.0%) favouring one
side with none the other way (8 with one the other way, 10 with two). A real effect that moves fewer pairs than that is
invisible to this test, so "no measured effect" is not "no effect". Several rows are tests at once, so p-values near
0.05 are read with care.

**Generalisation.** These counts hold for this model, this prompt (`v1`), these invented names and universities and greedy
decoding on our GPU. A different wording, a larger model or real applicants (real names, real universities with known
reputations, missing or messy fields) could behave differently; our run cannot support a claim about them.

## 2. Who is affected

- **Wrongly rejected applicants.** In the bare run 2 name-language pairs, 4 gender-coded pairs and 1 university pair ended
  with one twin rejected and the other shortlisted; these counts do not clear their floors, so we cannot attribute them to the
  attribute, but for the twin who lost the outcome is the same: rejected for no difference in merit. The clearest measured
  tilt is in scores: regional-university applicants scored 2.5 points lower than identical flagship applicants. For a scholarship the cost is
  concrete: the money, possibly the study place itself, a year of time, and a rejection the applicant cannot see the reason
  for. The bare pipeline also turns any unparseable reply into a rejection nobody decided (0 occurred in this run, but nothing
  in the bare design would have shown it).
- **Wrongly shortlisted applicants** displace someone from a limited number of places; the displaced person is invisible in
  any per-applicant view. The costs are not symmetric: a wrong shortlist is usually caught at a later human stage, a wrong
  rejection usually is not.
- **Groups.** Where the direction rows above favour one group (e.g. flagship over regional university, one name-language group
  over another), it is applicants from regional institutions or from the disfavoured language group who bear it, and these
  correlate in Kazakhstan with region, social origin and ethnicity. Where our counts do not clear the floor, these groups would
  still bear any effect smaller than the 6-pair threshold that n = 40 could not see.
- **Not in the brief:** people who never apply because they expect an automated rejection; the reviewers who must read the
  114 of 400 escalated cases (28.5%); applicants refused by the input rules (a real name with an initial or a digit);
  the fund, which carries the legal and reputational risk; and applicants from universities the model has never seen.
- **Compounding.** Used every year, a small tilt against regional universities means fewer students from those places become
  alumni and references, and the data of past winners then look as if regional students are weaker. If the score feeds a later
  stage (an interview list, another fund, a published ranking), the bias travels with it.

## 3. Professional responsibility

**Ours, as developers of the harness.** Before it runs: measure matched pairs against floors (done: Part 1, 200 pairs),
run the refusal path (done: Part 4, 5 refusals of 6 requests, 0 wrong refusals of 400
clean applicants), check the band on the configuration actually deployed (done: calibration on 80 blinded
profiles, band (35, 60); actual escalation share 28.5%). Tell the deployer: the limits in §1 (n = 40, one
prompt, invented names), that blinding changed 122 of 400 outcomes for reasons unrelated to the
applicant, that the log is not tamper-evident, and what the review workload is. Not claim: that the system is fair, that
blinding "removed bias" (the after-table is zero by construction), or that the numbers transfer to real applicants.

**What remains to be done:** repeat with real-format data and more pairs, test other wordings, log the human reviewer's
final decision, make the log append-only, and test the input filter on real names in Kazakh and Russian.

**The institution's responsibilities:** fixing the purpose and the eligibility criteria, informing applicants of automated
processing and handling objections within three working days (Law RK No. 94-V Art. 19-1(2)-(3)), choosing and training the
reviewers, staffing the review load, handling appeals, and deciding retention.

**Where they conflict.** If the fund wants to drop escalation to save the cost of 114 reviews per 400 applications, or to show
the university again because it "predicts success", we have measured what that control does: escalation is the only path by
which an uncertain or unparseable case reaches a person, and the university is not a merit criterion of the fund. Our duty is to
state that in writing, refuse to sign off on the change without a new measurement, and record the decision as the institution's.
