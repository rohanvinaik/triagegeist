# TriageGeist

**Auditable emergency-triage decisions at the cost of compute each one deserves — protocol where there's protocol, statistics where there's regularity, and a bounded, certified LLM only where the judgment is genuinely ambiguous.**

`99.86% of decisions use zero LLM reasoning · the 0.14% that do are typed and certified · ≥2-level under-triage: 3 in 80,000 · every decision traceable`

Modern language models already outscore median emergency physicians on published clinical-reasoning benchmarks. They still do not get deployed in emergency departments. The gap between capability and deployment is not technical — it is **epistemic**. Triage is a setting where the cost of one catastrophic mistriage is so asymmetric to a hundred small inefficiencies that a system whose reasoning cannot be inspected after the fact is categorically excluded, whatever its mean-case accuracy. A nurse's triage card is a poor reasoning trace, but it is a *traceable* one. A black-box model's confidence score is not.

So the conventional move — make the model more accurate — is the wrong problem. Inter-rater agreement among trained ED clinicians runs κ ≈ 0.7–0.85, and it concentrates on the ESI 3↔4 and 2↔3 boundaries, where two clinicians reach different protocol-defensible answers. A system that beats human agreement on the bulk of cases has already cleared the accuracy bar. The remaining work is **structural**: make the non-auditable fraction small, scoped to exactly the cases where humans also disagree, and bounded in form so that even the AI-mediated decision is machine-checkable. Here is what that produces on 20,000 test patients:

```
    575    (2.88%)   a hard-rule decision tree       a handbook rule fires; one-line evidence string
    19,396 (96.98%)  a deterministic ensemble        inspectable per-patient feature attributions
    29     (0.14%)   an LLM under a typed contract   closed-vocab JSON, certified, prose logged never parsed
    ────────────────────────────────────────────────
    99.86% of decisions resolved with zero LLM reasoning in the path
```

## Three regimes, each served by the inference its audit cost allows

ESI triage decomposes into three epistemic regimes, and each should be answered by the *form* of inference whose auditability matches its character.

- **Protocol execution → a symbolic rule tree.** GCS ≤ 8 needs an airway. Cardiac arrest is ESI 1. There is no judgment here, only a handbook rule that fires — traceable to a single condition. Changing protocol is a code edit, not a retraining run. *575 patients.*
- **Common pattern → a deterministic ensemble.** The bulk of visits are learned regularities. Eleven orthogonal clinical **banks** — severity/NEWS2, consciousness, respiratory, cardiovascular, thermal, pain, complaint, history, demographics, utilization, arrival — each emit a bounded ESI estimate with handbook-seeded floor/ceiling constraints, and enter a 3× CatBoost + 2× LightGBM ensemble as explicit clinical-geometry features. Ordinal QWK threshold optimization turns the argmax into an ESI-ordering-aware boundary. Every prediction is one inspectable trace. *19,396 patients.*
- **Genuine judgment → a bounded, certified LLM.** A few patients sit on a boundary with no decisive pattern — the ensemble's top-2 classes within 0.20, and the banks themselves in dissent (a low Kuramoto order parameter). Only here is an LLM invoked, under a **typed contract**: a `TriagePacket` renders the prompt deterministically, the model must emit `TriageDecision` JSON over a closed enum (its choice, which bank it sides with, 1–3 `decisive_evidence` categories from a fixed vocabulary of 25), and a deterministic `AnswerCertifier` runs five checks and rejects violators. **The LLM's free-text reasoning is logged for audit and never parsed into the prediction.** Auditability here is structural, not behavioral: the model cannot emit anything the certifier does not recognize. *29 patients.*

## The headline is a safety figure, not an accuracy figure

Evaluated on 5-fold out-of-fold predictions over 80,000 patients: **9 predictions (0.011%) deviate by ≥2 ESI levels**, and the clinically dangerous direction — ≥2-level *under*-triage — happens **3 times in 80,000 (0.004%)**. Absence of catastrophic mistriage is the property a triage system must clear before any other metric means anything. (Rubric metrics, for completeness: macro-F1 0.9756, accuracy 0.9777, quadratic-weighted κ 0.9895.)

Two properties fall out of the shape, not from tuning. **When it is wrong, it is wrong where being wrong is cheapest** — 77.4% of residual disagreements sit on the ESI 4↔5 boundary, and only 6.7% touch the high-acuity 1↔2 / 2↔3 boundaries. And **it diagnoses its own uncertainty** — the cases it routes to the LLM concentrate exactly where its own out-of-fold error rate is an order of magnitude higher. It escalates the cases it is about to get wrong, and only those.

## The forensic note is part of the result

A 12-variant fold-safe ablation asked which features actually earn the ensemble's lift. The answer is uncomfortable, and it is disclosed rather than buried. The entire **+0.0168 QWK** gain over the bank-only baseline lives in a *single* feature: `temporal_news2_deviation`, a patient's NEWS2 against the expected NEWS2 for their complaint and age. Its mechanism is only partly clinical. An elderly chest-pain patient with a surprisingly low NEWS2 is genuinely informative — a silent MI, a masking medication. But the feature is also partly *laundering* a declined complaint→ESI shortcut through the NEWS2 channel, because the synthetic generator's near-deterministic complaint→ESI mapping inflates every cohort-keyed feature. The feature is kept because the clinical content is real and the laundering is small enough to name — and on real ED data we would expect the lift to be **roughly halved**. The other 14 candidate features were inert or net-negative and were dropped. A system whose authors know which of its own features are doing the work, and which are cheating, clears a higher bar than one whose authors do not.

## What it costs to run

19,971 of the 20,000 patients are decided on a single CPU in about **14 seconds**. The 29 LLM-routed patients are the only network call and the only sub-dollar API cost. The bank decomposition, rule scan, ensemble, and threshold application fit an edge compute envelope — a triage cart, not a server room. A protocol change is a single-file edit to a bank threshold. The certifier is a small, code-reviewable Python class an institution's IT can audit against local policy. The customary assumption is that better systems cost more; in this corner of clinical AI it inverts. The share of decisions that are not auditable is 0.14% — and inside that 0.14%, the LLM is denied the linguistic degrees of freedom by which non-auditability would arise.

## Honest limits

The competition dataset is synthetic, and its near-deterministic complaint→ESI mapping inflates several feature families — the disclosed `temporal_news2_deviation` laundering is the load-bearing instance. Cross-validated agreement with assigned ESI labels is not agreement with downstream outcomes; a system that perfectly reproduces current triage inherits its biases. Thresholds are calibrated to this dataset and would need local re-tuning. The LLM call, though structurally bounded, remains stochastic and version-dependent. No prospective validation has been performed.

The full argument is in **[`submission_packet/WRITEUP.md`](submission_packet/WRITEUP.md)**; the runnable notebook is in **[`submission_packet/notebook/`](submission_packet/notebook/)**.

---

Built for the Triagegeist competition (Laitinen-Fredriksson Foundation, 2026). Data under the Foundation's Non-Commercial Research License.
