# TriageGeist

**Auditable emergency-triage decisions at the cost of compute each one deserves — protocol where there's protocol, statistics where there's regularity, and a bounded, certified LLM only where the judgment is genuinely ambiguous.**

`99.86% of decisions use zero LLM reasoning · the 0.14% that do are typed and certified · ≥2-level under-triage: 3 in 80,000 · every decision traceable`

Modern language models already outscore median emergency physicians on published clinical-reasoning benchmarks. They still do not get deployed in emergency departments, and the gap between capability and deployment is not technical — it is **epistemic**. Triage is a setting where the cost of one catastrophic mistriage is so asymmetric to a hundred small inefficiencies that a system whose reasoning cannot be inspected after the fact is categorically excluded, whatever its mean-case accuracy. A nurse's triage card is a poor reasoning trace, but it is a *traceable* one. A black-box model's confidence score is not.

So the conventional move — make the model more accurate — is the wrong problem. Inter-rater agreement among trained ED clinicians runs κ ≈ 0.7–0.85; a system that beats human agreement on the bulk of cases has already cleared the accuracy bar. The remaining work is **structural**: arrange the system so the non-auditable fraction of decision-making is small, scoped to exactly the cases where humans also disagree, and bounded in form so that even the AI-mediated decision is machine-checkable.

Of 20,000 test patients, **29 (0.14%)** routed through any LLM reasoning at all. The rest resolved by a hard-rule decision tree (**575 · 2.88%**) or a deterministic ensemble whose per-prediction attributions are inspectable (**19,396 · 96.98%**) — **99.86% with zero LLM in the decision.**

## Three regimes, each served by the inference its audit cost allows

ESI triage decomposes into three epistemic regimes, and each should be answered by the *form* of inference whose auditability matches its character.

- **Protocol execution → a symbolic rule tree.** GCS ≤ 8 needs an airway; cardiac arrest is ESI 1. There is no judgment here, only a handbook rule that fires — fully traceable to a single condition and a one-line evidence string. Changing protocol is a code edit, not a retraining run. *575 patients.*
- **Common pattern → a deterministic ensemble.** The bulk of visits are learned regularities over a high-dimensional space. Eleven orthogonal clinical **banks** — severity/NEWS2, consciousness, respiratory, cardiovascular, thermal, pain, complaint, history, demographics, utilization, arrival — each emit a bounded ESI estimate with handbook-seeded floor/ceiling constraints, and enter a 3× CatBoost + 2× LightGBM ensemble as explicit clinical-geometry features; ordinal QWK threshold optimization turns the argmax into an ESI-ordering-aware boundary. Every prediction is one inspectable trace: bank values, feature attributions, post-threshold class. *19,396 patients.*
- **Genuine judgment → a bounded, certified LLM.** A few patients sit on a boundary with no decisive pattern — the ensemble's top-2 classes within 0.20, and the banks themselves in dissent (a low Kuramoto order parameter). These are the cases where two clinicians would also disagree. Only here is an LLM invoked, and under a **typed contract**: a `TriagePacket` renders the prompt deterministically, the model must emit `TriageDecision` JSON over a closed enum (its choice, which bank it sides with, 1–3 `decisive_evidence` categories from a fixed vocabulary of 25), and a deterministic `AnswerCertifier` runs five checks and rejects violators. **The LLM's free-text reasoning is logged for audit and never parsed into the prediction.** Auditability here is structural, not behavioral: the model cannot emit anything the certifier does not recognize. *29 patients.*

## The headline is a safety figure, not an accuracy figure

Evaluated on 5-fold out-of-fold predictions over all 80,000 training patients: **9 predictions (0.011%) deviate by ≥2 ESI levels** from the label, and the clinically dangerous direction — ≥2-level *under*-triage — happens **3 times in 80,000 (0.004%)**. Absence of catastrophic mistriage is the property a triage system must clear before any other metric means anything, and the architecture clears it with room to spare. (Rubric metrics, for completeness: macro-F1 0.9756, accuracy 0.9777, quadratic-weighted κ 0.9895.)

Two properties fall out of the shape, not from tuning:

- **When it is wrong, it is wrong where being wrong is cheapest.** 77.4% of residual disagreements sit on the ESI 4↔5 boundary and 15.3% on 3↔4; only 6.7% touch the high-acuity 1↔2 / 2↔3 boundaries.
- **It diagnoses its own uncertainty.** The cases it routes to the LLM concentrate exactly where its own out-of-fold error rate is an order of magnitude higher — it escalates the cases it is about to get wrong, and only those.

## Reproducing

Competition data is not in the repo — download from Kaggle:

```bash
kaggle competitions download -c triagegeist -p data/ && unzip data/triagegeist.zip -d data/extracted/
python3 submission/train_and_save.py      # train once; writes every inference artifact to submission/models/
python3 analysis/oof_evidentiary.py       # regenerate the out-of-fold evidence used in the writeup
```

## Layout

```
src/                 pipeline: 11-bank decomposition, coherence scoring, complaint lexicon,
                     feature engine, ensemble, QWK optimizer, LLM residual, AnswerCertifier, TriagePacket
analysis/            5-fold OOF evidence + 12-variant forensic ablations
figures/             editorial figure chassis (the auditability triangle, disagreement topology)
submission/          one-shot training entrypoint + saved ensemble weights and schema
submission_packet/   the canonical shipping folder — WRITEUP.md (the full thesis),
                     the load-and-infer Kaggle notebook, format-validated predictions,
                     and a per-prediction provenance trail (submission_audit.json)
```

The complete argument is in **[`submission_packet/WRITEUP.md`](submission_packet/WRITEUP.md)**; the runnable notebook is **[`submission_packet/notebook/triagegeist.ipynb`](submission_packet/notebook/triagegeist.ipynb)**.

## License

Built for the Triagegeist competition (Laitinen-Fredriksson Foundation, 2026). Data is under the Foundation's Non-Commercial Research License — see the competition page.

---

The right shape for clinical AI is not a more accurate black box. It is protocol where there is protocol, statistics where there is regularity, and a small, contractually bounded, certified sliver of judgment where — and only where — the judgment is real.
