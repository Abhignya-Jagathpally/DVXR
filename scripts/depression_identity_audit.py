"""Depression identity-leakage audit (the literature review's #1 recommended check).

The Identity Trap (arXiv:2606.06647): resting-state EEG depression AUROC under subject-
disjoint CV can reflect subject-IDENTITY features that correlate with the label, not a
biomarker — because in Mumtaz each subject carries ONE diagnosis, so subject-held-out CV
cannot separate "depression biomarker" from "subject identity that happens to be MDD/healthy".

This audit measures, on the same band-power features:
  A) depression decodability (subject-held-out) — for context;
  B) SUBJECT-IDENTITY decodability (within-subject window split) — if identity is highly
     decodable AND diagnosis is subject-level, the biomarker claim is unfalsifiable here.

Writes outputs/_r2/depression_identity_audit.md. Honest — reports whatever it finds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main():
    from dvxr.bench.tasks import TASK_BUILDERS
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    task = TASK_BUILDERS["mumtaz_depression"]()
    X = np.concatenate([np.asarray(a, float) for a in task.features.values()], axis=1)
    X = np.nan_to_num(X)
    y = np.asarray(task.y, int)
    groups = np.asarray(task.subject_ids)
    subj = np.unique(groups)
    n_subj = len(subj)

    # A) depression, subject-held-out
    dep = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(X[tr]), y[tr])
        dep.append(roc_auc_score(y[te], m.predict_proba(sc.transform(X[te]))[:, 1]))
    dep_auroc = float(np.mean(dep)) if dep else float("nan")

    # B) subject-identity decodability (within-subject window split — CAN see each subject)
    sid = np.searchsorted(subj, groups)  # 0..n_subj-1
    ident = []
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, sid):
        sc = StandardScaler().fit(X[tr])
        m = HistGradientBoostingClassifier(max_iter=150, max_depth=4).fit(sc.transform(X[tr]), sid[tr])
        ident.append(accuracy_score(sid[te], m.predict(sc.transform(X[te]))))
    ident_acc = float(np.mean(ident))
    chance = 1.0 / n_subj

    # diagnosis is subject-level?
    subj_one_label = all(len(np.unique(y[groups == s])) == 1 for s in subj)

    verdict = ("CONFOUNDED: subject identity is highly decodable AND each subject carries one "
               "diagnosis — subject-held-out CV cannot separate a depression biomarker from subject "
               "identity. The 0.961 headline is an upper bound; a within-subject-label-variation "
               "cohort (or an identity-adversarial control) is required to validate it."
               if ident_acc > 5 * chance and subj_one_label else
               "Identity decodability is modest relative to chance; leakage risk is lower, but "
               "within-subject validation is still the honest confirmation.")

    md = [
        "# Depression identity-leakage audit (Mumtaz, band-power features)\n",
        f"- subjects: **{n_subj}** | windows: {len(X)} | features: {X.shape[1]}",
        f"- diagnosis is subject-level (each subject one label): **{subj_one_label}**",
        "",
        "| test | metric | value | reference |",
        "|---|---|---:|---|",
        f"| A) Depression (subject-held-out) | AUROC | {dep_auroc:.3f} | vs LaBraM headline 0.961 (band-power is weaker) |",
        f"| B) Subject-identity decodability | accuracy | {ident_acc:.3f} | chance = 1/{n_subj} = {chance:.3f} "
        f"({ident_acc/chance:.0f}x chance) |",
        "",
        f"**Verdict:** {verdict}",
        "",
        "This audits the *protocol confound*, not the model: any resting-state depression result on a "
        "between-subject cohort inherits it. The recommended fix is a cohort with within-subject label "
        "variation (e.g. pre/post-treatment) or an identity-adversarial training control.",
    ]
    out = REPO / "outputs/_r2/depression_identity_audit.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
