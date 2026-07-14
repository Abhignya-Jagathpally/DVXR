"""dvxr — command-line toolkit for the DVXR Screen product.

Wraps the validated serving layer (`dvxr.serve.Screener`) in four subcommands:

    dvxr fit      --task <task> --out <dir>        train + calibrate a screener, save it
    dvxr predict  --screener <dir> [--subject ID]  score a held-out subject, explained
    dvxr report   [--screener <dir>]               evidence one-pager (numbers + literature)
    dvxr demo     [--out <dir>]                     build the self-contained demo bundle

Everything runs offline / CPU / deterministic. The screener a subject is scored with reports the
SAME subject-held-out AUROC as the committed benchmark — this is research-grade screening, never a
diagnosis. `predict` scores a real subject drawn from a research cohort (by subject id); ingesting
arbitrary device exports goes through `scripts/convert_*_subject.py` into the canonical schema
first (see docs/MODEL_CARD.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TASKS = ("mumtaz_depression", "eegmat_workload", "wesad_stress", "stress")


def _eprint(*a):
    print(*a, file=sys.stderr)


def _fmt_result(res: dict, screener) -> str:
    lines = []
    lines.append(f"  screening task : {res['label']}")
    lines.append(f"  risk score     : {res['probability']:.3f}  ({res['risk_band'].upper()})")
    lines.append(f"  90% interval   : [{res['interval'][0]:.3f}, {res['interval'][1]:.3f}]")
    ci = res.get("heldout_auroc_ci") or []
    ci_s = f" (CI [{ci[0]:.3f}, {ci[1]:.3f}])" if len(ci) == 2 else ""
    lines.append(f"  held-out AUROC : {res['heldout_auroc']}{ci_s}   ← benchmark-reproduced")
    lines.append(f"  basis          : {res['basis']}")
    lines.append(f"  windows scored : {res['n_windows']}")
    lines.append(f"  caveat         : {res['caveat']}")
    return "\n".join(lines)


def cmd_fit(args) -> int:
    from dvxr.serve.screener import fit_screener
    _eprint(f"[dvxr fit] training screener for {args.task} "
            f"({args.repeats}x{args.folds} subject-held-out CV)…")
    s = fit_screener(args.task, n_repeats=args.repeats, n_folds=args.folds, seed=args.seed)
    out = Path(args.out)
    s.save(out)
    h = s.heldout
    print(f"Saved screener → {out}")
    print(f"  task           : {s.meta.get('label', s.task)}")
    print(f"  representation : {s.representation} ({s.meta.get('encoder','')})")
    print(f"  held-out AUROC : {h['auroc']}  CI {h['auroc_ci']}  ECE {h.get('ece')}")
    print(f"  protocol       : {h.get('protocol')}  "
          f"({h.get('n_subjects')} subjects, {h.get('n_windows')} windows)")
    return 0


def _load_or_fit(args):
    from dvxr.serve.screener import Screener, fit_screener
    if args.screener:
        return Screener.load(Path(args.screener))
    _eprint(f"[dvxr] no --screener given; fitting {args.task} in-memory…")
    return fit_screener(args.task, seed=args.seed)


def cmd_predict(args) -> int:
    import numpy as np
    from dvxr.serve.screener import embed_cohort
    from dvxr.serve.explain import explain, top_feature_attribution

    screener = _load_or_fit(args)
    task = args.task or screener.task
    _eprint(f"[dvxr predict] embedding cohort {task} to draw a held-out subject…")
    emb, y, subjects, _ = embed_cohort(task, screener.representation)

    uniq = list(dict.fromkeys(subjects.tolist()))
    if args.subject is not None:
        sid = args.subject
        cast = type(subjects[0])
        try:
            sid = cast(sid)
        except Exception:
            pass
        if sid not in subjects:
            _eprint(f"error: subject {args.subject!r} not in cohort. Available: {uniq[:12]}…")
            return 2
    else:
        sid = uniq[-1]           # deterministic pick: last subject id
        _eprint(f"[dvxr predict] no --subject; scoring held-out subject {sid!r}")

    mask = subjects == sid
    res = screener.score_subject(emb[mask])
    truth = int(round(float(np.mean(y[mask]))))

    print(f"\nDVXR Screen — {res['label']}")
    print(f"subject: {sid!r}   (cohort ground-truth label: {truth})")
    print(_fmt_result(res, screener))

    attr = top_feature_attribution(screener, emb[mask], k=5)
    if attr:
        print("  top drivers    :")
        for a in attr:
            print(f"      {a['direction']:6s} risk  {a['feature']:14s} ({a['contribution']:+.3f})")

    if not args.no_narrative:
        narr = explain([res])
        print("\n  clinician note :")
        for line in str(narr.get("clinician", "")).splitlines():
            print(f"      {line}")

    if args.json:
        print("\n" + json.dumps({"subject": str(sid), "truth": truth, **res}, indent=2))
    return 0


def cmd_report(args) -> int:
    try:
        from dvxr.serve.evidence import render_report
        print(render_report(screener_dir=args.screener))
        return 0
    except Exception:
        pass
    # fallback until evidence layer (P4) lands: report from the saved manifest
    if not args.screener:
        _eprint("dvxr report: pass --screener <dir>, or wait for the evidence layer (P4).")
        return 2
    m = json.loads((Path(args.screener) / "manifest.json").read_text())
    print(f"DVXR Screen — evidence for: {m['meta'].get('label', m['task'])}")
    print(f"  encoder        : {m['meta'].get('encoder','')}")
    print(f"  held-out AUROC : {m['heldout']['auroc']}  CI {m['heldout']['auroc_ci']}"
          f"  ECE {m['heldout'].get('ece')}")
    print(f"  protocol       : {m['heldout'].get('protocol')}")
    print("  literature     :")
    for ref in m["meta"].get("literature", []):
        print(f"      - {ref}")
    print(f"  caveat         : {m['meta'].get('caveat','')}")
    return 0


def cmd_demo(args) -> int:
    import subprocess
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_dashboard.py"
    if not script.exists():
        _eprint(f"demo builder not found at {script}")
        return 2
    cmd = [sys.executable, str(script)]
    if args.out:
        cmd += ["--out", args.out]
    _eprint(f"[dvxr demo] {' '.join(cmd)}")
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dvxr", description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=7, help="deterministic seed (default 7)")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="train + calibrate a screener and save it")
    f.add_argument("--task", required=True, choices=TASKS)
    f.add_argument("--out", required=True, help="output directory for the screener artifact")
    f.add_argument("--repeats", type=int, default=3)
    f.add_argument("--folds", type=int, default=5)
    f.set_defaults(func=cmd_fit)

    pr = sub.add_parser("predict", help="score a held-out subject with a screener")
    pr.add_argument("--screener", help="saved screener dir (omit to fit in-memory)")
    pr.add_argument("--task", choices=TASKS, help="cohort/task (default: the screener's task)")
    pr.add_argument("--subject", help="subject id to score (default: a held-out one)")
    pr.add_argument("--json", action="store_true", help="also emit the raw result as JSON")
    pr.add_argument("--no-narrative", action="store_true", help="skip the clinician note")
    pr.set_defaults(func=cmd_predict)

    rp = sub.add_parser("report", help="evidence one-pager (numbers + literature)")
    rp.add_argument("--screener", help="saved screener dir")
    rp.set_defaults(func=cmd_report)

    dm = sub.add_parser("demo", help="build the self-contained demo bundle")
    dm.add_argument("--out", help="output directory")
    dm.set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
