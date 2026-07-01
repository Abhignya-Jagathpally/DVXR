import sys, glob, numpy as np, pandas as pd
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
from convert_galea_subject import convert
F = glob.glob("/sessions/beautiful-eager-goldberg/mnt/outputs/data/galea/OpenBCISession_2026-06-08_11-23-34/BrainFlow-RAW_*.csv")[0]
raw = pd.read_csv(F, sep="\t", header=None)
if raw.shape[1]==1: raw = pd.read_csv(F, delim_whitespace=True, header=None)
# find the unix-epoch timestamp column
tcol=None
for c in raw.columns:
    m=raw[c].median()
    if 1.7e9 < m < 1.9e9: tcol=c; break
print(f"cols={raw.shape[1]} rows={len(raw)} ts_col={tcol} dur={(raw[tcol].iloc[-1]-raw[tcol].iloc[0]):.0f}s")
# first 30 s, 16 EEG channels (cols 1..16)
t0=raw[tcol].iloc[0]; raw=raw[raw[tcol]-t0<=30].reset_index(drop=True)
clean=pd.DataFrame({"timestamp": pd.to_datetime(raw[tcol], unit="s", utc=True)})
for i in range(1,17): clean[f"eeg_{i}"]=raw[i].astype(float)
clean.to_csv("/tmp/galea_clean.csv", index=False)
ev=convert("/tmp/galea_clean.csv","outputs/canonical_galea.csv",subject_id="galea_real",session_id="rest")
print(f"GALEA canonical: {len(ev)} rows, {ev['channel'].nunique()} channels, modalities={sorted(ev['modality'].unique())}")
