import sys, time, json, glob
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from convert_emotiv_subject import convert
from goal1_pipeline.features import build_signal_windows, feature_columns
from goal1_pipeline.encoders import FeatureEncoder
from goal1_pipeline.models import train_arousal_classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

RAW = glob.glob("/sessions/beautiful-eager-goldberg/mnt/outputs/data/emotiv/*.md.mc.pm.fe.bp.csv")[0]
CH = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
t=time.time()
raw = pd.read_csv(RAW, skiprows=1, low_memory=False)
t0 = raw["Timestamp"].iloc[0]
raw = raw[(raw["Timestamp"]-t0>=100)&(raw["Timestamp"]-t0<=340)].reset_index(drop=True)
clean = pd.DataFrame({"timestamp": pd.to_datetime(raw["Timestamp"], unit="s", utc=True)})
for ch in CH: clean[ch]=raw[f"EEG.{ch}"].astype(float)
clean.to_csv("/tmp/emotiv_clean.csv", index=False)
events = convert("/tmp/emotiv_clean.csv","outputs/canonical_emotiv.csv",device="epocx",
                 subject_id="emotiv_real",session_id="motorimagery",rate_hz=128.0,timestamp_col="timestamp")
print(f"[1-2] canonical: {len(events)} rows x {events['channel'].nunique()} ch")

# labels: strong command, DILATED into contiguous engaged blocks (+-1.5 s)
mc = raw.dropna(subset=["MC.Action"]).copy()
strong = ((mc["MC.Action"]!=1.0)&(mc["MC.ActionPower"]>0.1)).astype(float)
dil = strong.rolling(25, center=True, min_periods=1).max() > 0   # ~+-1.5s at 8 Hz
mcd = pd.DataFrame({"ts": pd.to_datetime(mc["Timestamp"], unit="s", utc=True), "active": dil.values}).sort_values("ts")
ev = events.sort_values("timestamp_utc").reset_index(drop=True)
ev = pd.merge_asof(ev, mcd, left_on="timestamp_utc", right_on="ts", direction="nearest", tolerance=pd.Timedelta("0.5s"))
ev["active"] = ev["active"].astype("boolean").fillna(False).astype(bool)
ev["label_name"]="arousal"; ev["label_value"]=np.where(ev["active"],"high_arousal","low_arousal")
span=(ev["timestamp_utc"].max()-ev["timestamp_utc"].min()).total_seconds()
sec=(ev["timestamp_utc"]-ev["timestamp_utc"].min()).dt.total_seconds()
ev["subject_id"]="emo_f"+np.clip((sec/(span/6)).astype(int),0,5).astype(str)
ev=ev.drop(columns=["ts","active"])
print(f"[3] sample labels: high={int((ev.label_value=='high_arousal').sum())} low={int((ev.label_value=='low_arousal').sum())}")

win = build_signal_windows(ev, window_seconds=4, step_seconds=1, label_name="arousal")
win.to_csv("outputs/emotiv_arousal_windows.csv", index=False)
fc = feature_columns(win)
print(f"[4] windows={len(win)} features={len(fc)} targets={win['target'].value_counts().to_dict()}")
emb = FeatureEncoder(max_components=16).fit_transform(win, fc); emb.to_csv("outputs/emotiv_arousal_embeddings.csv", index=False)
print(f"[5] embeddings {emb.shape}")

# ---- honest leave-one-fold-out CV AUROC on THEIR band-power features ----
y=(win["target"]=="high_arousal").astype(int).to_numpy()
groups=win["subject_id"].to_numpy()
pipe=Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=1000,class_weight="balanced"))])
proba=cross_val_predict(pipe, win[fc], y, cv=GroupKFold(6), groups=groups, method="predict_proba")[:,1]
auroc=roc_auc_score(y, proba)
print(f"[6] *** leave-one-fold-out AUROC (engaged-control vs rest) = {auroc:.3f} ***  (n={len(y)}, pos={int(y.sum())})")
tm=train_arousal_classifier(win); 
json.dump({**tm.metrics,"cv_auroc":float(auroc),"n_windows":int(len(y)),"pos":int(y.sum())}, open("outputs/emotiv_metrics.json","w"), indent=2)
print(f"      calibration: brier={tm.metrics['brier']:.3f} ece={tm.metrics['ece']:.3f}")

# ---- figure ----
mid=(win["window_start"].astype("int64")/1e9); mid=mid-mid.min()+2
order=np.argsort(mid.values)
fig,ax=plt.subplots(2,1,figsize=(11,7),gridspec_kw={"height_ratios":[2,1]}); plt.style.use("default")
ax[0].plot(mid.values[order], proba[order], color="#2456c7", lw=1.6, label="P(engaged) — held-out")
ax[0].scatter(mid.values, y*0.0+ (y*1.0), c=np.where(y==1,"#d6455d","#9bb0c9"), s=10)
for i in range(len(win)):
    if y[i]==1: ax[0].axvspan(mid.values[i]-0.5, mid.values[i]+0.5, color="#d6455d", alpha=0.06)
ax[0].set_title(f"Real EEG through DVXR pipeline — engaged mental-command vs rest  (leave-one-fold-out AUROC = {auroc:.2f})", fontsize=12, fontweight="bold")
ax[0].set_ylabel("P(engaged)"); ax[0].set_xlabel("time (s)"); ax[0].set_ylim(-0.05,1.05); ax[0].legend(loc="upper right", fontsize=9)
fpr,tpr,_=roc_curve(y,proba)
ax[1].plot(fpr,tpr,color="#2456c7",lw=2,label=f"AUROC={auroc:.2f}"); ax[1].plot([0,1],[0,1],"--",color="#999")
ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("true positive rate"); ax[1].set_title("ROC (held-out folds)", fontsize=11, loc="left"); ax[1].legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig("outputs/emotiv_result.png", dpi=130)
print(f"[done] {time.time()-t:.0f}s -> outputs/emotiv_result.png")