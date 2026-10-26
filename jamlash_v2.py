#!/usr/bin/env python3
# Natijalarni maqola uchun jamlash -> sonlar.json
import json, sys, os, numpy as np, pandas as pd
from scipy import stats
RES = sys.argv[1]
r = pd.read_csv(os.path.join(RES, "barcha_ishga_tushirishlar.csv"))
meta = json.load(open(os.path.join(RES, "meta_testlar.json")))
cen = pd.read_csv(os.path.join(RES, "haqiqiy_nosozliklar.csv"))
paired = r.post.str.startswith("Канегес")
out = {}

def ms(x):  # o'rtacha ± sd
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else 0.0]

# --- aniqlash (barcha postlar va juft postlar alohida)
det = {}
for grp, m in (("all", np.ones(len(r), bool)), ("paired", paired.values), ("single", ~paired.values)):
    d = {}
    for meth in ("zscore", "iforest", "HDR_noL3", "HDR_noNB", "HDR"):
        d[meth] = {k: ms(r.loc[m, f"det_{meth}_{k}"]) for k in ("P", "R", "F1")}
    d["kinds"] = {k: ms(r.loc[m, f"rec_{k}"]) for k in ("spike", "stuck", "drift", "offset", "unit", "q_formula")}
    det[grp] = d
out["det"] = det
# F1 bo'yicha HDR vs boshqalar (Wilcoxon, juft)
wt = {}
for b in ("zscore", "iforest", "HDR_noL3"):
    st = stats.wilcoxon(r["det_HDR_F1"], r[f"det_{b}_F1"], alternative="greater"); wt[b] = float(st.pvalue)
st = stats.wilcoxon(r.loc[paired, "det_HDR_F1"], r.loc[paired, "det_HDR_noNB_F1"], alternative="greater"); wt["noNB_paired"] = float(st.pvalue)
out["det_wilcoxon_p"] = wt
out["det_per_station"] = r.groupby("post")[["det_HDR_F1", "det_HDR_noNB_F1", "det_HDR_noL3_F1", "det_iforest_F1", "det_zscore_F1"]].mean().to_dict("index")

# --- tiklash
meths = ["LOCF", "seasonal", "linear", "Kalman_uni", "HDRK"]
rec = {}
for grp, m in (("paired", paired.values), ("single", ~paired.values), ("all", np.ones(len(r), bool))):
    g = r[m].groupby(["gap_rate", "gap_len"])
    rec[grp] = {f"{gr}_{gl}": {k: float(v[f"H_{k}_MAE"].mean()) for k in meths} | {f"NSE_{k}": float(v[f"H_{k}_NSE"].mean()) for k in meths}
                | {f"Q_{k}": float(v[f"Q_{k}_MAE"].mean()) for k in meths}
                for (gr, gl), v in g}
    rec[grp + "_bylen"] = {str(gl): {k: float(v[f"H_{k}_MAE"].mean()) for k in meths} for gl, v in r[m].groupby("gap_len")}
out["rec"] = rec
wil = {}
for grp, m in (("paired", paired.values), ("all", np.ones(len(r), bool))):
    wil[grp] = {}
    for b in ("Kalman_uni", "linear", "LOCF", "seasonal"):
        x, y = r.loc[m, "H_HDRK_MAE"].to_numpy(), r.loc[m, f"H_{b}_MAE"].to_numpy()
        k = np.isfinite(x) & np.isfinite(y)
        alt = "less"
        try:
            p = float(stats.wilcoxon(x[k], y[k], alternative=alt).pvalue)
        except ValueError:
            p = float("nan")
        wil[grp][b] = dict(p=p, gain_pct=float(100 * (y[k].mean() - x[k].mean()) / y[k].mean()))
out["rec_wilcoxon"] = wil
for gl in (1, 6, 24):
    mm = paired & (r.gap_len == gl)
    out.setdefault("paired_gain_by_len", {})[str(gl)] = float(100 * (r.loc[mm, "H_linear_MAE"].mean() - r.loc[mm, "H_HDRK_MAE"].mean()) / r.loc[mm, "H_linear_MAE"].mean())

# --- intervallar: xom va kalibrlangan (1-takrorlash bo'yicha kappa, qolganlarida tekshiruv)
zq = [c for c in r.columns if c.startswith("zq")]
pcs = np.array([int(c[2:]) for c in zq])
picp_cal = []
kap = {}
for s, v in r.groupby("post"):
    cal = v[v.run == 0]
    k95 = float(np.interp(95, pcs, cal[zq].mean().to_numpy()))
    kap[s] = k95
    for _, row in v[v.run > 0].iterrows():
        q = row[zq].to_numpy(float)
        picp_cal.append(float(np.interp(k95, q, pcs, left=50, right=99) / 100))
out["undetected"] = ms(r.undetected_share) if "undetected_share" in r else [float("nan"), 0.0]
out["PICP_raw"] = ms(r.PICP); out["PICP_cal"] = ms(picp_cal); out["kappa"] = kap
out["PI_width_raw_cm"] = float(r.PI_width.mean() * 100)

# --- DQI
out["DQI_rho"] = ms(r.DQI_rho)
out["DQI_MAE"] = {c: ms(r[f"DQI_MAE_{c}"] * 100) for c in "gyr"}
out["DQI_share"] = {c: ms(r[f"DQI_share_{c}"] * 100) for c in "gyr"}
out["DQI_sens_max"] = float(r.DQI_sens.max()); out["DQI_sens_mean"] = float(r.DQI_sens.mean())

# --- meta va haqiqiy nosozliklar
out["meta"] = meta["meta"]
c = cen[cen.yozuvlar > 0]
out["census"] = dict(n_posts=int(len(c)), n_records=int(c.yozuvlar.sum()), n_bad95=int((c.jami_nosoz >= 0.95).sum()),
                     n_ok5=int((c.jami_nosoz < 0.05).sum()), median_avail=float(c.mavjudlik.median()),
                     rows=c.round(4).to_dict("records"))
out["runs"] = int(r.run.max() + 1)
json.dump(out, open(os.path.join(RES, "sonlar.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: out[k] for k in ("det_wilcoxon_p", "rec_wilcoxon", "paired_gain_by_len", "PICP_raw", "PICP_cal", "DQI_rho", "DQI_MAE", "DQI_share", "census")}, indent=1, ensure_ascii=False, default=float)[:4000])
print(json.dumps(out["det"]["all"], indent=0)[:1500]); print(json.dumps(out["det"]["paired"]["HDR"])); print(json.dumps(out["det"]["paired"]["HDR_noNB"]))
