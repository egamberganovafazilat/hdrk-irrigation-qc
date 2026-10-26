#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tajriba v2 — haqiqiy "Aqlli-suv" (data.mwr.uz) ma'lumotlarida.

Asosiy kuzatuv: qurilma faqat sathni (H) o'lchaydi; sarf (Q) qurilma ichida
H dan sarf formulasi bilan hisoblanadi. Shu sababli:
  * H nosozliklari Q ga ham o'tadi va Q–H moslik tekshiruvi ularni ushlamaydi;
    Q–H tekshiruvi (L3a) faqat Q-maydoniga xos nosozliklarni (formula/konfiguratsiya
    xatosi, manfiy Q) aniqlaydi;
  * H nosozliklari uchun mustaqil ma'lumot — o'sha kanaldagi qo'shni post (L3b);
  * qurilma uzilganda H ham, Q ham yo'qoladi, shuning uchun tiklashdagi ikkinchi
    kanal — qo'shni postdan regressiya bilan bashorat qilingan H.

Ishga tushirish:
  python tajriba_v2.py --data "data/data mwr" --out natijalar_v2 --runs 10
"""
import argparse, glob, json, os, unicodedata as ud
import numpy as np, pandas as pd
from scipy import optimize, stats

RNG_SEED = 2026

# ---------------------------------------------------------------- yuklash
def load_all(folder):
    out = {}
    for f in glob.glob(os.path.join(folder, "aqlli_suv_*.csv")):
        key = ud.normalize("NFC", os.path.basename(f))[10:-4]
        if key == "TEST":
            continue
        meta = open(f, encoding="utf-8").readline().strip("# \n")
        df = pd.read_csv(f, comment="#")
        if df.empty:
            out[key] = dict(meta=meta, df=None); continue
        df["t"] = pd.to_datetime(df.olchangan_vaqt.astype(str), format="%Y%m%d%H%M", errors="coerce")
        df = df.dropna(subset=["t"]).sort_values("t").drop_duplicates("t").set_index("t")
        out[key] = dict(meta=meta, df=df[["sath", "sarf"]].rename(columns={"sath": "H", "sarf": "Q"}))
    return out


def hourly(df, start=None, end=None):
    h = df.resample("1h").mean()
    if start: h = h.loc[start:end]
    return h


def mad(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else np.nan


# ---------------------------------------------------------------- sarf formulasi
def fit_rating(H, Q):
    m = (H > 0) & (Q > 0) & np.isfinite(H) & np.isfinite(Q)
    h, q = H[m], Q[m]
    if h.size < 50:
        return None
    b, la = np.polyfit(np.log(h), np.log(q), 1)
    # robust qayta moslash (katta qoldiqlarni chiqarib)
    r = np.log(q) - (la + b * np.log(h)); keep = np.abs(r - np.median(r)) < 5 * max(mad(r), 1e-4)
    b, la = np.polyfit(np.log(h[keep]), np.log(q[keep]), 1)
    r = np.log(q) - (la + b * np.log(h))
    return dict(a=float(np.exp(la)), b=float(b), sigma=float(max(mad(r), 1e-3)))


def rating_q(H, rc):
    return rc["a"] * np.clip(H, 1e-6, None) ** rc["b"]


# ---------------------------------------------------------------- nosozliklar (haqiqiy) ro'yxati
def fault_census(name, df, rc, stuck_h=24):
    """Har bir post bo'yicha haqiqiy nosozliklar statistikasi (kelgan yozuvlar bo'yicha)."""
    H, Q = df.H.to_numpy(float), df.Q.to_numpy(float)
    n = H.size
    code = (H == -1) | (Q == -1)
    valid = ~code
    hv = H[valid]
    unit = "m" if np.nanmedian(np.abs(hv)) < 20 else "sm"
    lim = 10.0 if unit == "m" else 1000.0
    pos_station = np.nanmedian(hv) > 0.2 * (1 if unit == "m" else 100)
    neg_h = valid & (H < 0) & pos_station
    big_h = valid & (H > lim)
    neg_q = valid & (Q < 0)
    # qotib qolish: >= stuck_h soat davomida bir xil H (qiymat > 0)
    stuck = np.zeros(n, bool)
    t = df.index.to_numpy()
    i = 0
    while i < n:
        j = i
        while j + 1 < n and H[j + 1] == H[i]:
            j += 1
        if valid[i] and (t[j] - t[i]) / np.timedelta64(1, "h") >= stuck_h:
            stuck[i:j + 1] = True
        i = j + 1
    rq = np.zeros(n, bool)
    if rc is not None:
        m = valid & (H > 0) & (Q > 0)
        z = np.zeros(n)
        z[m] = (np.log(Q[m]) - np.log(rating_q(H[m], rc))) / rc["sigma"]
        rq = m & (np.abs(z) > 6)
    span_h = (t[-1] - t[0]) / np.timedelta64(1, "h") + 1
    hrs = df.resample("1h").size()
    return dict(
        post=name, yozuvlar=n, davr=f"{str(t[0])[:10]} – {str(t[-1])[:10]}",
        mavjudlik=float((hrs > 0).sum() / span_h), birlik=unit,
        xato_kodi=float(code.mean()), manfiy_H=float(neg_h.mean()), birlik_xatosi=float(big_h.mean()),
        manfiy_Q=float(neg_q.mean()), qotib_qolish=float(stuck.mean()), QH_nomuvofiq=float(rq.mean()),
        jami_nosoz=float((code | neg_h | big_h | neg_q | stuck | rq).mean()),
    )


# ---------------------------------------------------------------- aniqlash (HDR)
CFG = dict(roc_k=8.0, stuck_len=12, hampel_win=25, hampel_k=4.0, rc_k=5.0, nb_k=4.0, nb_win=7, tau=0.75)


def detect(H, Q, rc, h_lo, h_hi, nb=None, nb_map=None, cfg=CFG):
    n = H.size
    # L1: fizik qoidalar
    L1 = (H < h_lo) | (H > h_hi) | (Q < 0) | (H == -1) | (Q == -1)
    dH = np.diff(H, prepend=np.nan)
    thr = cfg["roc_k"] * max(mad(dH), 1e-3)
    med3 = pd.Series(H).rolling(3, center=True, min_periods=2).median().to_numpy()
    L1 |= (np.abs(dH) > thr) & (np.abs(H - med3) > thr / 2)
    stuck = np.zeros(n, bool); run = 1
    for i in range(1, n):
        if np.isfinite(H[i]) and H[i] == H[i - 1]:
            run += 1
            if run >= cfg["stuck_len"]:
                stuck[i - run + 1:i + 1] = True
        else:
            run = 1
    L1 |= stuck
    # L2: Hampel
    s = pd.Series(H)
    med = s.rolling(cfg["hampel_win"], center=True, min_periods=5).median()
    md = s.rolling(cfg["hampel_win"], center=True, min_periods=5).apply(lambda x: mad(x), raw=True)
    floor = 0.01 * np.nanmedian(np.abs(H))
    L2 = (np.abs(s - med) > cfg["hampel_k"] * np.maximum(md, floor)).to_numpy()
    # L3a: Q–H moslik (qurilma ichidagi formulaga nisbatan)
    zq = np.zeros(n)
    m = (H > 0) & (Q > 0)
    zq[m] = (np.log(Q[m]) - np.log(rating_q(H[m], rc))) / rc["sigma"]
    L3a = np.abs(zq) > cfg["rc_k"]
    # L3b: qo'shni post bilan moslik
    L3b = np.zeros(n, bool); zn = np.zeros(n)
    if nb is not None:
        pred = nb_map["c"] + nb_map["k"] * nb
        r = H - pred
        ok = np.isfinite(r)
        rs = pd.Series(r).rolling(cfg["nb_win"], center=True, min_periods=2).median().to_numpy()
        ok = np.isfinite(rs)
        zn[ok] = (rs[ok] - nb_map["mu"]) / nb_map["sd_s"]
        L3b = np.abs(zn) > cfg["nb_k"]
    # Q–H yoki qo'shni post bilan mustaqil nomuvofiqlik chegaradan oshsa — yetarli dalil (0.75);
    # chegaraga yaqin bo'lsa Hampel tasdig'i kerak bo'ladi
    g = np.clip(0.75 * np.maximum(np.abs(zq) / cfg["rc_k"], np.abs(zn) / cfg["nb_k"]), 0, 1)
    g[~(L3a | L3b)] = 0
    score = np.maximum.reduce([1.0 * L1, 0.5 * L2 + 0.5 * (L3a | L3b), g])
    flag = (score >= cfg["tau"]) & np.isfinite(H)
    return dict(flag=flag, score=np.clip(score, 0, 1), L1=L1, L2=L2, L3=(L3a | L3b))


def detect_zscore(H, k=3):
    z = (H - np.nanmean(H)) / np.nanstd(H); return np.abs(z) > k


def detect_iforest(H, Q, contamination=0.03, seed=0):
    from sklearn.ensemble import IsolationForest
    X = np.column_stack([H, Q, np.diff(H, prepend=H[0])])
    m = np.isfinite(X).all(1); out = np.zeros(H.size, bool)
    clf = IsolationForest(contamination=contamination, random_state=seed).fit(X[m])
    out[m] = clf.predict(X[m]) == -1
    return out


# ---------------------------------------------------------------- Kalman (ikki kanal: o'z datchigi + qo'shni post)
def _kf(y1, y2, R1, R2, q, qb, x0, P0):
    """Holat: [x (haqiqiy sath), b (qo'shni kanal siljishi)]. y1 = x + e1, y2 = x + b + e2."""
    n = y1.size
    X = np.zeros((n, 2)); PP = np.zeros((n, 2, 2)); XF = np.zeros((n, 2)); PF = np.zeros((n, 2, 2))
    x = np.array([x0, 0.0]); P = np.diag([P0, 1e-2]); ll = 0.0
    Qm = np.diag([q, qb])
    H1 = np.array([1.0, 0.0]); H2 = np.array([1.0, 1.0])
    for t in range(n):
        P = P + Qm
        X[t] = x; PP[t] = P
        for y, R, h in ((y1[t], R1, H1), (y2[t], R2, H2)):
            if y == y and np.isfinite(R):
                Ph = P @ h; S = h @ Ph + R; v = y - h @ x; K = Ph / S
                ll -= 0.5 * (np.log(6.283185307 * S) + v * v / S)
                x = x + K * v; P = P - np.outer(K, Ph)
        XF[t] = x; PF[t] = P
    XS = XF.copy(); PS = PF.copy()
    for t in range(n - 2, -1, -1):
        C = PF[t] @ np.linalg.inv(PP[t + 1])
        XS[t] = XF[t] + C @ (XS[t + 1] - X[t + 1])
        PS[t] = PF[t] + C @ (PS[t + 1] - PP[t + 1]) @ C.T
    return XS[:, 0], PS[:, 0, 0], ll


def _kf1(y1, R1, q, x0, P0):
    n = y1.size
    xp = np.empty(n); Pp = np.empty(n); xf = np.empty(n); Pf = np.empty(n)
    x, P, ll = x0, P0, 0.0
    for t in range(n):
        P = P + q
        xp[t], Pp[t] = x, P
        a = y1[t]
        if a == a:
            S = P + R1; v = a - x; K = P / S
            ll -= 0.5 * (np.log(6.283185307 * S) + v * v / S); x += K * v; P *= (1 - K)
        xf[t], Pf[t] = x, P
    xs = xf.copy(); Ps = Pf.copy()
    for t in range(n - 2, -1, -1):
        C = Pf[t] / Pp[t + 1]
        xs[t] = xf[t] + C * (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + C * C * (Ps[t + 1] - Pp[t + 1])
    return xs, Ps, ll


def kalman(y1, y2=None):
    sc = max(np.nanvar(np.diff(y1[np.isfinite(y1)])), 1e-8)
    x0 = y1[np.isfinite(y1)][0]; P0 = np.nanvar(y1)
    if y2 is None or not np.isfinite(y2).any():
        nll = lambda th: -_kf1(y1, np.exp(th[1]) * sc, np.exp(th[0]) * sc, x0, P0)[2]
        th = optimize.minimize(nll, [0.0, -2.0], method="L-BFGS-B", bounds=[(-10, 6)] * 2, options=dict(maxiter=80)).x
        xs, Ps, _ = _kf1(y1, np.exp(th[1]) * sc, np.exp(th[0]) * sc, x0, P0)
        return xs, np.sqrt(np.maximum(Ps, 0)), dict(q=np.exp(th[0]) * sc, R_sensor=np.exp(th[1]) * sc)
    # parametrlar qisqa (2 haftalik) bo'laklarda emas, butun qatorda; tezlik uchun har 3-qadam bo'yicha baholanadi
    sub = slice(None)
    nll = lambda th: -_kf(y1[sub], y2[sub], np.exp(th[1]) * sc, np.exp(th[2]) * sc, np.exp(th[0]) * sc, np.exp(th[3]) * sc, x0, P0)[2]
    th = optimize.minimize(nll, [0.0, -2.0, 0.0, -4.0], method="L-BFGS-B", bounds=[(-10, 6)] * 3 + [(-14, 2)],
                           options=dict(maxiter=60)).x
    xs, Ps, _ = _kf(y1, y2, np.exp(th[1]) * sc, np.exp(th[2]) * sc, np.exp(th[0]) * sc, np.exp(th[3]) * sc, x0, P0)
    return xs, np.sqrt(np.maximum(Ps, 0)), dict(q=np.exp(th[0]) * sc, R_sensor=np.exp(th[1]) * sc,
                                               R_nb=np.exp(th[2]) * sc, q_bias=np.exp(th[3]) * sc)


def seasonal_naive(y, period=24):
    out = y.copy()
    for i in np.where(~np.isfinite(out))[0]:
        j = i - period
        while j >= 0 and not np.isfinite(out[j]):
            j -= period
        out[i] = out[j] if j >= 0 else np.nan
    return pd.Series(out).ffill().bfill().to_numpy()


# ---------------------------------------------------------------- DQI
W = dict(C=0.25, A=0.25, T=0.20, U=0.30)


def dqi(Hobs, flag, score, sig, sd_ref, w=W, win=24):
    obs = np.isfinite(Hobs) & ~flag
    C = pd.Series(obs.astype(float)).rolling(win, min_periods=1).mean().to_numpy()
    A = 1 - np.where(np.isfinite(Hobs), score, 0)
    T = np.ones_like(C)  # IoT: real vaqtda keladi; qo'lda kiritilgan hisobotlarda kechikish bo'yicha
    U = 1 - np.minimum(1, sig / sd_ref)
    comps = dict(C=C, A=A, T=T, U=U)
    return np.exp(sum(w[k] * np.log(np.clip(v, 1e-3, 1)) for k, v in comps.items()))


# ---------------------------------------------------------------- nosozlik kiritish
KINDS = ["spike", "stuck", "drift", "offset", "unit", "q_formula"]


def inject(H, Q, rc, rng, gap_rate, gap_len, anom_rate=0.03):
    H = H.copy(); Q = Q.copy(); n = H.size
    truthH, truthQ = H.copy(), Q.copy()
    valid = np.where(np.isfinite(H))[0]
    anom = np.zeros(n, bool); kind = np.full(n, "", object)
    sd = np.nanstd(H)
    n_ev = max(1, int(anom_rate * valid.size / 6))
    for _ in range(n_ev):
        i = int(rng.choice(valid)); k = rng.choice(KINDS)
        L = 1 if k in ("spike", "unit") else int(rng.integers(6, 49))
        j = slice(i, min(i + L, n))
        if k == "spike":   H[j] = H[j] + rng.choice([-1, 1]) * rng.uniform(4, 8) * sd
        elif k == "stuck": H[j] = H[i]
        elif k == "drift": H[j] = H[j] + np.linspace(0, rng.uniform(0.5, 1.5) * sd, H[j].size)
        elif k == "offset": H[j] = H[j] + rng.choice([-1, 1]) * rng.uniform(0.8, 1.5) * sd
        elif k == "unit":  H[j] = H[j] * 100
        if k == "q_formula":
            Q[j] = Q[j] * rng.choice([0.5, 1.8])
        else:
            Q[j] = rating_q(np.clip(H[j], 0, None), rc) * np.where(np.isfinite(truthQ[j]) & (truthQ[j] > 0), truthQ[j] / rating_q(np.clip(truthH[j], 1e-6, None), rc), 1)
        hit = np.isfinite(truthH[j]) & ((np.abs(H[j] - truthH[j]) > 1e-9) | (np.abs(Q[j] - truthQ[j]) > 1e-9))
        anom[j] |= hit; kind[j] = np.where(hit, k, kind[j])
    gap = np.zeros(n, bool)
    for _ in range(max(1, int(gap_rate * valid.size / gap_len))):
        i = int(rng.choice(valid)); gap[i:i + gap_len] = True
    gap &= np.isfinite(truthH) & ~anom
    H[gap] = np.nan; Q[gap] = np.nan
    return H, Q, truthH, truthQ, anom, gap, kind


# ---------------------------------------------------------------- bitta post bo'yicha tajriba
def reg(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return dict(MAE=np.nan, RMSE=np.nan, NSE=np.nan, n=0)
    e = p[m] - y[m]
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                NSE=float(1 - np.sum(e ** 2) / np.sum((y[m] - y[m].mean()) ** 2)), n=int(m.sum()))


def cls(t, p, ev):
    t, p = t[ev], p[ev]
    tp = np.sum(t & p); fp = np.sum(~t & p); fn = np.sum(t & ~p)
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return dict(P=P, R=R, F1=2 * P * R / (P + R) if P + R else 0.0)


def experiment_station(name, Hc, Qc, rc, nb, nb_map, h_lo, h_hi, runs, rng,
                       gap_rates=(0.05, 0.10, 0.20), gap_lens=(1, 6, 24)):
    rows = []
    for gr in gap_rates:
        for gl in gap_lens:
            for r in range(runs):
                H, Q, tH, tQ, anom, gap, kind = inject(Hc, Qc, rc, rng, gr, gl)
                ev = np.isfinite(H)
                o = dict(post=name, gap_rate=gr, gap_len=gl, run=r)
                det = detect(H, Q, rc, h_lo, h_hi, nb, nb_map)
                det0 = detect(H, Q, rc, h_lo, h_hi, None, None)
                cfg_noL3 = dict(CFG); cfg_noL3["rc_k"] = 1e9; cfg_noL3["nb_k"] = 1e9
                det_noL3 = detect(H, Q, rc, h_lo, h_hi, nb, nb_map, cfg_noL3)
                for nm, pm in (("HDR", det["flag"]), ("HDR_noNB", det0["flag"]), ("HDR_noL3", det_noL3["flag"]),
                               ("zscore", detect_zscore(H)), ("iforest", detect_iforest(H, Q, seed=int(rng.integers(1e6))))):
                    for k, v in cls(anom, pm, ev).items():
                        o[f"det_{nm}_{k}"] = v
                for kd in KINDS:
                    m = (kind == kd) & ev
                    o[f"rec_{kd}"] = float(det["flag"][m].mean()) if m.any() else np.nan
                Hobs = H.copy(); Hobs[det["flag"]] = np.nan
                target = (gap | anom) & np.isfinite(tH)
                y2 = (nb_map["c"] + nb_map["k"] * nb) if nb is not None else None
                xs, sig, par = kalman(Hobs, y2)
                xu, su, _ = kalman(Hobs, None)
                rec = dict(HDRK=xs, Kalman_uni=xu, linear=pd.Series(Hobs).interpolate(limit_direction="both").to_numpy(),
                           LOCF=pd.Series(Hobs).ffill().bfill().to_numpy(), seasonal=seasonal_naive(Hobs))
                for nm, yp in rec.items():
                    for k, v in reg(tH[target], yp[target]).items():
                        o[f"H_{nm}_{k}"] = v
                    qp = rating_q(np.clip(yp, 0, None), rc)
                    o[f"Q_{nm}_MAE"] = reg(tQ[target], qp[target])["MAE"]
                lo, hi = xs - 1.96 * sig, xs + 1.96 * sig
                tg = target & ~np.isfinite(Hobs)  # model haqiqatan tiklagan soatlar (bo'shliq + aniqlangan nosozlik)
                o["PICP"] = float(np.mean((tH[tg] >= lo[tg]) & (tH[tg] <= hi[tg])))
                o["PI_width"] = float(np.mean(hi[tg] - lo[tg]))
                o["undetected_share"] = float(np.mean(anom & np.isfinite(Hobs)) / max(np.mean(anom), 1e-9))
                zz = np.abs(xs[tg] - tH[tg]) / np.maximum(sig[tg], 1e-9)
                for pc in range(50, 100, 1):
                    o[f"zq{pc}"] = float(np.percentile(zz, pc))
                D = dqi(H, det["flag"], det["score"], sig, np.nanstd(tH))
                err = np.abs(xs - tH); m = np.isfinite(err)
                o["DQI_rho"] = float(stats.spearmanr(D[m], err[m])[0])
                cl = np.where(D >= 0.8, "g", np.where(D >= 0.5, "y", "r"))
                for c in "gyr":
                    mm = m & (cl == c); o[f"DQI_MAE_{c}"] = float(err[mm].mean()) if mm.any() else np.nan; o[f"DQI_share_{c}"] = float(mm.mean())
                rhos = []
                for kk in W:
                    for f in (0.5, 1.5):
                        w2 = dict(W); w2[kk] *= f; tot = sum(w2.values()); w2 = {a: b / tot for a, b in w2.items()}
                        rhos.append(stats.spearmanr(dqi(H, det["flag"], det["score"], sig, np.nanstd(tH), w2)[m], err[m])[0])
                o["DQI_sens"] = float(np.max(np.abs(np.array(rhos) - o["DQI_rho"])))
                o.update({f"par_{k}": v for k, v in par.items() if v is not None})
                rows.append(o)
    return rows


_G = {}


def _run_station(s):
    D, a, pairs = _G["D"], _G["a"], _G["pairs"]
    rng = np.random.default_rng([RNG_SEED, sum(map(ord, s))])
    df = D[s]["df"]
    hist = df.loc[: a.start]  # parametrlar faqat test davridan OLDINGI ma'lumotdan
    rc = fit_rating(hist.H.to_numpy(float), hist.Q.to_numpy(float)) or fit_rating(df.H.to_numpy(float), df.Q.to_numpy(float))
    hh = hist.H[(hist.H > -0.9)]
    h_lo, h_hi = min(-0.05, hh.quantile(0.001) - 0.5 * mad(hh)), hh.quantile(0.999) * 1.5 + 0.1
    h = hourly(df, a.start, a.end)
    d0 = detect(h.H.to_numpy(float), h.Q.to_numpy(float), rc, h_lo, h_hi)
    Hc, Qc = h.H.to_numpy(float).copy(), h.Q.to_numpy(float).copy()
    Hc[d0["flag"]] = np.nan; Qc[d0["flag"]] = np.nan
    nb = nb_map = None
    if s in pairs:
        hn = hourly(D[pairs[s]]["df"], a.start, a.end).reindex(h.index)
        nbH = hn.H.to_numpy(float).copy(); nbH[(nbH < -0.5) | (nbH > 10)] = np.nan
        Hh = hourly(df).H; Hn = hourly(D[pairs[s]]["df"]).H.reindex(Hh.index)
        tr = pd.concat([Hh, Hn], axis=1).loc[:a.start].dropna(); tr = tr[(tr.iloc[:, 0] > 0) & (tr.iloc[:, 1] > 0)]
        k_, c_ = np.polyfit(tr.iloc[:, 1], tr.iloc[:, 0], 1); res = tr.iloc[:, 0] - (c_ + k_ * tr.iloc[:, 1])
        rsm = res.rolling(CFG["nb_win"], center=True, min_periods=2).median()
        nb_map = dict(k=float(k_), c=float(c_), mu=float(np.median(res)), sd=float(max(mad(res), 1e-3)),
                      sd_s=float(max(mad(rsm.dropna()), 1e-3)), n_train=int(len(tr)))
        nb = nbH
    meta = dict(rating=rc, h_lo=float(h_lo), h_hi=float(h_hi), n=int(np.isfinite(Hc).sum()), coverage=float(np.isfinite(Hc).mean()),
                baseline_flag_share=float(d0["flag"].mean()), nb=pairs.get(s), nb_map=nb_map, H_sd=float(np.nanstd(Hc)),
                H_mean=float(np.nanmean(Hc)))
    rows = experiment_station(s, Hc, Qc, rc, nb, nb_map, h_lo, h_hi, a.runs, rng)
    return s, meta, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="natijalar_v2")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--start", default="2025-04-01"); ap.add_argument("--end", default="2025-09-30")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D = load_all(a.data)

    # 1) haqiqiy nosozliklar ro'yxati
    cen = []
    for k, v in D.items():
        if v["df"] is None or v["df"].empty:
            cen.append(dict(post=k, yozuvlar=0)); continue
        rc = fit_rating(v["df"].H.to_numpy(float), v["df"].Q.to_numpy(float))
        cen.append(fault_census(k, v["df"], rc))
    pd.DataFrame(cen).to_csv(os.path.join(a.out, "haqiqiy_nosozliklar.csv"), index=False)

    # 2) tajriba: suv mavsumi, sath ham sarf ham ishlaydigan postlar
    stations = ["Полвон_канал_ПК_89_00", "Хонобод_1_ПК_1_00", "Канегес_ПК_99_00", "Канегес_ПК_0_50",
                "Зей_яб_ПК_250_00", "Нукус_яп_канал_ПК_193_00"]
    pairs = {"Канегес_ПК_99_00": "Канегес_ПК_0_50", "Канегес_ПК_0_50": "Канегес_ПК_99_00"}
    global _G
    _G = dict(D=D, a=a, pairs=pairs)
    from multiprocessing import Pool
    order = ["Канегес_ПК_99_00", "Канегес_ПК_0_50", "Полвон_канал_ПК_89_00", "Хонобод_1_ПК_1_00", "Зей_яб_ПК_250_00", "Нукус_яп_канал_ПК_193_00"]
    allrows, meta = [], {}
    with Pool(2) as pool:
        for st, m_, rows in pool.imap_unordered(_run_station, order):
            meta[st] = m_; allrows += rows
            pd.DataFrame(allrows).to_csv(os.path.join(a.out, "barcha_ishga_tushirishlar.csv"), index=False)
            print("tayyor:", st, flush=True)
    res = pd.DataFrame(allrows)
    tests = {}
    for b in ("Kalman_uni", "linear", "LOCF", "seasonal"):
        x, y = res["H_HDRK_MAE"].to_numpy(), res[f"H_{b}_MAE"].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        st = stats.wilcoxon(x[m], y[m], alternative="less")
        tests[b] = dict(p=float(st.pvalue), median_gain_pct=float(100 * np.median((y[m] - x[m]) / y[m])), n=int(m.sum()))
    json.dump(dict(meta=meta, wilcoxon=tests), open(os.path.join(a.out, "meta_testlar.json"), "w"), indent=2, ensure_ascii=False, default=float)
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
