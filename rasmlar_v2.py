#!/usr/bin/env python3
# Maqola rasmlari (haqiqiy ma'lumot asosida)
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import tajriba_v2 as tv

DATA = sys.argv[1]; RES = sys.argv[2]; OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4  # mm -> dyuym (Elsevier: 1 ustun 90 mm, 1,5 ustun 140 mm, 2 ustun 190 mm)
def save(name):
    """Elsevier talablari: vektor PDF, 600 dpi TIFF (LZW) va 600 dpi PNG."""
    base = os.path.join(OUT, name)
    plt.savefig(base + ".pdf", facecolor="white")
    plt.savefig(base + ".png", dpi=600, facecolor="white")
    plt.savefig(base + ".tiff", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e1e0d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6})
D = tv.load_all(DATA)

# --- 1-rasm: usul sxemasi
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(190 * MM, 100 * MM))
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off"); ax.grid(False)
def box(x, y, w, h, title, lines, fc, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#333", lw=0.8, ls=ls))
    ax.text(x + w/2, y + h - 2.0, title, ha="center", va="top", fontsize=8, weight="bold", color=INK)
    ax.text(x + w/2, y + h - 6.2, "\n".join(lines), ha="center", va="top", fontsize=7, linespacing=1.35, color=INK)
def arr(x1, y1, x2, y2, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333", ls=ls))
G, B, Y, R = "#f2f2f2", "#dde8f3", "#fbf1d6", "#e7f0e2"
box(1, 30, 19, 20, "«Aqlli suv» posti", ["Sath datchigi: H", "Qurilmada hisoblangan", "sarf: Q = a·Hᵇ", "Qo'shni post: Hⁿᵇ", "(soatlik, data.mwr.uz)"], G)
box(25, 30, 22, 20, "Tayyorlash", ["vaqt bo'yicha saralash,", "takrorlarni o'chirish,", "soatlik to'r, birliklar;", "a, b, c, k — test davridan", "oldingi ma'lumotda"], G)
box(52, 30, 22, 20, "Uch qatlamli aniqlash", ["L1: fizik qoidalar, −1 kodi", "L2: Hampel filtri", "L3a: Q–H formula moslik", "L3b: qo'shni post moslik", "→ ball sₜ, bayroq"], B)
box(79, 30, 20, 20, "Kalman silliqlagich", ["holat: sath x va", "qo'shni kanal siljishi b", "kanallar: H va c+k·Hⁿᵇ", "ML, RTS, kalibrlangan", "interval x̂ₜ ± κσ̂ₜ"], B)
box(52, 3, 22, 19, "Ishonchlilik indeksi", [r"$\mathrm{DQI}_t=\prod_k c_{k,t}^{\,w_k}$", "C, A, T, U komponentlar", "yashil / sariq / qizil"], Y)
box(79, 3, 20, 19, "Dispetcher qarori", ["SUVDATA paneli:", "H, Q + interval", "+ DQI sinfi", "(yakuniy qaror – inson)"], R)
box(25, 3, 22, 19, "Tajribaviy baholash", ["6 post, 2025 mavsumi;", "kiritilgan nosozliklar,", "bo'shliqlar: F1, MAE,", "NSE, PICP, Wilcoxon"], G, ls="--")
arr(20.5, 40, 24.5, 40); arr(47.5, 40, 51.5, 40); arr(74.5, 40, 78.5, 40)
arr(63, 29.5, 63, 22.5); arr(84, 29.5, 70, 22.5); arr(74.5, 12, 78.5, 12); arr(51.5, 33, 47.5, 20, ls="--")
plt.tight_layout(pad=0.2); save("rasm1_sxema")

# --- 2-rasm: haqiqiy nosozliklar ulushi
cen = pd.read_csv(os.path.join(RES, "haqiqiy_nosozliklar.csv"))
cen = cen[cen.yozuvlar > 0].copy()
types = {"xato_kodi": "−1 kodi", "qotib_qolish": "qotib qolish", "manfiy_H": "manfiy H",
         "birlik_xatosi": "birlik xatosi", "manfiy_Q": "manfiy Q", "QH_nomuvofiq": "Q–H nomuvofiq"}
cen["asosiy"] = cen[list(types)].idxmax(axis=1).map(types)
cen = cen.sort_values("jami_nosoz")
fig, ax = plt.subplots(figsize=(190 * MM, 140 * MM))
y = np.arange(len(cen))
ax.barh(y, cen.jami_nosoz * 100, color=C1, height=0.62)
for i, (v, lab) in enumerate(zip(cen.jami_nosoz * 100, cen.asosiy)):
    ax.text(v + 1.2, i, f"{v:.0f} %" + (f"  ({lab})" if v >= 1 else ""), va="center", fontsize=7.5, color=INK)
TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
          ["a","b","v","g","d","e","yo","j","z","i","y","k","l","m","n","o","p","r","s","t","u","f","x","ts","ch","sh","sh","","i","","e","yu","ya",
           "A","B","V","G","D","E","Yo","J","Z","I","Y","K","L","M","N","O","P","R","S","T","U","F","X","Ts","Ch","Sh","Sh","","I","","E","Yu","Ya"]))
NAMES = {"Канегес_ПК_0_50": "Kanegez PK 0+50", "Канегес_ПК_99_00": "Kanegez PK 99+00", "Полвон_канал_ПК_89_00": "Polvon kanali PK 89+00",
         "Хонобод_1_ПК_1_00": "Xonobod-1 PK 1+00", "Зей_яб_ПК_250_00": "Zey yab PK 250+00", "Нукус_яб_канал_ПК_84_00": "Nukus yab PK 84+00",
         "Нукус_яп_канал_ПК_193_00": "Nukus yap PK 193+00"}
def lab(p):
    import re
    t = NAMES.get(p, "".join(TR.get(ch, ch) for ch in p).replace("_", " "))
    t = t.replace("Abilko l", "Abilko'l").replace("R 8", "R-8")
    return re.sub(r"(\d+) (\d+)$", r"\1+\2", t)
ax.set_yticks(y, [lab(p) for p in cen.post], fontsize=7.5)
ax.set_xlim(0, 125); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Nosoz yozuvlar ulushi, %"); ax.grid(axis="y", visible=False)
plt.tight_layout(); save("rasm2_nosozliklar")

# --- 3-rasm: Канегес ПК 0+50 — qurilma ichidagi sarf formulasi buzilishi (2026)
df = D["Канегес_ПК_0_50"]["df"]
m = (df.H > 0) & (df.Q > 0)
a = df[m & (df.index < "2026-01-01")]; b = df[m & (df.index >= "2026-01-01")]
fig, ax = plt.subplots(figsize=(140 * MM, 95 * MM))
ax.scatter(a.H, a.Q, s=9, color=C1, edgecolor="white", linewidth=0.3, label="2022–2025", rasterized=True)
ax.scatter(b.H, b.Q, s=9, color=C2, edgecolor="white", linewidth=0.3, label="2026", rasterized=True)
hh = np.linspace(0.01, df.H[m].max(), 200)
rc = tv.fit_rating(a.H.to_numpy(float), a.Q.to_numpy(float))
ax.plot(hh, tv.rating_q(hh, rc), color=INK, lw=1.2, label=f"Q = {rc['a']:.2f}·H^{rc['b']:.2f}")
ax.set_xlabel("Sath H, m"); ax.set_ylabel("Sarf Q, m³/s"); ax.legend(frameon=False, fontsize=8, loc="upper left")
plt.tight_layout(); save("rasm3_formula_buzilishi")

# --- 4-rasm: 24 soatlik bo'shliqni tiklash misoli (Канегес ПК 99+00, 2025)
s, nbn = "Канегес_ПК_99_00", "Канегес_ПК_0_50"
h = tv.hourly(D[s]["df"], "2025-04-01", "2025-09-30")
hn = tv.hourly(D[nbn]["df"], "2025-04-01", "2025-09-30").reindex(h.index)
Hh = tv.hourly(D[s]["df"]).H; Hn = tv.hourly(D[nbn]["df"]).H.reindex(Hh.index)
tr = pd.concat([Hh, Hn], axis=1).loc[:"2025-04-01"].dropna(); tr = tr[(tr.iloc[:, 0] > 0) & (tr.iloc[:, 1] > 0)]
k_, c_ = np.polyfit(tr.iloc[:, 1], tr.iloc[:, 0], 1)
H = h.H.to_numpy(float); nb = hn.H.to_numpy(float).copy(); nb[(nb < -0.5) | (nb > 10)] = np.nan
# o'zgaruvchan davrdan 24 soatni yashirish
idx = h.index
win = (idx >= "2025-06-10") & (idx < "2025-06-17")
roll = pd.Series(H, index=idx).diff().abs().rolling(24).sum()
cand = roll[win].idxmax(); g0 = idx.get_loc(cand) - 30; gap = slice(g0, g0 + 24)
Hobs = H.copy(); Hobs[gap] = np.nan
xs, sig, _ = tv.kalman(Hobs, c_ + k_ * nb)
lin = pd.Series(Hobs).interpolate().to_numpy()
w = slice(g0 - 48, g0 + 72); t = idx[w]
fig, ax = plt.subplots(figsize=(190 * MM, 100 * MM))
ax.axvspan(idx[g0], idx[g0 + 23], color="#f2f2f2", lw=0)
ax.fill_between(t, (xs - 1.96 * sig)[w], (xs + 1.96 * sig)[w], color=C1, alpha=0.18, lw=0, label="HDR-K 95 % interval (kalibrlanmagan)")
ax.plot(t, H[w], color=INK, lw=1.4, label="Haqiqiy (yashirilgan) qiymat")
ax.plot(t, lin[w], color=C2, lw=1.6, ls="--", label="Chiziqli interpolyatsiya")
ax.plot(t, xs[w], color=C1, lw=2.0, label="HDR-K (qo'shni post bilan)")
ax.set_ylabel("Sath H, m"); ax.legend(frameon=False, fontsize=7.5, loc="best")
fig.autofmt_xdate(); plt.tight_layout(); save("rasm4_tiklash_misoli")
e_l = np.nanmean(np.abs(lin[gap] - H[gap])); e_k = np.nanmean(np.abs(xs[gap] - H[gap]))
print("rasm4 MAE linear", round(e_l, 3), "HDRK", round(e_k, 3), "oraliq", idx[g0], idx[g0 + 23])

# --- 5-rasm: tiklash xatoligi bo'shliq uzunligiga qarab (natijalar tayyor bo'lsa)
f = os.path.join(RES, "barcha_ishga_tushirishlar.csv")
if os.path.exists(f):
    r = pd.read_csv(f)
    pr = r[r.post.str.startswith("Канегес")].groupby("gap_len")[["H_linear_MAE", "H_Kalman_uni_MAE", "H_HDRK_MAE"]].mean() * 100
    fig, ax = plt.subplots(figsize=(140 * MM, 85 * MM))
    xg = np.arange(len(pr)); bw = 0.26
    for j, (col, lab, c) in enumerate([("H_linear_MAE", "Chiziqli", C2), ("H_Kalman_uni_MAE", "Kalman (bir kanal)", C3), ("H_HDRK_MAE", "HDR-K (qo'shni post)", C1)]):
        ax.bar(xg + (j - 1) * bw, pr[col], width=bw - 0.02, color=c, label=lab)
        for i, v in enumerate(pr[col]): ax.text(xg[i] + (j - 1) * bw, v + 0.2, f"{v:.1f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(xg, [f"{g} soat" for g in pr.index]); ax.set_xlabel("Bo'shliq uzunligi"); ax.set_ylabel("MAE, sm")
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="x", visible=False)
    plt.tight_layout(); save("rasm5_tiklash_xatoligi")
print("ok")
