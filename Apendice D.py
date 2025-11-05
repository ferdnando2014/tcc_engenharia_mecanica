import numpy as np
import pandas as pd
import os
os.environ["MPLBACKEND"] = "Agg"  # força backend offscreen
import matplotlib
matplotlib.use("Agg")             # redundância segura
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# ---------- ARQUIVOS ----------
PATH_WB   = Path("weibull_metricas_graficos_30RB011.xlsx")
PATH_IA_M = Path("ia_prev_falhas_metricas.xlsx")
PATH_IA_P = Path("ia_prev_falhas_predicoes.xlsx")
PATH_RAW  = Path("falhas_30RB011.xlsx")

# ---------- LER PARÂMETROS WEIBULL ----------
wb_params = pd.read_excel(PATH_WB, sheet_name="Weibull_Params")
beta  = float(wb_params.loc[0, "beta"])
eta_h = float(wb_params.loc[0, "eta_h"])

# ---------- LER PREVISÕES DE IA (TESTE) ----------
ia_preds = pd.read_excel(PATH_IA_P, sheet_name="predicoes_teste")
ia_preds["date"] = pd.to_datetime(ia_preds["date"]).dt.normalize()
y_true = ia_preds["true_next_day_fail"].astype(int).values
proba_tree = ia_preds["proba_tree"].values if "proba_tree" in ia_preds.columns else None
proba_mlp  = ia_preds["proba_mlp"].values  if "proba_mlp"  in ia_preds.columns else None
test_dates = ia_preds["date"]

# ---------- CALENDARIZAÇÃO E BASELINE WEIBULL D+1 ----------
df = pd.read_excel(PATH_RAW)
df.columns = [c.strip().lower() for c in df.columns]
dt_ini = df.get("início", df.get("inicio"))
df["inicio"]  = pd.to_datetime(dt_ini, errors="coerce", dayfirst=True)

start = pd.Timestamp("2025-01-01 00:00:00")
end   = pd.Timestamp("2025-04-30 23:59:59")
dfp = df[(df["inicio"]>=start) & (df["inicio"]<=end)].copy().sort_values("inicio")

cal = pd.DataFrame({"date": pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")})
counts = dfp.set_index("inicio").groupby(pd.Grouper(freq="D")).size()
cal = cal.merge(counts.rename("failures"), left_on="date", right_index=True, how="left").fillna({"failures":0})
cal["failures"] = cal["failures"].astype(int)

# dias desde a última falha (fim do dia)
days_since, cnt = [], 1e9
for _, r in cal.iterrows():
    if r["failures"] > 0: cnt = 0
    else: cnt = cnt + 1 if cnt < 1e8 else 1e9
    days_since.append(cnt)
cal["age_h"] = np.array(days_since, dtype=float).clip(0, 1e6) * 24.0

def weib_next24_prob(age_h, beta, eta):
    age_h = np.asarray(age_h, dtype=float)
    return 1.0 - np.exp(-(((age_h+24.0)/eta)**beta - (age_h/eta)**beta))

weib_p = weib_next24_prob(cal.set_index("date").loc[test_dates, "age_h"].values, beta, eta_h)

# ---------- MÉTRICAS (P/ BARRAS) ----------
def summarize(y_true, p):
    yhat = (p >= 0.5).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, p)) if len(np.unique(y_true))>1 else np.nan,
        "Accuracy": float(accuracy_score(y_true, yhat)),
        "Precision": float(precision_score(y_true, yhat, zero_division=0)),
        "Recall": float(recall_score(y_true, yhat, zero_division=0)),
        "F1": float(f1_score(y_true, yhat, zero_division=0)),
    }

sum_weib = summarize(y_true, weib_p)
sum_tree = summarize(y_true, proba_tree) if proba_tree is not None else None
sum_mlp  = summarize(y_true, proba_mlp)  if proba_mlp  is not None else None

# ---------- GRÁFICO 1: ROC ----------
plt.figure()
fpr_w, tpr_w, _ = roc_curve(y_true, weib_p)
plt.plot(fpr_w, tpr_w, label=f"Weibull (AUC={sum_weib['AUC']:.3f})")
if sum_tree is not None:
    fpr_t, tpr_t, _ = roc_curve(y_true, proba_tree)
    plt.plot(fpr_t, tpr_t, label=f"DecisionTree (AUC={sum_tree['AUC']:.3f})")
if sum_mlp is not None:
    fpr_m, tpr_m, _ = roc_curve(y_true, proba_mlp)
    plt.plot(fpr_m, tpr_m, label=f"MLP (AUC={sum_mlp['AUC']:.3f})")
plt.plot([0,1],[0,1])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC — Weibull vs IA (teste D+1)")
plt.legend()
plt.tight_layout()
plt.savefig("roc_comparativo.png", dpi=170, bbox_inches="tight")
plt.close()

# ---------- GRÁFICO 2: BARRAS DE MÉTRICAS ----------
labels = ["AUC","Accuracy","Precision","Recall","F1"]
models = ["Weibull"]
vals_matrix = [[sum_weib[k] for k in labels]]
if sum_tree is not None:
    models.append("DecisionTree")
    vals_matrix.append([sum_tree[k] for k in labels])
if sum_mlp is not None:
    models.append("MLP")
    vals_matrix.append([sum_mlp[k] for k in labels])

vals = np.array(vals_matrix)
x = np.arange(len(labels))
width = 0.8 / len(models)

plt.figure()
for i, m in enumerate(models):
    plt.bar(x + i*width, vals[i], width, label=m)
plt.xticks(x + width*(len(models)-1)/2, labels)
plt.ylim(0, 1.0)
plt.ylabel("Valor")
plt.title("Métricas — Weibull vs IA (teste D+1; threshold=0,5)")
plt.legend()
plt.tight_layout()
plt.savefig("metricas_barras_comparativo.png", dpi=170, bbox_inches="tight")
plt.close()

# ---------- GRÁFICO 3: SOBREVIVÊNCIA EMPÍRICA vs WEIBULL R(t) ----------
ev = dfp[["inicio"]].dropna().sort_values("inicio").rename(columns={"inicio":"dt"})
ev["tbf_h"] = ev["dt"].diff().dt.total_seconds()/3600.0
tbf = ev["tbf_h"].dropna().values

x_ord = np.sort(tbf); n = len(x_ord)
unique, counts = np.unique(x_ord, return_counts=True)
S_vals, at_risk, s = [], n, 1.0
for u, c in zip(unique, counts):
    s *= (1.0 - c/at_risk)
    at_risk -= c
    S_vals.append((u, s))
t_emp = np.array([u for u,_ in S_vals])
S_emp = np.array([s for _,s in S_vals])

t_grid = np.linspace(max(1e-6, float(x_ord.min())*0.8), float(x_ord.max())*1.2, 300)
R_weib = np.exp(- (t_grid/eta_h)**beta)

plt.figure()
plt.step(t_emp, S_emp, where="post", label="Sobrevivência empírica")
plt.plot(t_grid, R_weib, label=f"Weibull ajustada (β={beta:.2f}, η={eta_h:.2f}h)")
plt.xlabel("t (horas)")
plt.ylabel("S(t) / R(t)")
plt.title("Sobrevivência empírica vs Weibull — TBF")
plt.legend()
plt.tight_layout()
plt.savefig("sobrevivencia_empirica_vs_weibull.png", dpi=170, bbox_inches="tight")
plt.close()

# ---------- EXPORTAR MÉTRICAS USADAS ----------
rows = [{"Modelo":"Weibull", **sum_weib}]
if sum_tree is not None: rows.append({"Modelo":"DecisionTree", **sum_tree})
if sum_mlp  is not None: rows.append({"Modelo":"MLP", **sum_mlp})
pd.DataFrame(rows).to_excel("comparativo_metricas.xlsx", index=False)

print("[OK] Gerados: roc_comparativo.png, metricas_barras_comparativo.png, sobrevivencia_empirica_vs_weibull.png e comparativo_metricas.xlsx")
