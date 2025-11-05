# -*- coding: utf-8 -*-

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

# ---------- Ler dados ----------
path = "falhas_30RB011.xlsx"
assert os.path.exists(path), "Arquivo não encontrado."

df = pd.read_excel(path)
df.columns = [c.strip().lower() for c in df.columns]
df['inicio']  = pd.to_datetime(df['início'],  errors='coerce', dayfirst=True)
df['termino'] = pd.to_datetime(df['término'], errors='coerce', dayfirst=True)

# Converter duração hh:mm:ss -> minutos
dur_str = df['duração'].astype(str).str.replace(r'[^0-9:]', '', regex=True)
df['duracao_min'] = pd.to_timedelta(dur_str, errors='coerce').dt.total_seconds()/60.0

# Ordenar
dfp = df.sort_values('inicio').reset_index(drop=True)

# TBF (horas)
events = dfp[['inicio']].dropna().sort_values('inicio').rename(columns={'inicio':'dt'})
events['tbf_h'] = events['dt'].diff().dt.total_seconds()/3600.0
tbf = events['tbf_h'].dropna().values
n = len(tbf)

# MTBF e MTTR
mtbf_h = float(np.nanmean(tbf)) if n>0 else None
mttr_h = float((dfp['duracao_min'].dropna().mean()/60.0)) if dfp['duracao_min'].notna().any() else None

# ---------- Ajuste Weibull (regressão em ranks) ----------
def weibull_rank_regression(samples):
    x = np.array(sorted(np.asarray(samples).astype(float)))
    x = x[x>0]
    n = len(x)
    if n < 3:
        return None
    i = np.arange(1, n+1)
    F = (i - 0.3) / (n + 0.4)
    Y = np.log(-np.log(1 - F))
    X = np.log(x)
    b, a = np.polyfit(X, Y, 1)  # Y = a + bX
    beta = b
    eta = np.exp(-a / b)
    Y_hat = a + b*X
    ss_res = np.sum((Y - Y_hat)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    return {"beta": float(beta), "eta": float(eta), "r2": float(r2), "X": X, "Y": Y, "Y_hat": Y_hat}

fit = weibull_rank_regression(tbf)

def R_weibull(t, beta, eta):
    t = np.asarray(t, dtype=float)
    return np.exp(- (t/eta)**beta)

# Empirical survival S_emp(t) (amostra completa, sem censura) — step function
def empirical_survival(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    # Kaplan-Meier sem censura = produto (1 - d_i / n_i), aqui d_i=1 para cada tempo único.
    unique, counts = np.unique(x, return_counts=True)
    S = []
    s_val = 1.0
    at_risk = n
    idx = 0
    for u, c in zip(unique, counts):
        s_val *= (1.0 - c/at_risk)
        at_risk -= c
        S.append((u, s_val))
        idx += 1
    t_vals = np.array([u for u,_ in S])
    s_vals = np.array([v for _,v in S])
    return t_vals, s_vals

# ---------- Gráfico 1: Histograma TBF + PDF Weibull ----------
img_hist = "tbf_hist_weibull.png"
if fit is not None and n>0:
    beta, eta = fit["beta"], fit["eta"]
    plt.figure()
    # hist densidade
    plt.hist(tbf, bins=max(8, int(np.sqrt(n))), density=True)
    # pdf Weibull ajustada
    t_plot = np.linspace(max(1e-6, float(np.min(tbf)))*0.8, float(np.max(tbf))*1.2, 300)
    pdf = (beta/eta) * (t_plot/eta)**(beta-1) * np.exp(-(t_plot/eta)**beta)
    plt.plot(t_plot, pdf)
    plt.xlabel("TBF (horas)")
    plt.ylabel("Densidade")
    plt.title(f"Histograma do TBF com ajuste Weibull (β={beta:.2f}, η={eta:.2f} h)")
    plt.tight_layout(); plt.savefig(img_hist, dpi=160, bbox_inches="tight"); plt.close()

# ---------- Gráfico 2: Probabilidade Weibull ----------
img_prob = "tbf_weibull_probplot.png"
if fit is not None:
    X, Y, Yh = fit["X"], fit["Y"], fit["Y_hat"]
    plt.figure()
    plt.scatter(X, Y, label="Dados (ranks)")
    plt.plot(X, Yh, label="Ajuste")
    plt.xlabel("ln(TBF [h])")
    plt.ylabel("ln(-ln(1−F))")
    plt.title(f"Gráfico de Probabilidade Weibull (R²={fit['r2']:.3f})")
    plt.legend()
    plt.tight_layout(); plt.savefig(img_prob, dpi=160, bbox_inches="tight"); plt.close()

# ---------- Gráfico 3: Curva de sobrevivência empírica ----------
img_surv = "tbf_survival_empirica.png"
if n>0:
    t_emp, S_emp = empirical_survival(tbf)
    plt.figure()
    plt.step(t_emp, S_emp, where="post")
    plt.xlabel("t (horas)")
    plt.ylabel("Sobrevivência empírica S(t)")
    plt.title("Curva de sobrevivência empírica — TBF")
    plt.tight_layout(); plt.savefig(img_surv, dpi=160, bbox_inches="tight"); plt.close()

# ---------- R(t) e M(t) (valores de referência) ----------
# R(t): confiabilidade de NÃO falhar até t, pela Weibull ajustada
# M(t): manutenabilidade (prob. de concluir o reparo até t) pela distribuição empírica dos tempos de reparo (duração)
# Escolhemos tempos-padrão práticos: t = 8h e 12h para R; 60, 120 e 240 min para M.
metrics = []
if fit is not None:
    beta, eta = fit["beta"], fit["eta"]
    for t in [8, 12]:
        metrics.append({"Métrica":"R(t)", "Horizonte":"%dh" % t, "Valor": float(R_weibull(t, beta, eta))})

# M(t) a partir da CDF empírica dos tempos de reparo (minutos)
dur_min = dfp['duracao_min'].dropna().values
if dur_min.size > 0:
    for minutes in [60, 120, 240]:
        val = float(np.mean(dur_min <= minutes))
        metrics.append({"Métrica":"M(t)", "Horizonte":"%d min" % minutes, "Valor": val})

# MTBF & MTTR
metrics.append({"Métrica":"MTBF", "Horizonte":"—", "Valor": mtbf_h})
metrics.append({"Métrica":"MTTR", "Horizonte":"—", "Valor": mttr_h})

metrics_df = pd.DataFrame(metrics)

# ---------- Exportar Excel compacto ----------
out_xlsx = "weibull_metricas_graficos_30RB011.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
    pd.DataFrame({"TBF_h": tbf}).to_excel(xw, index=False, sheet_name="TBF")
    pd.DataFrame({"Duracao_min": dur_min}).to_excel(xw, index=False, sheet_name="Reparo(min)")
    pd.DataFrame([{"beta": fit["beta"], "eta_h": fit["eta"], "R2": fit["r2"]}]).to_excel(xw, index=False, sheet_name="Weibull_Params")
    metrics_df.to_excel(xw, index=False, sheet_name="Metricas")

{
 "graficos": [img_hist, img_prob, img_surv],
 "planilha": out_xlsx,
 "beta": float(fit["beta"]) if fit else None,
 "eta": float(fit["eta"]) if fit else None,
 "r2": float(fit["r2"]) if fit else None,
 "mtbf_h": mtbf_h,
 "mttr_h": mttr_h
}
