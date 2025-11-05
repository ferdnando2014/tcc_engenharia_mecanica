# Clean re-run using a fresh variable name for the directory to avoid collisions.
import os, numpy as np, pandas as pd, matplotlib
os.environ["MPLBACKEND"] = "Agg"
matplotlib.use("Agg", force=True)  # redundância segura
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

data_dir = Path("")
file_path = data_dir / "falhas_30RB011.xlsx"
assert file_path.exists()

# --- Load & prep
df = pd.read_excel(file_path)
df.columns = [c.strip().lower() for c in df.columns]
df['inicio']  = pd.to_datetime(df.get('início', df.get('inicio')), errors='coerce', dayfirst=True)
df['termino'] = pd.to_datetime(df.get('término', df.get('termino')), errors='coerce', dayfirst=True)
dur = df.get('duração', df.get('duracao'))
dur_str = dur.astype(str).str.replace(r'[^0-9:]', '', regex=True)
df['duracao_min'] = pd.to_timedelta(dur_str, errors='coerce').dt.total_seconds()/60.0

start = pd.Timestamp("2025-01-01 00:00:00")
end   = pd.Timestamp("2025-04-30 23:59:59")
dfp = df[(df['inicio']>=start) & (df['inicio']<=end)].copy().sort_values('inicio')

# calendar
cal = pd.DataFrame({'date': pd.date_range(start=start.normalize(), end=end.normalize(), freq='D')})
counts = dfp.set_index('inicio').groupby(pd.Grouper(freq='D')).size()
cal = cal.merge(counts.rename('failures'), left_on='date', right_index=True, how='left').fillna({'failures':0})
cal['failures'] = cal['failures'].astype(int)
down = dfp.set_index('inicio')[['duracao_min']].groupby(pd.Grouper(freq='D')).sum().rename(columns={'duracao_min':'daily_down_min'})
cal = cal.merge(down, left_on='date', right_index=True, how='left').fillna({'daily_down_min':0})
cal['label_next_day'] = (cal['failures'].shift(-1).fillna(0) > 0).astype(int)
cal['dow'] = cal['date'].dt.dayofweek; cal['month'] = cal['date'].dt.month
for w in [3,7,14]:
    cal[f'fail_roll_{w}d'] = cal['failures'].rolling(w, min_periods=1).sum().shift(1).fillna(0)
    cal[f'down_roll_{w}d'] = cal['daily_down_min'].rolling(w, min_periods=1).sum().shift(1).fillna(0)
# days since last fail
ds, cnt = [], 1e9
for _, r in cal.iterrows():
    if r['failures']>0: cnt = 0
    else: cnt = cnt + 1 if cnt<1e8 else 1e9
    ds.append(cnt)
cal['days_since_fail'] = np.array(ds, dtype=float).clip(0, 1e6)

# dow/hora stats
dow_map = {0:'Seg',1:'Ter',2:'Qua',3:'Qui',4:'Sex',5:'Sáb',6:'Dom'}
dow_stats = (cal.assign(dow_name=cal['dow'].map(dow_map))
             .groupby('dow_name', as_index=False)
             .agg(dias=('date','count'),
                  dias_com_falha=('failures', lambda x: int((x>0).sum())),
                  falhas_totais=('failures','sum')))
dow_stats['prob_dia_com_falha'] = dow_stats['dias_com_falha'] / dow_stats['dias']

dfp['hora'] = dfp['inicio'].dt.hour
hora_stats = (dfp.dropna(subset=['hora']).groupby('hora', as_index=False).size().rename(columns={'size':'falhas'}))
if not hora_stats.empty:
    total_falhas = int(hora_stats['falhas'].sum())
    hora_stats['participacao_%'] = 100.0*hora_stats['falhas']/total_falhas

# importances
features = [c for c in cal.columns if c not in ['date','failures','label_next_day','dow']]
X = cal[features].fillna(0).values; y = cal['label_next_day'].values
split = int(len(cal)*0.7)
X_train, X_test = X[:split], X[split:]; y_train, y_test = y[:split], y[split:]
tree = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42).fit(X_train, y_train)
gini_imp = tree.feature_importances_; feat_names = np.array(features)
imp_gini_df = pd.DataFrame({'feature': feat_names, 'importance_gini': gini_imp}).sort_values('importance_gini', ascending=False).head(12)
perm = permutation_importance(tree, X_test, y_test, n_repeats=50, random_state=42, scoring='f1')
imp_perm_df = pd.DataFrame({'feature': feat_names, 'importance_perm': perm.importances_mean}).sort_values('importance_perm', ascending=False).head(12)

# triggers
rows = []
for name, mask in [
    ('T1_down_roll_3d_gt_120', cal['down_roll_3d'] > 120),
    ('T2_fail_roll_7d_ge_3',   cal['fail_roll_7d'] >= 3),
    ('T3_dow_quarta',          cal['dow']==2),
]:
    n = int(mask.sum()); base = int(len(mask))
    prob = float((cal.loc[mask, 'label_next_day']==1).mean()) if n>0 else np.nan
    prob_not = float((cal.loc[~mask, 'label_next_day']==1).mean()) if (base-n)>0 else np.nan
    lift = (prob/prob_not) if (prob_not and not np.isnan(prob_not) and prob_not>0) else np.nan
    rows.append({"trigger": name, "dias_trigger": n, "dias_total": base, "P(falha|trigger)": prob, "P(falha|nao)": prob_not, "lift": lift})
trig_df = pd.DataFrame(rows)

# charts
p_dow = data_dir/"graf_prob_dia_semana.png"
plt.figure(); plt.bar(dow_stats['dow_name'], dow_stats['prob_dia_com_falha']); plt.ylabel("Probabilidade de dia com ≥1 falha"); plt.title("Probabilidade por dia da semana"); plt.tight_layout(); plt.savefig(p_dow, dpi=170, bbox_inches="tight"); plt.close()

p_hora = data_dir/"graf_falhas_por_hora.png"
plt.figure(); plt.bar(hora_stats['hora'], hora_stats['falhas']); plt.xlabel("Hora do dia"); plt.ylabel("Falhas"); plt.title("Distribuição de falhas por hora"); plt.tight_layout(); plt.savefig(p_hora, dpi=170, bbox_inches="tight"); plt.close()

p_imp_g = data_dir/"graf_importancias_gini.png"
plt.figure(); plt.bar(imp_gini_df['feature'], imp_gini_df['importance_gini']); plt.xticks(rotation=45, ha='right'); plt.ylabel("Importância (Gini)"); plt.title("Importância — Árvore (Gini)"); plt.tight_layout(); plt.savefig(p_imp_g, dpi=170, bbox_inches="tight"); plt.close()

p_trig = data_dir/"graf_triggers_prob.png"
plt.figure(); x = np.arange(len(trig_df)); w=0.35
plt.bar(x - w/2, trig_df["P(falha|trigger)"], w, label="Com trigger")
plt.bar(x + w/2, trig_df["P(falha|nao)"], w, label="Sem trigger")
plt.xticks(x, trig_df['trigger'], rotation=45, ha='right'); plt.ylabel("Prob. de falha D+1"); plt.title("Efeito dos gatilhos (D+1)"); plt.legend(); plt.tight_layout(); plt.savefig(p_trig, dpi=170, bbox_inches="tight"); plt.close()

# Weibull quick summary for sheet 'Resumo_Weibull'
events = dfp[["inicio"]].dropna().sort_values("inicio").rename(columns={"inicio":"dt"})
events["tbf_h"] = events["dt"].diff().dt.total_seconds()/3600.0
tbf = events["tbf_h"].dropna().values
def weibull_rank_regression(samples):
    x = np.array(sorted(np.asarray(samples).astype(float))); x = x[x>0]; n = len(x)
    if n < 3: return None
    i = np.arange(1, n+1); F = (i - 0.3) / (n + 0.4)
    Y = np.log(-np.log(1 - F)); X = np.log(x); b, a = np.polyfit(X, Y, 1)
    beta = b; eta = np.exp(-a/b); Yh = a + b*X
    ss_res = np.sum((Y - Yh)**2); ss_tot = np.sum((Y - Y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    return beta, eta, r2
wb = weibull_rank_regression(tbf); beta, eta_h, r2 = (wb if wb else (np.nan, np.nan, np.nan))
def R_weibull(t_h, beta, eta): return float(np.exp(- (t_h/eta)**beta)) if np.isfinite(beta) and np.isfinite(eta) else np.nan
mtbf_h = float(np.nanmean(tbf)) if tbf.size>0 else np.nan
mttr_h = float(dfp['duracao_min'].mean()/60.0) if dfp['duracao_min'].notna().any() else np.nan
wb_panel = pd.DataFrame([
    {"Métrica":"β","Horizonte":"—","Valor": beta},
    {"Métrica":"η(h)","Horizonte":"—","Valor": eta_h},
    {"Métrica":"R²","Horizonte":"—","Valor": r2},
    {"Métrica":"MTBF","Horizonte":"—","Valor": mtbf_h},
    {"Métrica":"MTTR","Horizonte":"—","Valor": mttr_h},
    {"Métrica":"R(t)","Horizonte":"8h","Valor": R_weibull(8, beta, eta_h)},
    {"Métrica":"R(t)","Horizonte":"12h","Valor": R_weibull(12, beta, eta_h)},
])

# Export workbook with images
out_xlsx = data_dir / "manutencao_padroes_recomendacoes.xlsx"
import xlsxwriter
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    dow_stats.to_excel(writer, index=False, sheet_name="DOW_Stats")
    hora_stats.to_excel(writer, index=False, sheet_name="Hora_Stats")
    imp_gini_df.to_excel(writer, index=False, sheet_name="Import_Gini")
    imp_perm_df.to_excel(writer, index=False, sheet_name="Import_Permut")
    trig_df.to_excel(writer, index=False, sheet_name="Triggers")
    wb_panel.to_excel(writer, index=False, sheet_name="Resumo_Weibull")

    wb_obj = writer.book
    ws = wb_obj.add_worksheet("Graficos")
    ws.set_column(0, 0, 20); ws.set_column(1, 1, 60)
    ws.write(0,0,"Figura 1"); ws.write(0,1,"Probabilidade de dia com ≥1 falha por dia da semana"); ws.insert_image(1,1, str(p_dow), {'x_scale':0.9, 'y_scale':0.9})
    ws.write(23,0,"Figura 2"); ws.write(23,1,"Distribuição de falhas por hora do dia"); ws.insert_image(24,1, str(p_hora), {'x_scale':0.9, 'y_scale':0.9})
    ws.write(46,0,"Figura 3"); ws.write(46,1,"Importância de atributos — Árvore (Gini)"); ws.insert_image(47,1, str(p_imp_g), {'x_scale':0.9, 'y_scale':0.9})
    ws.write(69,0,"Figura 4"); ws.write(69,1,"Efeito dos gatilhos em P(falha D+1)"); ws.insert_image(70,1, str(p_trig), {'x_scale':0.9, 'y_scale':0.9})

{"excel_path": str(out_xlsx), "figs": [str(p_dow), str(p_hora), str(p_imp_g), str(p_trig)]}
