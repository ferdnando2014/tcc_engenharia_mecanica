import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, textwrap, json
from datetime import datetime

# sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---------- Carregar base ----------
path = "falhas_30RB011.xlsx"
assert os.path.exists(path), "Arquivo não encontrado."

df = pd.read_excel(path)
df.columns = [c.strip().lower() for c in df.columns]
df['inicio']  = pd.to_datetime(df.get('início', df.get('inicio')), errors='coerce', dayfirst=True)

# Duração em hh:mm:ss -> minutos
dur = df.get('duração', df.get('duracao'))
dur_str = dur.astype(str).str.replace(r'[^0-9:]', '', regex=True)
df['duracao_min'] = pd.to_timedelta(dur_str, errors='coerce').dt.total_seconds()/60.0

# ---------- Filtrar período ----------
start = pd.Timestamp("2025-01-01")
end   = pd.Timestamp("2025-04-30 23:59:59")
dfp = df[(df['inicio']>=start) & (df['inicio']<=end)].copy().sort_values('inicio')

# ---------- Agregação diária ----------
calendar = pd.DataFrame({'date': pd.date_range(start=start.normalize(), end=end.normalize(), freq='D')})
counts = dfp.set_index('inicio').groupby(pd.Grouper(freq='D')).size()
calendar = calendar.merge(counts.rename('failures'), left_on='date', right_index=True, how='left').fillna({'failures':0})
calendar['failures'] = calendar['failures'].astype(int)

# soma diária de minutos de parada
down = dfp.set_index('inicio')[['duracao_min']].groupby(pd.Grouper(freq='D')).sum().rename(columns={'duracao_min':'daily_down_min'})
calendar = calendar.merge(down, left_on='date', right_index=True, how='left').fillna({'daily_down_min':0})

# ---------- Rótulo: falha no dia seguinte (D+1) ----------
calendar['label_next_day'] = (calendar['failures'].shift(-1).fillna(0) > 0).astype(int)

# ---------- Atributos (features) ----------
calendar['dow'] = calendar['date'].dt.dayofweek
calendar['month'] = calendar['date'].dt.month

for w in [3,7,14]:
    calendar[f'fail_roll_{w}d'] = calendar['failures'].rolling(window=w, min_periods=1).sum().shift(1).fillna(0)
    calendar[f'down_roll_{w}d'] = calendar['daily_down_min'].rolling(window=w, min_periods=1).sum().shift(1).fillna(0)

# tempo desde a última falha (em dias)
last_fail_idx = calendar.index[calendar['failures']>0]
days_since = []
count = 1e9
for i, row in calendar.iterrows():
    if row['failures']>0:
        count = 0
    else:
        count = count + 1 if count<1e8 else 1e9
    days_since.append(count)
calendar['days_since_fail'] = np.array(days_since, dtype=float).clip(0, 1e6)

# conjunto final
features = [c for c in calendar.columns if c not in ['date','failures','label_next_day']]
X = calendar[features].fillna(0).values
y = calendar['label_next_day'].values

# ---------- Split temporal 70/30 ----------
split = int(len(calendar)*0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = calendar['date'].iloc[split:].reset_index(drop=True)

# ---------- Modelos ----------
# Árvore de Decisão
tree = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
y_prob_tree = tree.predict_proba(X_test)[:,1]

# Rede Neural (MLP) com padronização
mlp = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                    MLPClassifier(hidden_layer_sizes=(8,), random_state=42, max_iter=600))
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
try:
    y_prob_mlp = mlp.predict_proba(X_test)[:,1]
except Exception:
    y_prob_mlp = None

# ---------- Métricas ----------
def metrics(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["roc_auc"] = None
    else:
        out["roc_auc"] = None
    return out

m_tree = metrics(y_test, y_pred_tree, y_prob_tree)
m_mlp  = metrics(y_test, y_pred_mlp, y_prob_mlp)

# ---------- Matrizes de confusão ----------
def plot_conf(cm, title, path):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['0','1']); plt.yticks(ticks, ['0','1'])
    plt.ylabel('Verdadeiro'); plt.xlabel('Previsto')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha='center', va='center')
    plt.tight_layout(); plt.savefig(path, dpi=170, bbox_inches='tight'); plt.close()

cm_tree = confusion_matrix(y_test, y_pred_tree, labels=[0,1])
cm_mlp  = confusion_matrix(y_test, y_pred_mlp,  labels=[0,1])

p_cm_tree = "cm_tree.png"; plot_conf(cm_tree, "Matriz de Confusão — Árvore de Decisão", p_cm_tree)
p_cm_mlp  = "cm_mlp.png";  plot_conf(cm_mlp,  "Matriz de Confusão — Rede Neural (MLP)", p_cm_mlp)

# ---------- Importância de atributos (Árvore) ----------
importances = getattr(tree, "feature_importances_", None)
p_imp = None
if importances is not None:
    order = np.argsort(importances)[::-1]
    names = [features[i] for i in order]
    vals = importances[order]
    plt.figure()
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), names, rotation=45, ha='right')
    plt.ylabel("Importância")
    plt.title("Importâncias — Árvore de Decisão")
    p_imp = "tree_importances.png"
    plt.tight_layout(); plt.savefig(p_imp, dpi=170, bbox_inches='tight'); plt.close()

# ---------- Arquivos de saída ----------
# Predições por dia (teste)
pred_df = pd.DataFrame({
    "date": dates_test,
    "true_next_day_fail": y_test.astype(int),
    "pred_tree": y_pred_tree.astype(int),
    "pred_mlp": y_pred_mlp.astype(int),
})
if y_prob_tree is not None:
    pred_df["proba_tree"] = y_prob_tree
if y_prob_mlp is not None:
    pred_df["proba_mlp"] = y_prob_mlp

# Tabela de métricas
metrics_df = pd.DataFrame([
    {"modelo":"DecisionTree", **m_tree},
    {"modelo":"MLP", **m_mlp},
])

out_metrics = "ia_prev_falhas_metricas.xlsx"
out_preds   = "ia_prev_falhas_predicoes.xlsx"
with pd.ExcelWriter(out_metrics, engine="openpyxl") as xw:
    metrics_df.to_excel(xw, index=False, sheet_name="metricas")
with pd.ExcelWriter(out_preds, engine="openpyxl") as xw:
    pred_df.to_excel(xw, index=False, sheet_name="predicoes_teste")
