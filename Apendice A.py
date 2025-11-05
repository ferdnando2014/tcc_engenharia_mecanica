# -*- coding: utf-8 -*-

path = "falhas_30RB011.xlsx"

import pandas as pd
from pathlib import Path

# --- Ler e normalizar ---
df = pd.read_excel(path)
df.columns = [c.strip().lower() for c in df.columns]

# Datas e duração (hh:mm:ss -> minutos)
df["inicio"]  = pd.to_datetime(df["início"],  errors="coerce", dayfirst=True)
df["termino"] = pd.to_datetime(df["término"], errors="coerce", dayfirst=True)
dur_str = df["duração"].astype(str).str.replace(r"[^0-9:]", "", regex=True)
df["duracao_min"] = pd.to_timedelta(dur_str, errors="coerce").dt.total_seconds() / 60.0

# --- Ordenar ---
dfp = df.copy().sort_values("inicio")

# --- TBF (horas) ---
dfp["tbf_h"] = dfp["inicio"].diff().dt.total_seconds() / 3600.0

# --- Resumos simples ---
mes_map = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
dfp["mês"] = dfp["inicio"].dt.month.map(mes_map)

resumo_geral = pd.DataFrame({
    "Período analisado":[f'{dfp["inicio"].min():%d/%m/%Y} a {dfp["inicio"].max():%d/%m/%Y}' if not dfp.empty else "—"],
    "Registros (N)":[int(dfp.shape[0])],
    "Parada total (min)":[float(dfp["duracao_min"].sum())],
    "Parada total (h)":[float(dfp["duracao_min"].sum()/60.0)],
    "Duração média (min)":[float(dfp["duracao_min"].mean())],
    "Duração mediana (min)":[float(dfp["duracao_min"].median())],
    "MTBF (h)":[float(dfp["tbf_h"].mean()) if dfp["tbf_h"].notna().any() else None],
    "Primeiro evento":[dfp["inicio"].min().strftime("%d/%m/%Y %H:%M") if not dfp.empty else "—"],
    "Último evento":[dfp["inicio"].max().strftime("%d/%m/%Y %H:%M") if not dfp.empty else "—"],
})

resumo_mensal = (dfp
    .groupby("mês", dropna=False)
    .agg(Registros=("inicio","count"),
         Parada_min=("duracao_min","sum"),
         Parada_h=("duracao_min", lambda x: x.sum()/60.0),
         Duracao_media_min=("duracao_min","mean"),
         Duracao_mediana_min=("duracao_min","median"))
    .reset_index()
)

def pareto(col):
    if col not in dfp.columns: 
        return pd.DataFrame(columns=[col,"Registros","Parada_min","Parada_h"])
    t = (dfp
         .assign(**{col: dfp[col].fillna("—")})
         .groupby(col, dropna=False)
         .agg(Registros=("inicio","count"), Parada_min=("duracao_min","sum"))
         .reset_index()
         .sort_values(["Parada_min","Registros"], ascending=[False,False]))
    t["Parada_h"] = t["Parada_min"]/60.0
    return t.head(10)

por_modo    = pareto("modo de falha")
por_sistema = pareto("sistema")
por_turno   = pareto("turno")

# --- Exportar Excel ---
out_xlsx = Path(path).with_name("quadro_resumo.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
    resumo_geral.to_excel(xw, index=False, sheet_name="00_Resumo_Geral")
    resumo_mensal.to_excel(xw, index=False, sheet_name="01_Resumo_Mensal")
    por_modo.to_excel(xw, index=False, sheet_name="02_Pareto_Modo")
    por_sistema.to_excel(xw, index=False, sheet_name="03_Pareto_Sistema")
    por_turno.to_excel(xw, index=False, sheet_name="04_Por_Turno")
    dfp.to_excel(xw, index=False, sheet_name="99_Dados_Filtrados")

print(f"[OK] Arquivo gerado: {out_xlsx}")
