import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Deserción escolar (2000–2024)", layout="wide")

st.title("Dashboard: Deserción escolar (2000–2024)")
st.caption("Carga tu CSV (el mismo que usaste en Colab) y verás las mismas vistas y gráficas.")

# =========================
# Carga de archivo (reemplaza google.colab.files.upload)
# =========================
uploaded = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if not uploaded:
    st.info("Sube un CSV para comenzar.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"No pude leer el CSV: {e}")
    st.stop()

# =========================
# Limpieza / preparación (basado en tu notebook)
# =========================
required_cols = {"anio", "nombre_escuela", "alcaldia", "nivel"}
missing = sorted(list(required_cols - set(df.columns)))
if missing:
    st.error(f"Tu CSV no tiene estas columnas requeridas: {missing}")
    st.stop()

df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
for col in ["matricula_inicial", "desertores", "tasa_desercion"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "tasa_desercion" not in df.columns and {"desertores", "matricula_inicial"}.issubset(df.columns):
    df["tasa_desercion"] = (df["desertores"] / df["matricula_inicial"]).replace([np.inf, -np.inf], np.nan)

# filtrar 2000–2024 como en tu Colab (si hay años fuera, se recortan)
df = df[(df["anio"] >= 2000) & (df["anio"] <= 2024)].copy()

# quitar nulos críticos
df = df.dropna(subset=["nombre_escuela", "alcaldia", "nivel", "anio"])

# =========================
# Controles (reemplaza ipywidgets)
# =========================
with st.sidebar:
    st.header("Filtros")

    alcaldias = ["Todas"] + sorted(df["alcaldia"].dropna().unique().tolist())
    niveles = ["Todos"] + sorted(df["nivel"].dropna().unique().tolist())

    alcaldia_sel = st.selectbox("Alcaldía", options=alcaldias, index=0)
    nivel_sel = st.selectbox("Nivel", options=niveles, index=0)

    y_min = int(df["anio"].min()) if pd.notna(df["anio"].min()) else 2000
    y_max = int(df["anio"].max()) if pd.notna(df["anio"].max()) else 2024
    year_range = st.slider("Años", min_value=y_min, max_value=y_max, value=(max(2000, y_min), min(2024, y_max)))

    view = st.radio(
        "Vista",
        options=["General", "Por alcaldía", "Por nivel", "Heatmap", "Escuelas Top"],
        index=0,
    )

    pred = st.radio("Predicción", options=["Sin predicción", "Con predicción 2025-2030"], index=0)

def filtrar_df():
    d = df.copy()
    y0, y1 = year_range
    d = d[(d["anio"] >= y0) & (d["anio"] <= y1)]
    if alcaldia_sel != "Todas":
        d = d[d["alcaldia"] == alcaldia_sel]
    if nivel_sel != "Todos":
        d = d[d["nivel"] == nivel_sel]
    return d

def kpis(d):
    agg = d.groupby("anio").agg(
        matricula=("matricula_inicial", "sum"),
        desertores=("desertores", "sum"),
    ).reset_index()

    agg["tasa"] = (agg["desertores"] / agg["matricula"]).replace([np.inf, -np.inf], np.nan)

    tasa_prom = float(agg["tasa"].mean()) if len(agg) else np.nan
    desertores_tot = float(agg["desertores"].sum()) if len(agg) else np.nan
    matricula_tot = float(agg["matricula"].sum()) if len(agg) else np.nan

    # Pearson entre matrícula y desertores (agregado por año)
    r = p = np.nan
    if len(agg.dropna(subset=["matricula", "desertores"])) >= 3:
        r, p = pearsonr(agg["matricula"].astype(float), agg["desertores"].astype(float))

    return tasa_prom, desertores_tot, matricula_tot, r, p, agg

def add_prediction_line(agg):
    # predicción lineal simple de tasa por año (2025–2030)
    a = agg.dropna(subset=["anio", "tasa"])
    if len(a) < 3:
        return None
    X = a[["anio"]].astype(float).values
    y = a["tasa"].astype(float).values
    model = LinearRegression().fit(X, y)
    future = pd.DataFrame({"anio": list(range(2025, 2031))})
    yhat = model.predict(future[["anio"]].astype(float).values)
    future["tasa_pred"] = np.clip(yhat, 0, 1)
    return future

# =========================
# Render
# =========================
d = filtrar_df()
tasa_prom, desertores_tot, matricula_tot, r, p, agg = kpis(d)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tasa promedio", f"{tasa_prom*100:.2f}%" if not np.isnan(tasa_prom) else "—")
c2.metric("Desertores (total)", f"{desertores_tot:,.0f}" if not np.isnan(desertores_tot) else "—")
c3.metric("Matrícula (total)", f"{matricula_tot:,.0f}" if not np.isnan(matricula_tot) else "—")
if not np.isnan(r):
    c4.metric("Pearson (matrícula vs desertores)", f"r={r:.3f}", f"p={p:.3g}" if not np.isnan(p) else "")
else:
    c4.metric("Pearson (matrícula vs desertores)", "—")

if not np.isnan(r):
    sig = "significativa" if (not np.isnan(p) and p < 0.05) else "no significativa"
    st.write(f"**Pearson:** r={r:.4f}, p={p:.4g} → tendencia **{sig}** en el rango seleccionado.")

# ---------- VISTAS ----------
if view == "General":
    fig = px.line(agg, x="anio", y="tasa", markers=True, title="Tasa de deserción (agregado) por año")
    if pred == "Con predicción 2025-2030":
        future = add_prediction_line(agg)
        if future is not None:
            fig.add_trace(go.Scatter(x=future["anio"], y=future["tasa_pred"], mode="lines+markers", name="Predicción 2025-2030"))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(agg, x="anio", y="desertores", title="Desertores totales por año")
    st.plotly_chart(fig2, use_container_width=True)

elif view == "Por alcaldía":
    alc = d.groupby(["alcaldia", "anio"]).agg(
        matricula=("matricula_inicial", "sum"),
        desertores=("desertores", "sum"),
    ).reset_index()
    alc["tasa"] = (alc["desertores"] / alc["matricula"]).replace([np.inf, -np.inf], np.nan)

    fig = px.line(alc, x="anio", y="tasa", color="alcaldia", title="Tasa por alcaldía (líneas)")
    st.plotly_chart(fig, use_container_width=True)

    rank = alc.groupby("alcaldia")["tasa"].mean().sort_values(ascending=False).head(12).reset_index()
    fig2 = px.bar(rank, x="alcaldia", y="tasa", title="Top 12 alcaldías (tasa promedio)")
    fig2.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)

elif view == "Por nivel":
    nv = d.groupby(["nivel", "anio"]).agg(
        matricula=("matricula_inicial", "sum"),
        desertores=("desertores", "sum"),
    ).reset_index()
    nv["tasa"] = (nv["desertores"] / nv["matricula"]).replace([np.inf, -np.inf], np.nan)

    fig = px.line(nv, x="anio", y="tasa", color="nivel", title="Tasa por nivel (líneas)")
    st.plotly_chart(fig, use_container_width=True)

    if pred == "Con predicción 2025-2030":
        preds = []
        future_years = pd.DataFrame({"anio": list(range(2025, 2031))})
        for nivel in nv["nivel"].unique():
            dn = nv[nv["nivel"] == nivel].dropna(subset=["tasa"])
            if len(dn) < 3:
                continue
            model = LinearRegression().fit(dn[["anio"]].astype(float).values, dn["tasa"].astype(float).values)
            yhat = model.predict(future_years[["anio"]].astype(float).values)
            tmp = future_years.copy()
            tmp["nivel"] = nivel
            tmp["tasa_pred"] = np.clip(yhat, 0, 1)
            preds.append(tmp)

        if preds:
            pred_df = pd.concat(preds, ignore_index=True)
            figp = px.line(pred_df, x="anio", y="tasa_pred", color="nivel", title="Predicción 2025-2030 por nivel")
            st.plotly_chart(figp, use_container_width=True)
        else:
            st.warning("No hubo suficientes datos para predecir por nivel (se necesitan al menos 3 años por nivel).")

elif view == "Heatmap":
    alc = d.groupby(["alcaldia", "anio"]).agg(
        matricula=("matricula_inicial", "sum"),
        desertores=("desertores", "sum"),
    ).reset_index()
    alc["tasa"] = (alc["desertores"] / alc["matricula"]).replace([np.inf, -np.inf], np.nan)
    pivot = alc.pivot_table(index="alcaldia", columns="anio", values="tasa", aggfunc="mean")

    fig = px.imshow(pivot, aspect="auto", title="Heatmap: tasa (alcaldía vs año)")
    st.plotly_chart(fig, use_container_width=True)

elif view == "Escuelas Top":
    esc = d.groupby("nombre_escuela").agg(
        matricula=("matricula_inicial", "sum"),
        desertores=("desertores", "sum"),
    ).reset_index()
    esc["tasa"] = (esc["desertores"] / esc["matricula"]).replace([np.inf, -np.inf], np.nan)
    esc = esc.sort_values("tasa", ascending=False).head(20)

    st.subheader("Top 20 escuelas (por tasa)")
    st.dataframe(esc, use_container_width=True)

    table = go.Figure(data=[go.Table(
        header=dict(values=list(esc.columns)),
        cells=dict(values=[esc[c] for c in esc.columns])
    )])
    table.update_layout(height=420, title="Tabla: Top 20 escuelas")
    st.plotly_chart(table, use_container_width=True)

st.divider()
with st.expander("Ver datos filtrados (preview)"):
    st.dataframe(d.head(200), use_container_width=True)
