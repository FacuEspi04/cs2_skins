#!/usr/bin/env python3
"""Construye la aplicación web del proyecto (docs/index.html).

Toma el CSV que publica el pipeline, reproduce el modelo de la Entrega 3 (versión V2,
elegido con R²) con la misma semilla, calcula todo lo que muestra la aplicación y
escribe una sola página HTML con los gráficos de Altair y los datos adentro.

    python build_app.py                                  # snapshot del 30/10
    python build_app.py --csv data/cs2_skins_dataset_AAAA-MM-DD.csv

Si los datos cambian, se vuelve a correr y la aplicación queda actualizada.
Requiere scikit-learn 1.6.1 (la versión del notebook).
"""
import argparse
import json
import re
from datetime import date
from functools import partial
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RAIZ = Path(__file__).resolve().parent
SEMILLA = 42

# ----------------------------------------------------------------- argumentos
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--csv", default=str(RAIZ / "data" / "cs2_skins_dataset_2026-10-30.csv"))
ap.add_argument("--plantilla", default=str(RAIZ / "app" / "plantilla.html"))
ap.add_argument("--salida", default=str(RAIZ / "docs" / "index.html"))
ap.add_argument("--verificar", default=str(RAIZ / "data" / "modelo_metadatos_notebook.json"),
                help="metadatos que guardó el notebook, para comprobar que los números coinciden")
args = ap.parse_args()

if sklearn.__version__ != "1.6.1":
    print(f"AVISO: scikit-learn {sklearn.__version__}; el notebook usa 1.6.1 y los números pueden diferir.")
np.random.seed(SEMILLA)

# ================================================================ 1 · datos
ruta = Path(args.csv)
SNAPSHOT = re.search(r"(\d{4}-\d{2}-\d{2})", ruta.name).group(1)
df = pd.read_csv(ruta)
for c in ("fecha_snapshot", "captured_at", "collection_release", "caja_fecha"):
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")
for c in ("stattrak", "souvenir"):
    df[c] = df[c].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
if "antiguedad_col_dias" not in df.columns or df["antiguedad_col_dias"].isna().all():
    df["antiguedad_col_dias"] = (df["fecha_snapshot"] - df["collection_release"]).dt.days
print(f"snapshot {SNAPSHOT}: {len(df)} filas")

calidad_ruta = ruta.with_name(f"calidad_{SNAPSHOT}.json")
CALIDAD = json.loads(calidad_ruta.read_text(encoding="utf-8")) if calidad_ruta.exists() else {}
embudo_ruta = ruta.with_name(f"embudo_{SNAPSHOT}.json")
EMBUDO = json.loads(embudo_ruta.read_text(encoding="utf-8")) if embudo_ruta.exists() else {}

COL_ORDINALES = ["rarity_orden", "wear_orden"]
COL_BOOLEANAS = ["stattrak", "souvenir"]
COL_CATEGORICAS = ["weapon", "origen"]
COL_PROB_DROP = ["neg_log_p_item"]
COLUMNAS_MODELO = COL_ORDINALES + COL_BOOLEANAS + COL_CATEGORICAS + COL_PROB_DROP
# Decisión de la sección 7 del notebook: con R², sacar cualquiera de las dos columnas en
# duda (origen, neg_log_p_item) baja el R² por más que el desvío, así que quedan las siete.
COLUMNAS_FINALES = list(COLUMNAS_MODELO)

# ================================================================ 2 · partición (sección 3 del notebook)
FRACCION_TEST = 0.20
orden_col = (df.dropna(subset=["collection_release"]).groupby("collection")["collection_release"]
               .first().sort_values(ascending=False))
filas_acum = df["collection"].value_counts().reindex(orden_col.index).cumsum()
n_col = int((filas_acum < FRACCION_TEST * len(df)).sum()) + 1
COLECCIONES_TEST = set(orden_col.index[:n_col])
es_test = df["collection"].isin(COLECCIONES_TEST)
train = df[~es_test].reset_index(drop=True)
test = df[es_test].reset_index(drop=True)


def percentil_contra(precios, mercado):
    mercado = np.sort(np.asarray(mercado, dtype=float))
    return np.searchsorted(mercado, np.asarray(precios, dtype=float), side="right") / len(mercado) * 100


MERCADO = np.sort(train["sell_price_usd"].to_numpy(dtype=float))
train["percentil"] = percentil_contra(train["sell_price_usd"], MERCADO)
test["percentil"] = percentil_contra(test["sell_price_usd"], MERCADO)
y_tr, y_te = train["percentil"], test["percentil"]
COLUMNAS_X = COLUMNAS_MODELO + ["antiguedad_col_dias", "rarity", "wear"]
X_tr, X_te = train[COLUMNAS_X], test[COLUMNAS_X]
assert not {"percentil", "sell_price_usd", "sell_listings"} & set(X_tr.columns)


def pedazos_temporales(datos, n_bloques=4, fraccion_inicial=0.40):
    fecha = datos.groupby("collection")["collection_release"].first().dropna().sort_values()
    filas = datos["collection"].value_counts().reindex(fecha.index)
    acumulado = (filas.cumsum() / filas.sum()).to_numpy()
    cortes = [fraccion_inicial + (1 - fraccion_inicial) * k / n_bloques for k in range(n_bloques + 1)]
    bloque_de = dict(zip(fecha.index, np.searchsorted(cortes, acumulado, side="left")))
    bloque = datos["collection"].map(bloque_de).fillna(0).astype(int).to_numpy()
    return [(np.where(bloque < k)[0], np.where(bloque == k)[0]) for k in range(1, n_bloques + 1)]


PEDAZOS = pedazos_temporales(train)
tabla_pedazos = []
for i, (i_fit, i_val) in enumerate(PEDAZOS, start=1):
    f_val = train.iloc[i_val]
    tabla_pedazos.append({"pedazo": i, "entrena": len(i_fit), "valida": len(i_val),
                          "desde": f"{f_val['collection_release'].min():%Y}",
                          "hasta": f"{f_val['collection_release'].max():%Y}"})

# ================================================================ 3 · pipeline, métricas, pisos
def preprocesamiento(columnas):
    pasos = []
    numericas = [c for c in COL_ORDINALES + ["antiguedad_col_dias"] if c in columnas]
    if numericas:
        pasos.append(("ordinales", Pipeline([("imputar", SimpleImputer(strategy="median")),
                                             ("escalar", StandardScaler())]), numericas))
    if "neg_log_p_item" in columnas:
        pasos.append(("prob_drop", Pipeline([("imputar", SimpleImputer(strategy="median", add_indicator=True)),
                                             ("escalar", StandardScaler())]), ["neg_log_p_item"]))
    booleanas = [c for c in COL_BOOLEANAS if c in columnas]
    if booleanas:
        pasos.append(("booleanas", "passthrough", booleanas))
    categoricas = [c for c in COL_CATEGORICAS if c in columnas]
    if categoricas:
        pasos.append(("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas))
    return ColumnTransformer(pasos, remainder="drop")


RECORTE = partial(np.clip, a_min=0, a_max=100)


def armar_modelo(regresor, columnas=None):
    columnas = COLUMNAS_MODELO if columnas is None else columnas
    pipe = Pipeline([("prep", preprocesamiento(columnas)), ("modelo", regresor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.asarray, inverse_func=RECORTE, check_inverse=False)


def sesgo(y_real, y_pred):
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_real)))


METRICAS = {"r2": "r2", "mae": "neg_mean_absolute_error", "sesgo": make_scorer(sesgo)}


def evaluar(nombre, modelo):
    r = cross_validate(modelo, X_tr, y_tr, cv=PEDAZOS, scoring=METRICAS)
    return {"modelo": nombre, "r2": float(r["test_r2"].mean()), "desvio": float(r["test_r2"].std()),
            "mae": float(-r["test_mae"].mean()), "sesgo": float(r["test_sesgo"].mean()),
            "_pedazos": r["test_r2"]}


def comparar(base, otro):
    d = base["_pedazos"] - otro["_pedazos"]
    return {"dif": float(d.mean()), "desvio_dif": float(d.std()), "gana": int((d > 0).sum())}


class MediaPorGrupo(RegressorMixin, BaseEstimator):
    def __init__(self, columnas=("rarity",)):
        self.columnas = columnas

    def fit(self, X, y):
        cols = list(self.columnas)
        d = X[cols].copy()
        d["_y"] = np.asarray(y, dtype=float)
        self.tabla_ = d.groupby(cols)["_y"].mean()
        self.media_global_ = float(np.mean(d["_y"]))
        return self

    def predict(self, X):
        cols = list(self.columnas)
        claves = pd.MultiIndex.from_frame(X[cols]) if len(cols) > 1 else X[cols[0]]
        return pd.Series(self.tabla_.reindex(claves).to_numpy()).fillna(self.media_global_).to_numpy()


print("pisos y comparación de modelos…")
pisos = [evaluar("Piso: el promedio", armar_modelo(DummyRegressor(strategy="mean"))),
         evaluar("Tabla por rareza", MediaPorGrupo(("rarity",))),
         evaluar("Tabla por rareza y exterior", MediaPorGrupo(("rarity", "wear")))]
PISO, MEJOR_PISO = pisos[0], max(pisos, key=lambda r: r["r2"])

CANDIDATOS = {
    "Regresión lineal (Ridge)": Ridge(alpha=1.0),
    "Árbol de decisión": DecisionTreeRegressor(random_state=SEMILLA),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEMILLA, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=SEMILLA),
}
REFERENCIA = "Regresión lineal (Ridge)"
comparacion = [evaluar(n, armar_modelo(m)) for n, m in CANDIDATOS.items()]
res = {r["modelo"]: r for r in comparacion}
duelos = []
for nombre in list(CANDIDATOS)[1:]:
    c = comparar(res[nombre], res[REFERENCIA])
    duelos.append({"retador": nombre, **c, "supera": c["dif"] > c["desvio_dif"]})
MEJOR_R2 = max(comparacion, key=lambda r: r["r2"])["modelo"]
ELEGIDO = MEJOR_R2
if MEJOR_R2 != REFERENCIA and not next(d for d in duelos if d["retador"] == MEJOR_R2)["supera"]:
    ELEGIDO = REFERENCIA
print("elegido:", ELEGIDO)

# ================================================================ 4 · ajuste (sección 8)
P = "regressor__modelo__"
GRILLAS = {
    "Regresión lineal (Ridge)": {P + "alpha": [0.01, 0.1, 1, 3, 10, 30, 100]},
    "Árbol de decisión": {P + "max_depth": [3, 5, 8, 12, None], P + "min_samples_leaf": [1, 5, 10, 20, 40]},
    "Random Forest": {P + "max_depth": [None, 8, 16], P + "min_samples_leaf": [1, 3, 5, 10],
                      P + "max_features": [1.0, 0.5, "sqrt"]},
    "Gradient Boosting": {P + "learning_rate": [0.03, 0.1], P + "max_leaf_nodes": [15, 31],
                          P + "min_samples_leaf": [10, 20, 40], P + "loss": ["squared_error", "absolute_error"]},
}
print("grilla…")
busqueda = GridSearchCV(armar_modelo(CANDIDATOS[ELEGIDO], COLUMNAS_FINALES), GRILLAS[ELEGIDO],
                        cv=PEDAZOS, scoring=METRICAS, refit="r2", n_jobs=1).fit(X_tr, y_tr)
cvr = pd.DataFrame(busqueda.cv_results_)
i_mejor = busqueda.best_index_
R2_CV, DESVIO_CV = float(busqueda.best_score_), float(cvr.loc[i_mejor, "std_test_r2"])
MAE_CV, SESGO_CV = float(-cvr.loc[i_mejor, "mean_test_mae"]), float(cvr.loc[i_mejor, "mean_test_sesgo"])
ajustado = {"_pedazos": np.array([cvr.loc[i_mejor, f"split{i}_test_r2"] for i in range(len(PEDAZOS))])}
AJUSTE = comparar(ajustado, res[ELEGIDO])
MEJORES = {k.replace(P, ""): v for k, v in busqueda.best_params_.items()}
N_GRILLA = len(cvr)

# ================================================================ 5 · la prueba (sección 9)
modelo_final = busqueda.best_estimator_
test["predicho"] = modelo_final.predict(X_te)
test["error"] = test["predicho"] - test["percentil"]
piso_te = armar_modelo(DummyRegressor(strategy="mean")).fit(X_tr, y_tr).predict(X_te)
cols_piso = ("rarity", "wear") if "exterior" in MEJOR_PISO["modelo"] else ("rarity",)
piso_inf_te = MediaPorGrupo(cols_piso).fit(X_tr, y_tr).predict(X_te)
R2_TEST, MAE_TEST, SESGO_TEST = r2_score(y_te, test["predicho"]), mean_absolute_error(y_te, test["predicho"]), sesgo(y_te, test["predicho"])
PRUEBA = [
    {"modelo": "Piso: el promedio", "validacion": PISO["r2"], "prueba": r2_score(y_te, piso_te),
     "sesgo_prueba": sesgo(y_te, piso_te)},
    {"modelo": "Tabla por rareza y exterior", "validacion": MEJOR_PISO["r2"], "prueba": r2_score(y_te, piso_inf_te),
     "sesgo_prueba": sesgo(y_te, piso_inf_te)},
    {"modelo": ELEGIDO, "validacion": R2_CV, "prueba": R2_TEST, "sesgo_prueba": SESGO_TEST},
]
por_col = test.groupby("collection")["error"].mean()
COLECCIONES_SESGO_POS = int((por_col > 0).sum())

corrimiento = []
for o in ["caja", "drop de mapa", "paquete souvenir", "sin identificar"]:
    a, b = train[train["origen"] == o], test[test["origen"] == o]
    corrimiento.append({"origen": o, "lado": "entrenamiento", "mediana": float(a["percentil"].median()), "filas": len(a)})
    corrimiento.append({"origen": o, "lado": "prueba", "mediana": float(b["percentil"].median()), "filas": len(b),
                        "sesgo": float(b["error"].mean())})
mapa_tr = train[train["origen"] == "drop de mapa"]
mapa_te = test[test["origen"] == "drop de mapa"]
MAPA = {"col_tr": int(mapa_tr["collection"].nunique()), "col_te": int(mapa_te["collection"].nunique()),
        "ultima_tr": f"{mapa_tr['collection_release'].max():%m/%Y}",
        "primera_te": f"{mapa_te['collection_release'].min():%m/%Y}"}

TRAMOS = [f"{a}–{a + 10}" for a in range(0, 100, 10)]
test["tramo"] = pd.cut(test["percentil"], bins=range(0, 101, 10), labels=TRAMOS, include_lowest=True)


def percentil_a_dolares(p, mercado=MERCADO):
    return np.quantile(mercado, np.clip(p, 0, 100) / 100)


test["precio_predicho"] = percentil_a_dolares(test["predicho"].to_numpy())
test["error_usd"] = test["precio_predicho"] - test["sell_price_usd"]
test["veces"] = 10 ** (np.log10(test["precio_predicho"]) - np.log10(test["sell_price_usd"])).abs()
USD = {
    "real_mediano": float(test["sell_price_usd"].median()),
    "pred_mediano": float(test["precio_predicho"].median()),
    "abs_medio": float(test["error_usd"].abs().mean()),
    "abs_mediano": float(test["error_usd"].abs().median()),
    "veces_mediana": float(test["veces"].median()),
    "dentro_x2": float(100 * (test["veces"] <= 2).mean()),
    "mas_caro": float(100 * (test["error_usd"] > 0).mean()),
}
por_tramo = (test.groupby("tramo", observed=True)
                 .agg(filas=("error", "size"), mae=("error", lambda e: e.abs().mean()), sesgo=("error", "mean"),
                      real=("sell_price_usd", "median"), pred=("precio_predicho", "median"),
                      usd=("error_usd", lambda e: e.abs().median()))
                 .reset_index())
peores = (test.reindex(test["error"].abs().sort_values(ascending=False).index)
              .head(6)[["market_hash_name", "collection", "origen", "percentil", "predicho",
                        "sell_price_usd", "precio_predicho"]])

print("importancia por permutación…")
imp = permutation_importance(modelo_final, X_te[COLUMNAS_FINALES], y_te, scoring="r2", n_repeats=20,
                             random_state=SEMILLA)
NOMBRES_COL = {"rarity_orden": "Rareza", "wear_orden": "Exterior", "stattrak": "StatTrak", "souvenir": "Souvenir",
               "weapon": "Arma", "origen": "Origen", "neg_log_p_item": "Dificultad de drop"}
IMPORTANCIA = sorted([{"columna": NOMBRES_COL[c], "baja": float(m), "desvio": float(s)}
                      for c, m, s in zip(COLUMNAS_FINALES, imp.importances_mean, imp.importances_std)],
                     key=lambda r: -r["baja"])

# ================================================================ 6 · predicciones para explorar
# Prueba: el modelo evaluado de la sección 9. Entrenamiento: un modelo igual que no vio la
# colección del skin (validación cruzada por colección, 5 grupos, con los mismos hiperparámetros).
print("predicciones fuera de muestra para explorar…")
oof = np.zeros(len(train))
for i_fit, i_val in GroupKFold(n_splits=5).split(X_tr, y_tr, groups=train["collection"].fillna("sin colección")):
    m = clone(modelo_final).fit(X_tr.iloc[i_fit], y_tr.iloc[i_fit])
    oof[i_val] = m.predict(X_tr.iloc[i_val])
train["predicho"] = oof
R2_OOF = r2_score(y_tr, oof)

# ================================================================ 7 · el modelo de la aplicación (sección 12)
MERCADO_TOTAL = np.sort(df["sell_price_usd"].to_numpy(dtype=float))
df["q"] = percentil_contra(df["sell_price_usd"], MERCADO_TOTAL)
modelo_app = clone(modelo_final).fit(df[COLUMNAS_X], df["q"])

RAREZAS = ["Consumer Grade", "Industrial Grade", "Mil-Spec Grade", "Restricted", "Classified", "Covert"]
EXTERIORES = ["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn", "Battle-Scarred"]
ARMAS = sorted(df["weapon"].unique())
ORIGENES = ["caja", "drop de mapa", "paquete souvenir", "sin identificar"]
VARIANTES = {"caja": ["Normal", "StatTrak"], "drop de mapa": ["Normal", "StatTrak"],
             "paquete souvenir": ["Souvenir"], "sin identificar": ["Souvenir"]}
NIVELES = [("fácil", 0.15), ("típica", 0.5), ("difícil", 0.85)]   # cuantiles de neg_log_p_item

caja_tr = df[df["origen"] == "caja"]
filas = []
for o in ORIGENES:
    for var in VARIANTES[o]:
        st, sv = int(var == "StatTrak"), int(var == "Souvenir")
        for r_i, r in enumerate(RAREZAS):
            for w_i, w in enumerate(EXTERIORES):
                if o == "caja":
                    g = caja_tr[(caja_tr["rarity"] == r) & (caja_tr["wear"] == w) & (caja_tr["stattrak"] == st)]
                    base_g = g if len(g) >= 5 else caja_tr
                    niveles = [(n, float(base_g["neg_log_p_item"].quantile(qq))) for n, qq in NIVELES]
                else:
                    niveles = [("—", np.nan)]
                for n_i, (nivel, nl) in enumerate(niveles):
                    for a in ARMAS:
                        filas.append({"rarity_orden": r_i, "wear_orden": w_i, "stattrak": st, "souvenir": sv,
                                      "weapon": a, "origen": o, "neg_log_p_item": nl,
                                      "antiguedad_col_dias": np.nan, "rarity": r, "wear": w,
                                      "nivel": n_i if o == "caja" else 1})
grilla = pd.DataFrame(filas)
grilla["pred"] = modelo_app.predict(grilla[COLUMNAS_X])
# Comprobación: la grilla da lo mismo que el modelo sobre un caso armado a mano.
caso = grilla.iloc[[123]]
assert np.isclose(modelo_app.predict(caso[COLUMNAS_X])[0], caso["pred"].iloc[0])

visto_ar = df.groupby(["weapon", "rarity_orden"]).size()
visto_or = df.groupby(["origen", "rarity_orden"]).size()
visto_ov = df.groupby(["origen", "stattrak"]).size()
# Cuántos ítems de caja respaldan cada arma+rareza+exterior+variante: la rebanada exacta
# que mueve "Dificultad de sacarlo de la caja". Mediana real: 4 filas; con tan pocas, el
# nivel elegido no tiene de dónde sacar una relación confiable para esa arma puntual.
visto_an = caja_tr.groupby(["weapon", "rarity_orden", "wear_orden", "stattrak"]).size()
GRILLA = {}
for row in grilla.itertuples(index=False):
    clave = f"{ARMAS.index(row.weapon)}|{row.rarity_orden}|{row.wear_orden}|{ORIGENES.index(row.origen)}|{row.stattrak}|{row.nivel}"
    GRILLA[clave] = round(float(row.pred), 1)
VISTOS = {
    "arma_rareza": {f"{ARMAS.index(a)}|{r}": int(n) for (a, r), n in visto_ar.items()},
    "origen_rareza": {f"{ORIGENES.index(o)}|{r}": int(n) for (o, r), n in visto_or.items()},
    "origen_stattrak": {f"{ORIGENES.index(o)}|{s}": int(n) for (o, s), n in visto_ov.items()},
    "arma_nivel": {f"{ARMAS.index(a)}|{r}|{w}|{st}": int(n) for (a, r, w, st), n in visto_an.items()},
}
CUANTILES_PRECIO = [round(float(np.quantile(MERCADO_TOTAL, k / 1000)), 2) for k in range(1001)]

# ---------------------------------------------------------------- 7b · la fórmula de respaldo
# El instructivo pide que, en una página HTML, el modelo de «Probá» se pueda escribir adentro como
# fórmula. El Gradient Boosting viaja como la tabla completa de sus respuestas (arriba); además, la
# página trae la regresión lineal de la comparación (Ridge, alpha=1, R² en validación a menos de un
# desvío del elegido) escrita como fórmula: estos coeficientes, y la cuenta se hace en el navegador.
ridge_app = armar_modelo(Ridge(alpha=1.0)).fit(df[COLUMNAS_X], df["q"])
pipe_r = ridge_app.regressor_
prep_r, coef_r = pipe_r.named_steps["prep"], pipe_r.named_steps["modelo"].coef_
nombres_r = list(prep_r.get_feature_names_out())
coef_de = dict(zip(nombres_r, coef_r))
esc_ord = prep_r.named_transformers_["ordinales"].named_steps["escalar"]
imp_pd = prep_r.named_transformers_["prob_drop"].named_steps["imputar"]
esc_pd = prep_r.named_transformers_["prob_drop"].named_steps["escalar"]
ohe = prep_r.named_transformers_["categoricas"]
FORMULA = {
    "b": float(pipe_r.named_steps["modelo"].intercept_),
    "num": [[float(esc_ord.mean_[i]), float(esc_ord.scale_[i]), float(coef_de[f"ordinales__{c}"])]
            for i, c in enumerate(COL_ORDINALES)],
    "nl": {"med": float(imp_pd.statistics_[0]), "m": [float(x) for x in esc_pd.mean_], "s": [float(x) for x in esc_pd.scale_],
           "c": [float(coef_de[n_]) for n_ in nombres_r if n_.startswith("prob_drop__")]},
    "st": float(coef_de["booleanas__stattrak"]), "sv": float(coef_de["booleanas__souvenir"]),
    "arma": {a: float(coef_de.get(f"categoricas__weapon_{a}", 0.0)) for a in ARMAS},
    "origen": {o: float(coef_de.get(f"categoricas__origen_{o}", 0.0)) for o in ORIGENES},
}
# los valores de dificultad que usa cada nivel (son datos de entrada, no del modelo)
NL_NIVEL = {f"{int(r.rarity_orden)}|{int(r.wear_orden)}|{int(r.stattrak)}|{int(r.nivel)}": round(float(r.neg_log_p_item), 6)
            for r in grilla[grilla["origen"] == "caja"].itertuples()}


def formula_ridge(fila):
    """La misma cuenta que hace la página, para comprobar que da lo mismo que scikit-learn."""
    y = FORMULA["b"]
    for (m, s, c), x in zip(FORMULA["num"], (fila["rarity_orden"], fila["wear_orden"])):
        y += c * (x - m) / s
    nl = fila["neg_log_p_item"]
    falta = pd.isna(nl)
    for x, m, s, c in zip((FORMULA["nl"]["med"] if falta else nl, float(falta)), FORMULA["nl"]["m"], FORMULA["nl"]["s"], FORMULA["nl"]["c"]):
        y += c * (x - m) / s
    y += FORMULA["st"] * fila["stattrak"] + FORMULA["sv"] * fila["souvenir"]
    y += FORMULA["arma"][fila["weapon"]] + FORMULA["origen"][fila["origen"]]
    return min(100.0, max(0.0, y))


assert len(FORMULA["nl"]["c"]) == 2, "se esperaba la dificultad y su indicador de faltante"
pred_r = ridge_app.predict(grilla[COLUMNAS_X])
assert np.allclose([formula_ridge(f) for f in grilla.to_dict(orient="records")], pred_r, atol=1e-6), \
    "la fórmula escrita no da lo mismo que scikit-learn"
N_COEF = len(coef_r) + 1
print(f"fórmula de la regresión lineal: {N_COEF} números, coincide con scikit-learn en los {len(grilla)} casos")

# ================================================================ 8 · hipótesis de la Entrega 2, recalculadas
base = df[(df["stattrak"] == 0) & (df["souvenir"] == 0)]
sv = df[df["souvenir"] == 1].groupby(["skin_id", "wear_orden"])["q"].first()
nr = base.groupby(["skin_id", "wear_orden"])["q"].first()
pares_sv = pd.concat([sv.rename("sv"), nr.rename("nr")], axis=1, join="inner")
dif_sv = pares_sv["sv"] - pares_sv["nr"]
H2 = {"pares": len(pares_sv), "d": float(dif_sv.mean() / dif_sv.std(ddof=1)), "mediana": float(dif_sv.median())}
rhos = []
for _, g in base.groupby("skin_id"):
    if g["wear_orden"].nunique() >= 3:
        rhos.append(g["wear_orden"].corr(g["q"], method="spearman"))
rhos = pd.Series(rhos).dropna()
# El notebook del EDA (Entrega 2) informa el PROMEDIO de las correlaciones por skin; acá van
# las dos, para que el número de la página se pueda comparar con el del notebook.
H3 = {"skins": len(rhos), "rho": float(rhos.median()), "media": float(rhos.mean()),
      "baja": float(100 * (rhos < 0).mean())}
caja = df[df["origen"] == "caja"]
r4, n4 = [], []
for _, g in caja.groupby("rarity"):
    if len(g) > 30:
        r4.append(g["neg_log_p_item"].corr(g["q"], method="spearman")); n4.append(len(g))
H4 = {"r": float(np.average(r4, weights=n4)), "filas": int(sum(n4))}
MEDIANA_RAREZA = df.groupby("rarity_orden")["q"].median().to_dict()
SPEARMAN_RAREZA = float(df["rarity_orden"].corr(df["q"], method="spearman"))

# ================================================================ 8b · el Armory, en números
# El 02/10/2024 Valve lanzó el Armory: un pase pago que da créditos para canjear skins de
# colecciones nuevas. Esas colecciones no tienen caja, así que el pipeline las clasifica como
# «drop de mapa», igual que las colecciones de mapa viejas; pero se consiguen mucho más fácil.
ARMORY_DESDE = pd.Timestamp("2024-10-02")
mapa = df[(df["origen"] == "drop de mapa") & df["collection_release"].notna()].copy()
mapa["armory"] = mapa["collection_release"] >= ARMORY_DESDE
pre, arm = mapa[~mapa["armory"]], mapa[mapa["armory"]]
ERA_PRE, ERA_ARM = "Antes del Armory", "Armory (desde oct. 2024)"
ARM = {"col_pre": int(pre["collection"].nunique()), "col_arm": int(arm["collection"].nunique()),
       "usd_pre": float(pre["sell_price_usd"].median()), "usd_arm": float(arm["sell_price_usd"].median()),
       "q_pre": float(pre["q"].median()), "q_arm": float(arm["q"].median()),
       "skins_arm": len(arm)}
ARM["veces"] = ARM["usd_pre"] / ARM["usd_arm"]
col_mapa = (mapa.groupby("collection")
                .agg(fecha=("collection_release", "first"), skins=("q", "size"), med=("q", "median"),
                     usd=("sell_price_usd", "median"), armory=("armory", "first"))
                .reset_index())
col_mapa["era"] = np.where(col_mapa["armory"], ERA_ARM, ERA_PRE)
col_mapa["fecha"] = col_mapa["fecha"].dt.strftime("%Y-%m-%d")
col_mapa["nombre"] = col_mapa["collection"].str.replace("The ", "", regex=False).str.replace(" Collection", "", regex=False)
# La misma rareza, antes y después: así se ve que no es que el Armory traiga rarezas más bajas.
rar_mapa = mapa.groupby(["rarity_orden", "armory"])["sell_price_usd"].median().unstack()
RAR_MAPA = [{"rareza": RAREZAS[int(r)], "antes": float(fila[False]), "armory": float(fila[True]),
             "veces": float(fila[False] / fila[True])}
            for r, fila in rar_mapa.dropna().iterrows()]
MILSPEC_MAPA = next(x for x in RAR_MAPA if x["rareza"] == "Mil-Spec Grade")
# Qué parte de la prueba es del Armory, y cuántas colecciones del Armory vio el entrenamiento.
test_arm = test["collection_release"] >= ARMORY_DESDE
mapa_arm_te = test_arm & (test["origen"] == "drop de mapa")
ARM["test_col_arm"] = int(test.loc[test_arm, "collection"].nunique())
ARM["train_col_arm"] = int(train.loc[train["collection_release"] >= ARMORY_DESDE, "collection"].nunique())
ARM["test_pct_mapa_arm"] = float(100 * mapa_arm_te.mean())
ARM["sesgo_mapa_arm"] = float(test.loc[mapa_arm_te, "error"].mean())
ARM["sesgo_resto"] = float(test.loc[~mapa_arm_te, "error"].mean())
ARM["caro_mapa_arm"] = float(100 * (test.loc[mapa_arm_te, "error"] > 0).mean())
print("armory:", {k: round(v, 2) for k, v in ARM.items()})

# ================================================================ 8c · la prueba de la causa: ¿y si el modelo supiera qué es el Armory?
# Mismo procedimiento, con un corte más tarde que deja colecciones del Armory en el entrenamiento
# (prueba: colecciones desde 03/2025), y un origen propio para ellas: «drop del Armory».
# Hiperparámetros: los que eligió la grilla de cada modelo con pedazos temporales de ese
# entrenamiento (exp_modelos_armory.py, en el repositorio). El modelo entregado NO se toca:
# esto explica el error de la prueba, no lo corrige.
def es_armory(origen, fecha):
    return origen == "drop de mapa" and pd.notna(fecha) and fecha >= ARMORY_DESDE


print("experimento: el Armory como origen propio…")
CORTE_EXP = pd.Timestamp("2025-03-01")
PARAMS_EXP = {
    "sin": {"Regresión lineal (Ridge)": Ridge(alpha=10),
            "Árbol de decisión": DecisionTreeRegressor(min_samples_leaf=40, random_state=SEMILLA),
            "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=SEMILLA, n_jobs=-1),
            "Gradient Boosting": HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=40,
                                                               random_state=SEMILLA)},
    "con": {"Regresión lineal (Ridge)": Ridge(alpha=10),
            "Árbol de decisión": DecisionTreeRegressor(min_samples_leaf=20, random_state=SEMILLA),
            "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=SEMILLA, n_jobs=-1),
            "Gradient Boosting": HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=40,
                                                               random_state=SEMILLA)},
}
fe_ = df["collection_release"].fillna(pd.Timestamp("2000-01-01"))
tr_e, te_e = df[fe_ < CORTE_EXP].reset_index(drop=True), df[fe_ >= CORTE_EXP].reset_index(drop=True)
M_e = np.sort(tr_e["sell_price_usd"].to_numpy(float))
ytr_e, yte_e = percentil_contra(tr_e["sell_price_usd"], M_e), percentil_contra(te_e["sell_price_usd"], M_e)
arm_tr_e = np.array([es_armory(o, f) for o, f in zip(tr_e["origen"], tr_e["collection_release"])])
arm_te_e = np.array([es_armory(o, f) for o, f in zip(te_e["origen"], te_e["collection_release"])])
mapa_e = (te_e["origen"] == "drop de mapa").to_numpy()
DEFENSA = []
for version, modelos in PARAMS_EXP.items():
    Xa, Xb = tr_e[COLUMNAS_X].copy(), te_e[COLUMNAS_X].copy()
    if version == "con":
        Xa["origen"] = np.where(arm_tr_e, "drop del Armory", tr_e["origen"])
        Xb["origen"] = np.where(arm_te_e, "drop del Armory", te_e["origen"])
    for nombre, reg in modelos.items():
        p = armar_modelo(reg).fit(Xa, ytr_e).predict(Xb)
        DEFENSA.append({"modelo": nombre, "version": version, "r2": float(r2_score(yte_e, p)),
                        "sesgo_mapa": float(np.mean(p[mapa_e] - yte_e[mapa_e])),
                        "arriba": float(100 * np.mean(p > yte_e))})
DEFENSA_TABLA = float(r2_score(yte_e, MediaPorGrupo(("rarity", "wear")).fit(tr_e, ytr_e).predict(te_e)))
DEF_INFO = {"train": len(tr_e), "test": len(te_e), "col_test": int(te_e["collection"].nunique()),
            "arm_train": int(arm_tr_e.sum()), "arm_test": int(arm_te_e.sum())}
print("  ", [(d["modelo"][:6], d["version"], round(d["r2"], 3)) for d in DEFENSA], "tabla", round(DEFENSA_TABLA, 3))

# La escalera de precios en dólares dentro de las cajas (la comparación más limpia: mismo origen).
CAJA_RAREZA = []
for r_i in range(2, 6):
    g = caja[caja["rarity_orden"] == r_i]["sell_price_usd"]
    CAJA_RAREZA.append({"rareza": RAREZAS[r_i], "med": float(g.median()), "p25": float(g.quantile(0.25)),
                        "p75": float(g.quantile(0.75)), "skins": len(g)})
for a, b in zip(CAJA_RAREZA, CAJA_RAREZA[1:]):
    b["paso"] = b["med"] / a["med"]

# ================================================================ 9 · verificación contra el notebook
VERIFICADO = "no se encontró el archivo de metadatos del notebook"
ruta_meta = Path(args.verificar)
if ruta_meta.exists():
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    if meta.get("snapshot") == SNAPSHOT:
        for k, v in {"r2_cv": R2_CV, "r2_prueba": R2_TEST, "mae_prueba": MAE_TEST, "sesgo_prueba": SESGO_TEST}.items():
            assert abs(meta[k] - v) < 6e-3, f"{k}: notebook {meta[k]} contra aplicación {v}"
        VERIFICADO = (f"coincide con el notebook: R² {meta['r2_cv']:.3f} en validación y "
                      f"{meta['r2_prueba']:.3f} en la prueba")
    else:
        VERIFICADO = f"datos de otra fecha que el notebook ({meta.get('snapshot')}): números nuevos"
print("verificación:", VERIFICADO)

# ================================================================ 10 · datos que viajan con la página
lado = np.r_[np.zeros(len(train), int), np.ones(len(test), int)]
todo = pd.concat([train, test], ignore_index=True)
todo["q"] = percentil_contra(todo["sell_price_usd"], MERCADO_TOTAL)
todo["ph"] = percentil_a_dolares(todo["predicho"].to_numpy())
VAR = np.select([todo["stattrak"] == 1, todo["souvenir"] == 1], ["StatTrak", "Souvenir"], "Normal")
# Corrimiento en días, sólo para dibujar: los skins de una colección comparten fecha y, sin esto,
# en Explorá quedarían todos apilados en una raya vertical.
JITTER = np.random.default_rng(SEMILLA).integers(-50, 51, len(todo))
SKINS = pd.DataFrame({
    "j": JITTER,
    "n": todo["market_hash_name"], "a": todo["weapon"], "r": todo["rarity_orden"].astype(int),
    "w": todo["wear_orden"].astype(int), "v": VAR, "o": todo["origen"],
    "c": todo["collection"].fillna("sin colección"),
    "f": todo["collection_release"].dt.strftime("%Y-%m-%d").fillna(""),
    "p": todo["sell_price_usd"].round(2), "q": todo["q"].round(1),
    "y": todo["percentil"].round(1), "yh": todo["predicho"].round(1), "ph": todo["ph"].round(2),
    "l": lado,
})

# Imágenes. Cada fila apunta a un índice de IMAGENES, o -1. Dos fuentes posibles:
#  1) imagenes.py (versión 1654): data/imagenes_<snapshot>.json, nombre de mercado → archivo en docs/img/
#  2) descargar_estaticos.py (versión 1817): una imagen por diseño (skin_id). Si la miniatura está en
#     docs/img/skins/<skin_id>.png se usa la copia local; si no, la imagen oficial en los servidores de Steam.
imagenes_ruta = ruta.with_name(f"imagenes_{SNAPSHOT}.json")
catalogo_ruta = RAIZ / "data" / "estaticos" / "catalogo.json"
if imagenes_ruta.exists():
    IMAGEN_DE = json.loads(imagenes_ruta.read_text(encoding="utf-8"))
elif catalogo_ruta.exists():
    cat_disenos = json.loads(catalogo_ruta.read_text(encoding="utf-8"))["disenos"]
    def img_diseno(sid):
        if (Path(args.salida).parent / "img" / "skins" / f"{sid}.png").exists():
            return f"skins/{sid}.png"
        url = cat_disenos.get(sid, {}).get("imagen", "")
        return url + ("/256fx256f" if "steamstatic" in url else "") if url else None
    img_sid = {sid: img_diseno(sid) for sid in df["skin_id"].unique()}
    IMAGEN_DE = {nom: img_sid[sid] for nom, sid in zip(df["market_hash_name"], df["skin_id"]) if img_sid.get(sid)}
else:
    IMAGEN_DE = {}
IMAGENES = sorted(set(IMAGEN_DE.values()))
indice_img = {a: i for i, a in enumerate(IMAGENES)}
img_de_fila = lambda nombre: indice_img.get(IMAGEN_DE.get(nombre), -1)
SKINS["i"] = [img_de_fila(x) for x in SKINS["n"]]
print(f"imágenes: {len(IMAGENES)} archivos, {int((SKINS['i'] >= 0).sum())} de {len(SKINS)} skins con imagen"
      if IMAGENES else "imágenes: no hay (correr descargar_estaticos.py o imagenes.py)")

# Diseños para «Fabricá un skin»: cada uno con su arma, rareza, origen y una imagen por exterior.
DISENOS = []
for sid, g in df.groupby("skin_id"):
    normal = g[g["souvenir"] == 0]
    imgs = []
    for w in range(5):
        filas = g[g["wear_orden"] == w]
        imgs.append(img_de_fila(filas["market_hash_name"].iloc[0]) if len(filas) else -1)
    DISENOS.append({"a": ARMAS.index(g["weapon"].iloc[0]), "r": int(g["rarity_orden"].iloc[0]),
                    "o": ORIGENES.index((normal if len(normal) else g)["origen"].iloc[0]),
                    "d": str(g["pattern"].iloc[0]), "c": str(g["collection"].iloc[0]).replace("The ", "").replace(" Collection", ""),
                    "im": imgs, "pop": int(len(g))})
DISENOS.sort(key=lambda d: (d["a"], d["r"], -d["pop"], d["d"]))

# Portada: seis AK-47 reales, del más barato al más caro (índices en SKINS).
ak = todo[todo["weapon"] == "AK-47"].sort_values("sell_price_usd")
GANCHO = [int(ak.index[int(round(k * (len(ak) - 1)))]) for k in (0, 0.25, 0.5, 0.75, 0.93, 1)]

# El modelo, paso 2: cada colección en el tiempo, de qué lado de la partición quedó.
COL_ARM, COL_RESTO = "Colecciones de drop de mapa del Armory", "Las demás colecciones"
PARTICION = []
for c, g in todo[todo["collection_release"].notna()].groupby("collection"):
    normales = g[(g["stattrak"] == 0) & (g["souvenir"] == 0)]
    o = (normales if len(normales) else g)["origen"].mode().iloc[0]
    f = g["collection_release"].iloc[0]
    PARTICION.append({"nombre": re.sub(r"^The |\s*Collection$", "", c), "f": f"{f:%Y-%m-%d}", "origen": o,
                      "filas": len(g), "lado": "prueba" if c in COLECCIONES_TEST else "entrenamiento",
                      "grupo": COL_ARM if es_armory(o, f) else COL_RESTO})

# ---------------------------------------------------------------- CS2 en dos minutos: ejemplos reales
# (las demos de la versión 1817, con los cuatro orígenes del modelo y el Armory marcado por fecha)
# Cada ejemplo apunta a una fila de SKINS (índice k), así la página toma de ahí nombre, precio e imagen.
todo["var"] = np.select([todo["stattrak"] == 1, todo["souvenir"] == 1], [1, 2], 0)
todo["armory"] = [es_armory(o, f) for o, f in zip(todo["origen"], todo["collection_release"])]
CAT_EST = (json.loads(catalogo_ruta.read_text(encoding="utf-8")) if catalogo_ruta.exists()
           else {"disenos": {}, "colecciones": {}, "cajas": {}})
corto_col = lambda c: re.sub(r"^The |\s*Collection$", "", str(c))


def src_estatico(url, carpeta):
    """Una imagen del catálogo: la copia local de docs/img/<carpeta>/ si existe, si no la de internet."""
    if not url:
        return ""
    local = Path(args.salida).parent / "img" / carpeta / url.rsplit("/", 1)[-1]
    if local.exists():
        return f"img/{carpeta}/{local.name}"
    return url + ("/256fx256f" if "steamstatic" in url else "")


# 1 y 3 · el nombre y el exterior: un AK-47 | Asiimov (o, si no está, el StatTrak con más exteriores a la venta)
ej = todo[(todo["weapon"] == "AK-47") & (todo["pattern"] == "Asiimov")]
if ej.empty or not (ej["var"] == 1).any():
    st_ = todo[todo["var"] == 1]
    ej = todo[todo["skin_id"] == st_.groupby("skin_id")["wear_orden"].nunique().idxmax()]
ext_sid = ej["skin_id"].iloc[0]


def fila_de(datos, w, v):
    g = datos[(datos["wear_orden"] == w) & (datos["var"] == v)].sort_values("sell_price_usd")
    return (int(g.index[0]), float(g["sell_price_usd"].iloc[0])) if len(g) else (-1, None)


EXTERIOR = {"nombre": f"{ej['weapon'].iloc[0]} | {ej['pattern'].iloc[0]}", "arma": ej["weapon"].iloc[0],
            "pintura": str(ej["pattern"].iloc[0]), "r": int(ej["rarity_orden"].iloc[0]),
            "col": corto_col(ej["collection"].iloc[0]),
            "k": [fila_de(ej, w, 0)[0] for w in range(5)], "k_st": [fila_de(ej, w, 1)[0] for w in range(5)],
            "normal": [fila_de(ej, w, 0)[1] for w in range(5)], "stattrak": [fila_de(ej, w, 1)[1] for w in range(5)],
            "float": [CAT_EST["disenos"].get(ext_sid, {}).get("float_min"), CAT_EST["disenos"].get(ext_sid, {}).get("float_max")]}

# StatTrak y Souvenir contra la versión normal del mismo skin y exterior
par = todo.pivot_table(index=["skin_id", "wear_orden"], columns="var", values="sell_price_usd", aggfunc="min")
VECES_ST = float((par[1] / par[0]).dropna().median())
VECES_SV = float((par[2] / par[0]).dropna().median())

# 2 · la escalera de rarezas: precio típico, probabilidad en una caja y un ejemplo conocido
PROB_CAJA = {2: 79.92, 3: 15.98, 4: 3.20, 5: 0.64}   # las que publica Valve para las cajas
POPULARES = ["AK-47", "AWP", "M4A4", "M4A1-S", "USP-S", "Glock-18", "Desert Eagle"]
ESCALERA = []
for r_i in range(6):
    g = todo[todo["rarity_orden"] == r_i]
    pop = g[g["weapon"].isin(POPULARES) & (g["var"] == 0)]
    base_ej = pop if len(pop) else g[g["var"] == 0]
    sid = base_ej.groupby("skin_id").size().sort_values(ascending=False).index[0]
    k = int(base_ej[base_ej["skin_id"] == sid].sort_values("wear_orden").index[0])
    ESCALERA.append({"usd": round(float(g["sell_price_usd"].median()), 2), "q": round(float(g["q"].median())),
                     "skins": len(g), "prob": PROB_CAJA.get(r_i), "k": k})

# 4 · cómo se consigue: los cuatro orígenes del modelo; «drop de mapa», partido por época
ORIGENES_INTRO = []
for o in ORIGENES:
    g = todo[todo["origen"] == o]
    col_ej = g["collection"].value_counts().index[0]
    item = {"skins": len(g), "colecciones": int(g["collection"].nunique()), "usd": round(float(g["sell_price_usd"].median()), 2),
            "icono": src_estatico(CAT_EST["colecciones"].get(col_ej, {}).get("icono", ""), "colecciones"), "ej": corto_col(col_ej)}
    if o == "drop de mapa":
        item["epocas"] = [{"skins": len(gg), "colecciones": int(gg["collection"].nunique()),
                           "usd": round(float(gg["sell_price_usd"].median()), 2)}
                          for gg in (g[~g["armory"]], g[g["armory"]])]
    ORIGENES_INTRO.append(item)

# 4b · probá abrir una caja: los skins reales de una caja, con su precio en Field-Tested
caja_col = "The Kilowatt Collection" if (todo["collection"] == "The Kilowatt Collection").any() else \
    todo[todo["origen"] == "caja"]["collection"].value_counts().index[0]
g_caja = todo[(todo["collection"] == caja_col) & (todo["var"] == 0)]
caja_nombre = g_caja["caja"].dropna().iloc[0] if "caja" in g_caja and g_caja["caja"].notna().any() else corto_col(caja_col)
CAJA = {"nombre": caja_nombre, "col": corto_col(caja_col),
        "img": src_estatico(CAT_EST["cajas"].get(caja_nombre, {}).get("imagen", ""), "cajas"), "items": []}
for sid, gg in g_caja.groupby("skin_id"):
    ft = gg[gg["wear_orden"] == 2]
    fila = (ft if len(ft) else gg).sort_values("sell_price_usd").iloc[0]
    CAJA["items"].append({"k": int(fila.name), "r": int(fila["rarity_orden"]), "p": round(float(fila["sell_price_usd"]), 2),
                          "w": int(fila["wear_orden"])})

# 5 · el puesto: siete skins reales repartidos por la fila del mercado
FILA_MERCADO = [int((todo["q"] - k).abs().idxmin()) for k in (3, 15, 30, 50, 70, 85, 97)]

INTRO = {"exterior": EXTERIOR, "escalera": ESCALERA, "origenes": ORIGENES_INTRO, "caja": CAJA, "fila": FILA_MERCADO}
SKINS = SKINS.to_dict(orient="records")

# ================================================================ 11 · gráficos (Altair)
# Regla de todos los gráficos: cada uno muestra UNA idea, y esa idea está escrita adentro del
# gráfico (un rótulo, una flecha, un número), no sólo en el texto de al lado.
def n(x, d=0):
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x):
    return f"{n(x)}\u00a0%"   # espacio duro: el % no queda solo en otra línea


def usd(x):
    return "USD " + (n(x, 2) if x < 100 else n(x, 0))


COL_RAREZA = ["#9fb3cc", "#5e98d9", "#4b69ff", "#8847ff", "#d32ce6", "#eb4b4b"]   # los colores de CS2
# Un color, un rol, en toda la página (también en el CSS y en el glosario, que salen de acá):
# rareza → COL_RAREZA · modelo elegido y su predicción → AMBAR · origen → COL_ORIGEN ·
# Armory → ARMORY · todos juntos, el resto, pisos y referencias → GRIS.
# Viejas contra nuevas (validación/prueba, aprendió/encontró) NO va con color sino con forma:
# círculo lleno = viejas, triángulo hueco = nuevas.
COL_ORIGEN = ["#26b5c7", "#9a724a", "#cea8d5", "#6f6a64"]   # caja, drop de mapa, paquete souvenir, sin identificar
GRIS, ARMORY, AMBAR = "#919191", "#ffab91", "#e8962e"   # gris neutro: ΔE ≥ 15 contra Industrial y «sin identificar»
FORMA_VIEJAS, FORMA_NUEVAS = "circle", "triangle-left"
esc_rareza = alt.Scale(domain=RAREZAS, range=COL_RAREZA)
esc_origen = alt.Scale(domain=ORIGENES, range=COL_ORIGEN)
esc_era = alt.Scale(domain=[ERA_PRE, ERA_ARM], range=[GRIS, ARMORY])
skins = alt.NamedData("skins")
rareza_nombre = "['Consumer Grade','Industrial Grade','Mil-Spec Grade','Restricted','Classified','Covert'][datum.r]"
corta = "split(datum.label, ' ')[0]"
EJE_USD = "'USD ' + format(datum.value, ',~r')"
EJE_PUESTO = "datum.value == 0 ? '0 · el más barato' : datum.value == 100 ? 'el más caro · 100' : datum.value"
datos = lambda filas: alt.Data(values=filas)

# 11.1 portada: no lleva gráfico de Altair. Los seis AK-47 (GANCHO) los arma la plantilla con
# las fotos y los precios de APP.SKINS, como en la versión 1817.

# 11.2 los datos: cuántos skins hay en cada rango de precio
mediana_precio = float(np.median(df["sell_price_usd"]))
p90_precio = float(np.quantile(df["sell_price_usd"], 0.9))
base_hist = (alt.Chart(skins).transform_calculate(lp="log(datum.p)/LN10")
             .transform_bin("lp_bin", "lp", bin=alt.Bin(step=0.1, extent=[-1.6, 3.4])))
hist = (base_hist.mark_bar(color=GRIS, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(x=alt.X("lp_bin:Q", bin="binned", scale=alt.Scale(domain=[-1.6, 3.4]),
                        axis=alt.Axis(values=[-1, 0, 1, 2, 3], labelExpr="'USD ' + format(pow(10, datum.value), ',~r')"),
                        title="Precio en Steam (cada marca vale 10 veces la anterior)"),
                x2="lp_bin_end:Q",
                y=alt.Y("count():Q", title="Cantidad de skins"),
                tooltip=[alt.Tooltip("count():Q", title="Skins")]))
marcas = [{"lp": float(np.log10(mediana_precio)), "t": f"La mitad cuesta menos de {usd(mediana_precio)}", "a": "right", "dx": -6},
          {"lp": float(np.log10(p90_precio)), "t": f"1 de cada 10 cuesta más de {usd(p90_precio)}", "a": "left", "dx": 6}]
# sin color propio: toman el del texto del tema (--ink), así no se leen como «modelo»
reglas = alt.Chart(datos(marcas)).mark_rule(strokeDash=[4, 3], strokeWidth=1.5).encode(x="lp:Q")
rotulos = [alt.Chart(datos([m])).mark_text(align=m["a"], dx=m["dx"], fontWeight=600)
           .encode(x="lp:Q", y=alt.value(12), text="t:N") for m in marcas]
graf_datos = alt.layer(hist, reglas, *rotulos).properties(width="container", height=240)

# 11.2b por qué tanta diferencia: la escalera de la rareza en dólares, dentro de las cajas
y_caja = alt.Y("rareza:N", sort=[r["rareza"] for r in CAJA_RAREZA][::-1], title=None,
               axis=alt.Axis(labelExpr=corta, labelFontSize=12.5, labelFontWeight=600, ticks=False, domain=False))
x_usd = alt.X("p25:Q", scale=alt.Scale(type="log", domain=[0.1, 600]), title="Precio en Steam (escala de 10 en 10)",
              axis=alt.Axis(values=[0.1, 1, 10, 100], labelExpr=EJE_USD))
for r in CAJA_RAREZA:
    r["t"] = f"mitad: {usd(r['med'])}"
    r["paso_t"] = f"× {n(r['paso'], 0)} que la rareza de abajo" if "paso" in r else "la rareza más barata de las cajas"
b_cr = alt.Chart(datos(CAJA_RAREZA))
graf_precios = alt.layer(
    b_cr.mark_bar(height=16, opacity=0.35, cornerRadius=3).encode(
        x=x_usd, x2="p75:Q", y=y_caja,
        color=alt.Color("rareza:N", scale=esc_rareza, legend=None),
        tooltip=[alt.Tooltip("rareza:N"), alt.Tooltip("p25:Q", title="Un cuarto cuesta menos de", format=",.2f"),
                 alt.Tooltip("med:Q", title="La mitad cuesta menos de", format=",.2f"),
                 alt.Tooltip("p75:Q", title="Tres cuartos cuestan menos de", format=",.2f"), alt.Tooltip("skins:Q")]),
    b_cr.mark_point(filled=True, size=200, opacity=1, stroke="#ffffff", strokeWidth=1.5).encode(
        x="med:Q", y=y_caja, color=alt.Color("rareza:N", scale=esc_rareza, legend=None)),
    b_cr.mark_text(dy=-18, fontWeight=600, fontSize=12).encode(x="med:Q", y=y_caja, text="t:N"),
    b_cr.mark_text(align="left", dx=10, color=GRIS, fontSize=11.5).encode(x="p75:Q", y=y_caja, text="paso_t:N"),
).properties(width="container", height=4 * 58)

# 11.3 lo que encontramos (1): la rareza ordena el precio dentro de cada origen
med_ro = (df.groupby(["origen", "rarity_orden"])["q"].agg(["median", "size"]).reset_index()
            .query("size >= 20"))
med_ro["rareza"] = med_ro["rarity_orden"].map(dict(enumerate(RAREZAS)))
med_todos = df.groupby("rarity_orden")["q"].agg(["median", "size"]).reset_index()
med_todos["rareza"] = med_todos["rarity_orden"].map(dict(enumerate(RAREZAS)))
med_todos["origen"] = "todos juntos"
x_r = alt.X("rareza:N", sort=RAREZAS, title="Rareza (de la más común a la más rara →)",
            axis=alt.Axis(labelAngle=0, labelExpr=corta, labelFontWeight=600))
y_r = alt.Y("median:Q", scale=alt.Scale(domain=[0, 100]), title="Puesto típico en el mercado (0 a 100)")
tip_r = [alt.Tooltip("origen:N"), alt.Tooltip("rareza:N"), alt.Tooltip("median:Q", title="Puesto típico", format=".0f"),
         alt.Tooltip("size:Q", title="Skins")]
# En tres pasos (de la versión 1817): 1 todos juntos · 2 separados por origen · 3 «drop de mapa»
# partido por época. Arranca en el paso 3, que muestra todo junto: así el gráfico dice lo suyo sin
# tocar nada (regla del instructivo) y los botones cuentan la historia desde el principio.
mapa_df = df[df["origen"] == "drop de mapa"]
mapa_arm_df = mapa_df["collection_release"].notna() & (mapa_df["collection_release"] >= ARMORY_DESDE)
series_h = [med_todos.assign(k="todos")]
series_h += [med_ro[med_ro["origen"] == o].assign(k=o) for o in ORIGENES]
for clave, parte in (("clasico", mapa_df[~mapa_arm_df]), ("armory", mapa_df[mapa_arm_df])):
    m = parte.groupby("rarity_orden")["q"].agg(["median", "size"]).reset_index().query("size >= 20")
    m["rareza"] = m["rarity_orden"].map(dict(enumerate(RAREZAS)))
    series_h.append(m.assign(k=clave))
med_h = pd.concat([m[["rarity_orden", "rareza", "median", "size", "k"]] for m in series_h], ignore_index=True)
SERIE = {"todos": "Todos juntos", "caja": "Caja", "drop de mapa": "Drop de mapa", "paquete souvenir": "Paquete souvenir",
         "sin identificar": "Sin identificar", "clasico": "Drop de mapa, antes del Armory", "armory": "Drop de mapa del Armory"}
COLOR_SERIE = {"todos": GRIS, "caja": COL_ORIGEN[0], "drop de mapa": COL_ORIGEN[1], "paquete souvenir": COL_ORIGEN[2],
               "sin identificar": COL_ORIGEN[3], "clasico": COL_ORIGEN[1], "armory": ARMORY}
med_h["serie"] = med_h["k"].map(SERIE)
p_paso = alt.param(name="paso", value=3)
OTROS = "(datum.k == 'caja' || datum.k == 'paquete souvenir' || datum.k == 'sin identificar')"
OPACIDAD_H = ("paso == 1 ? (datum.k == 'todos' ? 1 : 0)"
              " : paso == 2 ? (datum.k == 'todos' ? 0.25 : (datum.k == 'clasico' || datum.k == 'armory') ? 0 : 1)"
              f" : (datum.k == 'clasico' || datum.k == 'armory') ? 1 : {OTROS} ? 0.35 : 0")
esc_serie = alt.Scale(domain=list(SERIE.values()), range=[COLOR_SERIE[k] for k in SERIE])
b_h = alt.Chart(datos(med_h.to_dict(orient="records"))).transform_calculate(op=OPACIDAD_H).transform_filter("datum.op > 0")
tip_h = [alt.Tooltip("serie:N", title="Grupo"), alt.Tooltip("rareza:N"),
         alt.Tooltip("median:Q", title="Puesto típico", format=".0f"), alt.Tooltip("size:Q", title="Skins")]
color_h = alt.Color("serie:N", scale=esc_serie, legend=None)
lin_h = b_h.mark_line().encode(
    x=x_r, y=y_r, color=color_h, detail="k:N", opacity=alt.Opacity("op:Q", scale=None),
    strokeWidth=alt.condition("datum.k == 'caja' || datum.k == 'clasico' || datum.k == 'armory'", alt.value(4), alt.value(2.5)),
    strokeDash=alt.condition("datum.k == 'todos'", alt.value([6, 4]), alt.value([1, 0])))
pts_h = b_h.mark_point(filled=True, size=80).encode(x=x_r, y=y_r, color=color_h, opacity=alt.Opacity("op:Q", scale=None),
                                                     tooltip=tip_h)
MED_H = {(k, int(r)): float(m) for k, r, m in zip(med_h["k"], med_h["rarity_orden"], med_h["median"])}
med_mil_clas = MED_H.get(("clasico", 2), 60.0)
med_mil_arm = MED_H.get(("armory", 2), 20.0)
NOTAS_H = [
    {"paso": 1, "rareza": RAREZAS[2], "median": MED_H[("todos", 2)], "t": "¿Más raro y más barato?", "dy": 30},
    {"paso": 2, "rareza": RAREZAS[1], "median": 95, "t": "Separados por cómo se consiguen, todas suben", "dy": 0},
    {"paso": 3, "rareza": RAREZAS[2], "median": (med_mil_clas + med_mil_arm) / 2 - 8, "t": "Misma etiqueta, otra época",
     "dy": 0},
]
# sin color propio: el anillo y los textos toman el del texto del tema (--ink)
anillo = (alt.Chart(datos([NOTAS_H[0]])).transform_filter("paso == 1")
          .mark_point(size=900, strokeWidth=2.5).encode(x=x_r, y="median:Q"))
notas_h = alt.layer(*[alt.Chart(datos([nota])).transform_filter(f"paso == {nota['paso']}")
                      .mark_text(fontWeight=700, fontSize=13, dy=nota["dy"], align="right" if nota["paso"] == 3 else "center",
                                 dx=-10 if nota["paso"] == 3 else 0)
                      .encode(x=x_r, y="median:Q", text="t:N") for nota in NOTAS_H])
flecha_h = (alt.Chart(datos([{"rareza": RAREZAS[2], "a": med_mil_arm + 4, "b": med_mil_clas - 4}])).transform_filter("paso == 3")
            .mark_rule(strokeDash=[3, 3], strokeWidth=1.5).encode(x=x_r, y="a:Q", y2="b:Q"))
graf_hallazgo = (lin_h + pts_h + anillo + flecha_h + notas_h).add_params(p_paso).properties(width="container", height=340)
MED_RO = {(o, int(r)): float(m) for o, r, m in zip(med_ro["origen"], med_ro["rarity_orden"], med_ro["median"])}

# 11.3b lo que encontramos (2): el Armory, colección por colección
fin_snap = SNAPSHOT
eras = [{"a": col_mapa["fecha"].min(), "b": ARMORY_DESDE.strftime("%Y-%m-%d"), "y": ARM["q_pre"],
         "t": f"Antes del Armory: puesto típico {n(ARM['q_pre'])} ({usd(ARM['usd_pre'])})", "al": "left", "x": "2015-11-01", "dy": -9},
        {"a": ARMORY_DESDE.strftime("%Y-%m-%d"), "b": fin_snap, "y": ARM["q_arm"],
         "t": f"Armory: puesto típico {n(ARM['q_arm'])} ({usd(ARM['usd_arm'])}) →", "al": "right",
         "x": ARMORY_DESDE.strftime("%Y-%m-%d"), "dy": 4}]
x_f = alt.X("fecha:T", title="Año en que salió la colección", axis=alt.Axis(format="%Y", tickCount=12))
zona_arm = alt.Chart(datos([{"a": ARMORY_DESDE.strftime("%Y-%m-%d"), "b": fin_snap}])) \
    .mark_rect(color=ARMORY, opacity=0.10).encode(x="a:T", x2="b:T")
zona_t = alt.Chart(datos([{"a": ARMORY_DESDE.strftime("%Y-%m-%d"), "t": "Llega el Armory →"}])) \
    .mark_text(align="right", dx=-6, color=ARMORY, fontWeight=700, fontSize=12).encode(x="a:T", y=alt.value(12), text="t:N")
pts_col = (alt.Chart(datos(col_mapa.to_dict(orient="records")))
           .mark_circle(opacity=0.85, stroke="#ffffff", strokeWidth=0.6)
           .encode(x=x_f,
                   y=alt.Y("med:Q", scale=alt.Scale(domain=[0, 100]), title="Puesto típico de sus skins (0 a 100)"),
                   size=alt.Size("skins:Q", scale=alt.Scale(range=[30, 380]), legend=None),
                   color=alt.Color("era:N", scale=esc_era, legend=alt.Legend(orient="top", title=None)),
                   tooltip=[alt.Tooltip("nombre:N", title="Colección"), alt.Tooltip("fecha:T", format="%m/%Y", title="Salió"),
                            alt.Tooltip("med:Q", format=".0f", title="Puesto típico"),
                            alt.Tooltip("usd:Q", format=",.2f", title="Precio típico (USD)"), alt.Tooltip("skins:Q")]))
lin_era = alt.Chart(datos(eras)).mark_rule(strokeDash=[5, 4], strokeWidth=2).encode(
    x="a:T", x2="b:T", y="y:Q", color=alt.Color("t:N", scale=alt.Scale(range=[GRIS, ARMORY]), legend=None))
txt_era = [alt.Chart(datos([e])).mark_text(align=e["al"], dx=-8 if i else 0, dy=e["dy"], fontWeight=700, fontSize=12,
                                            color=GRIS if i == 0 else ARMORY)
           .encode(x="x:T", y="y:Q", text="t:N") for i, e in enumerate(eras)]
graf_armory_tiempo = alt.layer(zona_arm, zona_t, lin_era, pts_col, *txt_era).properties(width="container", height=300)

# 11.3c lo que encontramos (3): la misma rareza, antes y con el Armory
for r in RAR_MAPA:
    r["t"] = f"{n(r['veces'], 0)} veces más barato"
y_rm = alt.Y("rareza:N", sort=[r["rareza"] for r in RAR_MAPA][::-1], title=None,
             axis=alt.Axis(labelExpr=corta, labelFontSize=12.5, labelFontWeight=600, ticks=False, domain=False))
x_rm = alt.Scale(type="log", domain=[0.02, 2000])
largo_rm = [{"rareza": r["rareza"], "era": e, "usd": r[k]} for r in RAR_MAPA for e, k in ((ERA_PRE, "antes"), (ERA_ARM, "armory"))]
b_rm = alt.Chart(datos(RAR_MAPA))
graf_armory_rareza = alt.layer(
    b_rm.mark_rule(strokeWidth=3, color=GRIS, opacity=0.5).encode(
        x=alt.X("armory:Q", scale=x_rm, title="Precio típico de un skin de drop de mapa (escala de 10 en 10)",
                axis=alt.Axis(values=[0.1, 1, 10, 100, 1000], labelExpr=EJE_USD)), x2="antes:Q", y=y_rm),
    alt.Chart(datos(largo_rm)).mark_point(filled=True, size=190, opacity=1).encode(
        x="usd:Q", y=y_rm, color=alt.Color("era:N", scale=esc_era, legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("rareza:N"), alt.Tooltip("era:N"), alt.Tooltip("usd:Q", title="Precio típico (USD)", format=",.2f")]),
    b_rm.mark_text(align="left", dx=12, fontWeight=700, color=ARMORY, fontSize=12).encode(x="antes:Q", y=y_rm, text="t:N"),
).properties(width="container", height=len(RAR_MAPA) * 46)

# 11.4 el modelo: validación contra prueba. Color = qué forma de predecir (el elegido en ámbar,
# las referencias en gris); forma = colecciones viejas (círculo lleno) o nuevas (triángulo hueco).
MOMENTOS = ["En colecciones viejas que no vio (validación)", f"En las {test['collection'].nunique()} colecciones más nuevas (prueba)"]
prueba_largo = []
for r in PRUEBA:
    tipo = "elegido" if r["modelo"] == ELEGIDO else "referencia"
    prueba_largo += [{"modelo": r["modelo"], "momento": MOMENTOS[0], "r2": r["validacion"], "orden": 0, "tipo": tipo},
                     {"modelo": r["modelo"], "momento": MOMENTOS[1], "r2": r["prueba"], "orden": 1, "tipo": tipo}]
orden_mod = [r["modelo"] for r in PRUEBA][::-1]
col_tipo = alt.Color("tipo:N", scale=alt.Scale(domain=["elegido", "referencia"], range=[AMBAR, GRIS]), legend=None)


def puntos_viejas_nuevas(base, campo, dominio, x, y, color, tooltip):
    """Círculo lleno = viejas, triángulo hueco = nuevas, con una leyenda de formas que se ve igual.

    La capa de abajo dibuja las dos formas huecas y lleva la leyenda; la de arriba rellena los
    círculos con su color. Vega-Lite dibuja huecos todos los símbolos de la leyenda de una marca
    hueca: la página rellena el del círculo (ver «usermeta.viejas» en la plantilla)."""
    leyenda = alt.Legend(orient="top", title=None, direction="vertical", symbolStrokeColor=GRIS, symbolStrokeWidth=2)
    formas = base.mark_point(size=170, strokeWidth=2.5, opacity=1).encode(
        x=x, y=y, color=color,
        shape=alt.Shape(f"{campo}:N", scale=alt.Scale(domain=dominio, range=[FORMA_VIEJAS, FORMA_NUEVAS]), legend=leyenda),
        tooltip=tooltip)
    # filled=True: el relleno sale del canal color, sin tocar la escala de relleno de la leyenda
    llenos = base.transform_filter(f"datum.{campo} == '{dominio[0]}'").mark_point(
        shape=FORMA_VIEJAS, filled=True, size=170, strokeWidth=2.5, opacity=1).encode(x=x, y=y, color=color, tooltip=tooltip)
    return formas + llenos


b_m = alt.Chart(datos(prueba_largo))
lineas_m = b_m.mark_line(strokeWidth=2, opacity=0.6).encode(
    x=alt.X("r2:Q", scale=alt.Scale(domain=[-0.2, 0.7]), title="R²: cuánto del orden de precios explica (0 = nada, 1 = todo)"),
    y=alt.Y("modelo:N", sort=orden_mod, title=None, axis=alt.Axis(labelFontWeight=600, labelLimit=220)),
    color=col_tipo, detail="modelo:N", order="orden:Q")
pts_m = puntos_viejas_nuevas(
    b_m, "momento", MOMENTOS, "r2:Q", alt.Y("modelo:N", sort=orden_mod), col_tipo,
    [alt.Tooltip("modelo:N"), alt.Tooltip("momento:N"), alt.Tooltip("r2:Q", title="R²", format=".3f")])
txt_m = b_m.mark_text(dy=-15, fontWeight=600).encode(x="r2:Q", y=alt.Y("modelo:N", sort=orden_mod),
                                                     text=alt.Text("r2:Q", format=".2f"))
cero = alt.Chart(datos([{"x": 0}])).mark_rule(color=GRIS, strokeDash=[2, 3]).encode(x="x:Q")
graf_modelo = alt.layer(cero, lineas_m, pts_m, txt_m).properties(width="container", height=220,
                                                               usermeta={"viejas": MOMENTOS[0]})

# 11.4b el modelo: dónde se equivoca en la prueba. El Armory, arriba de la diagonal.
GRUPO_ARM, GRUPO_RESTO = "Skins de drop de mapa del Armory", "Todos los demás de la prueba"
arriba = (alt.Chart(datos([{"x": 0, "y": 0, "y2": 100}, {"x": 100, "y": 100, "y2": 100}]))
          .mark_area(color=ARMORY, opacity=0.07).encode(x="x:Q", y="y:Q", y2="y2:Q"))
diag_e = (alt.Chart(datos([{"x": 0, "y": 0}, {"x": 100, "y": 100}]))
          .mark_line(color=GRIS, strokeDash=[5, 4], strokeWidth=1.5).encode(x="x:Q", y="y:Q"))
# Dos vistas (de la versión 1817): «vista» 1 = la prueba (arranca acá, que es lo que dice el título) y
# 0 = las colecciones conocidas, cada skin predicho por un modelo que no vio su colección. En las
# conocidas no hay Armory, así que todo va en gris. La leyenda la arma la página (cambia con la vista).
p_vista = alt.param(name="vista", value=1)
pts_e = (alt.Chart(skins).transform_filter("datum.l == vista")
         .transform_calculate(grupo=f"datum.o == 'drop de mapa' && datum.f >= '{ARMORY_DESDE:%Y-%m-%d}' ? '{GRUPO_ARM}' : '{GRUPO_RESTO}'",
                              primero="datum.grupo == '" + GRUPO_ARM + "' ? 1 : 0")
         .mark_circle(size=24, opacity=0.65)
         .encode(x=alt.X("y:Q", scale=alt.Scale(domain=[0, 100]), title="Puesto real en el mercado"),
                 y=alt.Y("yh:Q", scale=alt.Scale(domain=[0, 100]), title="Puesto que dijo el modelo"),
                 color=alt.Color("grupo:N", scale=alt.Scale(domain=[GRUPO_ARM, GRUPO_RESTO], range=[ARMORY, GRIS]),
                                 legend=None),
                 order=alt.Order("primero:Q"),
                 tooltip=[alt.Tooltip("n:N", title="Skin"), alt.Tooltip("c:N", title="Colección"),
                          alt.Tooltip("y:Q", title="Puesto real", format=".0f"), alt.Tooltip("yh:Q", title="Puesto del modelo", format=".0f"),
                          alt.Tooltip("p:Q", title="Precio real (USD)", format=",.2f"), alt.Tooltip("ph:Q", title="Precio del modelo (USD)", format=",.2f")]))
txt_e = alt.layer(
    alt.Chart(datos([{"t": "Zona salmón, arriba de la línea: el modelo lo pone más caro de lo que vale"}]))
    .mark_text(align="left", baseline="top", color=ARMORY, fontWeight=700, fontSize=12).encode(x=alt.value(10), y=alt.value(8), text="t:N"),
    alt.Chart(datos([{"x": 99, "y": 2, "t": "Sobre la línea punteada: acierta. Debajo: lo pone más barato"}]))
    .mark_text(align="right", baseline="bottom", color=GRIS, fontWeight=600, fontSize=11.5).encode(x="x:Q", y="y:Q", text="t:N"))
graf_error = alt.layer(arriba, diag_e, pts_e, txt_e).add_params(p_vista).properties(width="container", height=400)
# el marcador de cada vista
NUBE = {"tr_r2": R2_OOF, "tr_arriba": float(100 * (train["predicho"] > train["percentil"]).mean()),
        "tr_sesgo": float((train["predicho"] - train["percentil"]).mean()),
        "te_r2": R2_TEST, "te_arriba": float(100 * (test["predicho"] > test["percentil"]).mean())}

# 11.5 el Armory en el modelo: lo que aprendió contra lo que encontró, por origen
LADOS = {"entrenamiento": "Lo que aprendió (colecciones viejas)", "prueba": "Lo que encontró (colecciones nuevas)"}
corr_g = [dict(c, lado=LADOS[c["lado"]]) for c in corrimiento]
b_a = alt.Chart(datos(corr_g))
y_a = alt.Y("origen:N", sort=ORIGENES, title=None, axis=alt.Axis(labelFontWeight=600))
lin_a = b_a.mark_line(strokeWidth=3, opacity=0.6).encode(
    x=alt.X("mediana:Q", scale=alt.Scale(domain=[0, 100]), title="Puesto típico en el mercado (0 a 100)"),
    y=y_a, color=alt.Color("origen:N", scale=esc_origen, legend=None), detail="origen:N")
pts_a = puntos_viejas_nuevas(
    b_a, "lado", list(LADOS.values()), "mediana:Q", y_a, alt.Color("origen:N", scale=esc_origen, legend=None),
    [alt.Tooltip("origen:N"), alt.Tooltip("lado:N"), alt.Tooltip("mediana:Q", format=".1f", title="Puesto típico"),
     alt.Tooltip("filas:Q", title="Skins"), alt.Tooltip("sesgo:Q", format="+.1f", title="Error medio del modelo")])
txt_a = b_a.mark_text(dy=-15, fontWeight=600, fontSize=11.5).encode(x="mediana:Q", y=y_a, text=alt.Text("mediana:Q", format=".0f"))
graf_armory = (lin_a + pts_a + txt_a).properties(width="container", height=220,
                                                usermeta={"viejas": LADOS["entrenamiento"]})

# 11.4c el modelo, paso 2: la partición en el tiempo. Forma = de qué lado quedó (círculo lleno:
# entrenamiento; triángulo hueco: prueba). Color = Armory o no, igual que en el resto de la página.
LADO_TR, LADO_TE = "Entrenamiento (las vio)", "Prueba (nunca las vio)"
for p_ in PARTICION:
    p_["lado_txt"] = LADO_TR if p_["lado"] == "entrenamiento" else LADO_TE
esc_grupo = alt.Scale(domain=[COL_ARM, COL_RESTO], range=[ARMORY, GRIS])
b_p = alt.Chart(datos(PARTICION))
x_p = alt.X("f:T", title="Año en que salió la colección", axis=alt.Axis(format="%Y", tickCount=12), scale=alt.Scale(padding=14))
y_p = alt.Y("origen:N", sort=ORIGENES, title=None, axis=alt.Axis(labelFontWeight=600))
pts_p = b_p.mark_point(strokeWidth=2, opacity=0.9).encode(
    x=x_p, y=y_p,
    size=alt.Size("filas:Q", scale=alt.Scale(range=[40, 520]), legend=None),
    color=alt.Color("grupo:N", scale=esc_grupo, legend=alt.Legend(orient="top", title=None)),
    fill=alt.condition("datum.lado == 'entrenamiento'", alt.Fill("grupo:N", scale=esc_grupo), alt.value("transparent")),
    shape=alt.Shape("lado_txt:N", scale=alt.Scale(domain=[LADO_TR, LADO_TE], range=[FORMA_VIEJAS, FORMA_NUEVAS]),
                    legend=alt.Legend(orient="top", title=None, symbolStrokeColor=GRIS, symbolStrokeWidth=2)),
    tooltip=[alt.Tooltip("nombre:N", title="Colección"), alt.Tooltip("f:T", title="Salió", format="%m/%Y"),
             alt.Tooltip("origen:N", title="Cómo se consigue"), alt.Tooltip("lado_txt:N", title="Lado"),
             alt.Tooltip("filas:Q", title="Skins")])
inicio_test = f"{test['collection_release'].min():%Y-%m-%d}"
corte_p = alt.Chart(datos([{"f": inicio_test}])).mark_rule(strokeDash=[4, 3], strokeWidth=1.5).encode(x="f:T")
corte_t = alt.Chart(datos([{"f": inicio_test, "t": "Desde acá, el modelo no vio nada →"}])) \
    .mark_text(align="right", dx=-6, fontWeight=600, fontSize=11.5).encode(x="f:T", y=alt.value(-8), text="t:N")
zona_p = alt.Chart(datos([{"a": ARMORY_DESDE.strftime("%Y-%m-%d"), "b": fin_snap}])) \
    .mark_rect(color=ARMORY, opacity=0.08).encode(x="a:T", x2="b:T")
graf_particion = alt.layer(zona_p, corte_p, corte_t, pts_p).properties(width="container", height=230,
                                                                      usermeta={"viejas": LADO_TR})

# 11.5b la prueba de la causa: con el Armory como origen propio, el error casi desaparece
VER_SIN, VER_CON = "Sin avisarle (como lo entrenamos)", "Con el Armory como origen propio"
def_largo = [dict(d, version_txt=VER_SIN if d["version"] == "sin" else VER_CON) for d in DEFENSA]
orden_def = list(PARAMS_EXP["sin"].keys())
esc_ver = alt.Scale(domain=[VER_SIN, VER_CON], range=[GRIS, ARMORY])
b_d = alt.Chart(datos(def_largo))
graf_defensa = alt.layer(
    b_d.mark_bar(height=15, cornerRadiusEnd=3).encode(
        x=alt.X("r2:Q", scale=alt.Scale(domain=[0, 1]), title=f"R² en la prueba (colecciones desde {CORTE_EXP:%m/%Y})"),
        y=alt.Y("modelo:N", sort=orden_def, title=None, axis=alt.Axis(labelFontWeight=600, labelLimit=220)),
        yOffset=alt.YOffset("version_txt:N", sort=[VER_SIN, VER_CON]),
        color=alt.Color("version_txt:N", scale=esc_ver, legend=alt.Legend(orient="top", title=None, direction="vertical")),
        tooltip=[alt.Tooltip("modelo:N"), alt.Tooltip("version_txt:N", title="Versión"),
                 alt.Tooltip("r2:Q", title="R²", format=".3f"), alt.Tooltip("sesgo_mapa:Q", title="Error medio en drop de mapa", format="+.1f")]),
    b_d.mark_text(align="left", dx=5, fontSize=11.5, fontWeight=600).encode(
        x="r2:Q", y=alt.Y("modelo:N", sort=orden_def), yOffset=alt.YOffset("version_txt:N", sort=[VER_SIN, VER_CON]),
        text=alt.Text("r2:Q", format=".2f")),
    alt.Chart(datos([{"x": DEFENSA_TABLA}])).mark_rule(strokeDash=[4, 3], color=GRIS, strokeWidth=1.5).encode(x="x:Q"),
    alt.Chart(datos([{"x": DEFENSA_TABLA, "t": f"Tabla de rareza y exterior: {n(DEFENSA_TABLA, 2)}"}]))
    .mark_text(align="left", dx=5, color=GRIS, fontSize=11.5).encode(x="x:Q", y=alt.value(-8), text="t:N"),
).properties(width="container", height=250)

# 11.6 cómo lo sabemos del modelo: la comparación con media y desvío
comp = [{"modelo": r["modelo"], "r2": r["r2"], "lo": r["r2"] - r["desvio"], "hi": r["r2"] + r["desvio"],
         "tipo": "piso" if r in pisos else ("elegido" if r["modelo"] == ELEGIDO else "modelo")}
        for r in pisos + comparacion]
orden_c = [c["modelo"] for c in sorted(comp, key=lambda c: -c["r2"])]
b_c = alt.Chart(datos(comp))
col_c = alt.Color("tipo:N", scale=alt.Scale(domain=["piso", "modelo", "elegido"], range=[GRIS, GRIS, AMBAR]), legend=None)
graf_comp = alt.layer(
    b_c.mark_rule(strokeWidth=3, opacity=0.6).encode(
        x=alt.X("lo:Q", title="R² en validación (media ± desvío entre pedazos)"), x2="hi:Q",
        y=alt.Y("modelo:N", sort=orden_c, title=None), color=col_c,
        strokeDash=alt.condition("datum.tipo == 'piso'", alt.value([3, 3]), alt.value([1, 0]))),
    b_c.mark_point(size=110, strokeWidth=2, opacity=1).encode(
        x="r2:Q", y=alt.Y("modelo:N", sort=orden_c), color=col_c,
        fill=alt.condition("datum.tipo == 'piso'", alt.value("transparent"), col_c),
        tooltip=[alt.Tooltip("modelo:N"), alt.Tooltip("r2:Q", format=".3f", title="R²"),
                 alt.Tooltip("lo:Q", format=".3f", title="− desvío"), alt.Tooltip("hi:Q", format=".3f", title="+ desvío")]),
).properties(width="container", height=230)

graf_imp = (alt.Chart(datos(IMPORTANCIA)).mark_bar(color=AMBAR, cornerRadiusEnd=3, height=16)
            .encode(x=alt.X("baja:Q", title="Cuánto empeora el modelo si le borramos esa columna (baja del R²)"),
                    y=alt.Y("columna:N", sort="-x", title=None, axis=alt.Axis(labelFontWeight=600)),
                    tooltip=[alt.Tooltip("columna:N"), alt.Tooltip("baja:Q", format=".3f"), alt.Tooltip("desvio:Q", format=".3f")])
            ).properties(width="container", height=210)

# 11.7 explorá: todos los skins, por año y por precio, con los colores del juego
p_origen = alt.param(name="f_origen", value="todos")
p_rareza = alt.param(name="f_rareza", value=-1)
p_arma = alt.param(name="f_arma", value="todas")
p_var = alt.param(name="f_var", value="todas")
p_buscar = alt.param(name="f_buscar", value="")   # texto del buscador, en minúscula
filtro = ("(f_origen == 'todos' || datum.o == f_origen) && (f_rareza < 0 || datum.r == f_rareza)"
          " && (f_arma == 'todas' || datum.a == f_arma) && (f_var == 'todas' || datum.v == f_var)"
          " && (f_buscar == '' || indexof(lower(datum.n), f_buscar) >= 0)")
precio_min, precio_max = float(df["sell_price_usd"].min()), float(df["sell_price_usd"].max())
zona_e = alt.Chart(datos([{"a": ARMORY_DESDE.strftime("%Y-%m-%d"), "b": fin_snap}])) \
    .mark_rect(color=ARMORY, opacity=0.08).encode(x="a:T", x2="b:T")
zona_et = alt.Chart(datos([{"a": ARMORY_DESDE.strftime("%Y-%m-%d"), "t": "Armory"}])) \
    .mark_text(align="left", dx=6, color=ARMORY, fontWeight=700, fontSize=12).encode(x="a:T", y=alt.value(12), text="t:N")
nube = (alt.Chart(skins).transform_filter("datum.f != ''").transform_filter(filtro)
        .transform_calculate(fecha="time(toDate(datum.f)) + datum.j * 86400000", rareza=rareza_nombre)
        .mark_circle(size=18, opacity=0.8, stroke=None)
        .encode(x=alt.X("fecha:T", title="Año en que salió la colección", axis=alt.Axis(format="%Y", tickCount=14, grid=False)),
                y=alt.Y("p:Q", scale=alt.Scale(type="log", domain=[precio_min * 0.8, precio_max * 1.25], nice=False),
                        title="Precio en Steam (escala de 10 en 10)",
                        axis=alt.Axis(values=[0.1, 1, 10, 100, 1000], labelExpr=EJE_USD)),
                color=alt.Color("rareza:N", scale=esc_rareza,
                                legend=alt.Legend(orient="top", title=None, symbolType="circle", labelExpr=corta)),
                order=alt.Order("r:Q"),
                # Sin tooltip, Vega no detecta el punto bajo el mouse y la ficha no se actualiza;
                # la página desactiva la ventanita de este gráfico (la ficha ya muestra todo).
                tooltip=alt.Tooltip("n:N")))
graf_explora = alt.layer(zona_e, zona_et, nube).add_params(p_origen, p_rareza, p_arma, p_var, p_buscar) \
    .properties(width="container", height=480)

# 11.8 probá el modelo: dónde cae el skin fabricado en la escalera del mercado
p_pred = alt.param(name="pred", value=50)
p_lo = alt.param(name="lo", value=35)
p_hi = alt.param(name="hi", value=65)
fondo = (alt.Chart(skins).transform_calculate(rareza=rareza_nombre)
         .mark_tick(orient="vertical", thickness=1, opacity=0.35)
         .encode(x=alt.X("q:Q", scale=alt.Scale(domain=[0, 100]), title=None,
                         axis=alt.Axis(values=[0, 25, 50, 75, 100], labelExpr=EJE_PUESTO)),
                 color=alt.Color("rareza:N", scale=esc_rareza, legend=None)))
uno = alt.Chart(datos([{}]))
banda = uno.transform_calculate(a="lo", b="hi").mark_rect(color=AMBAR, opacity=0.18).encode(x="a:Q", x2="b:Q")
marca = uno.transform_calculate(a="pred").mark_rule(color=AMBAR, strokeWidth=3).encode(x="a:Q")
punta = uno.transform_calculate(a="pred").mark_point(shape="triangle-down", size=260, filled=True, color=AMBAR) \
    .encode(x="a:Q", y=alt.value(-6))
graf_proba = (fondo + banda + marca + punta).add_params(p_pred, p_lo, p_hi).properties(width="container", height=90)

ESPECS = {k: v.to_dict() for k, v in {
    "datos": graf_datos, "precios": graf_precios, "hallazgo": graf_hallazgo,
    "armory_tiempo": graf_armory_tiempo, "armory_rareza": graf_armory_rareza, "modelo": graf_modelo,
    "error": graf_error, "armory": graf_armory, "comparacion": graf_comp, "importancia": graf_imp,
    "particion": graf_particion, "defensa": graf_defensa,
    "explora": graf_explora, "proba": graf_proba}.items()}


# ================================================================ 12 · textos y página

def con_mayuscula(texto):
    return re.sub(r"^((?:<[^>]+>)*)([a-záéíóúñ])", lambda m: m.group(1) + m.group(2).upper(), str(texto))


fila_tabla = lambda celdas: "<tr>" + "".join(f"<td>{con_mayuscula(c)}</td>" for c in celdas) + "</tr>"
T = {}
T["SNAPSHOT"] = date.fromisoformat(SNAPSHOT).strftime("%d/%m/%Y")
T["FILAS"] = n(len(df))
T["COLECCIONES"] = n(df["collection"].nunique())
T["SKINS_BASE"] = n(df["skin_id"].nunique())
T["ARMAS_N"] = n(len(ARMAS))
T["PRECIO_MIN"] = n(df["sell_price_usd"].min(), 2)
T["PRECIO_MAX"] = n(df["sell_price_usd"].max(), 0)
T["PRECIO_MED"] = n(mediana_precio, 2)
T["VECES_RANGO"] = n(df["sell_price_usd"].max() / df["sell_price_usd"].min(), 0)
T["COBERTURA"] = CALIDAD.get("cobertura", "—").replace(".", ",")
T["MERCADO_ARMAS"] = n(CALIDAD.get("items_de_arma_en_el_mercado", EMBUDO.get("steam_items_de_arma", 0)))
T["DATOS_FRESCOS"] = "sí" if CALIDAD.get("datos_frescos") else "no"
e = EMBUDO
T["EMBUDO"] = "".join(fila_tabla(c) for c in [
    ("Ítems de arma publicados en Steam ese día", "—", n(e.get("steam_items_de_arma", 0)), "Es el universo: la consulta ya pide sólo rifles, pistolas, subfusiles y pesadas"),
    ("Muestra sistemática: páginas repartidas por todo el catálogo", "—", n(e.get("steam_items_leidos", 0)),
     f"{n(e.get('steam_paginas', 0))} páginas de 10 ítems, en proporción a cada categoría"),
    ("Duplicados por la paginación en vivo", "−" + n(e.get("steam_items_leidos", 0) - e.get("steam_sin_duplicados", 0)),
     n(e.get("steam_sin_duplicados", 0)), "El mercado se mueve mientras se pagina: un ítem puede aparecer en dos páginas"),
    ("Sin precio o con precio ≤ 0", "−" + n(e.get("steam_sin_duplicados", 0) - e.get("steam_con_precio", 0)),
     n(e.get("steam_con_precio", 0)), "Un ítem sin precio no sirve para predecir precio"),
    ("Unión con el catálogo de ByMykel por market_hash_name", "−" + n(e.get("steam_con_precio", 0) - e.get("unidos", 0)),
     n(e.get("unidos", 0)), f"El catálogo tiene {n(e.get('catalogo_armas', 0))} ítems de arma; todos los precios encontraron su ficha"),
    ("Validación: clave única, colección en el 100 %, más de 1.000 filas", "0", n(len(df)), "Si algo falla, el pipeline no publica"),
])
T["PEDAZOS"] = "".join(fila_tabla((p["pedazo"], n(p["entrena"]), n(p["valida"]), f"{p['desde']} a {p['hasta']}"))
                       for p in tabla_pedazos)
T["N_TRAIN"], T["N_TEST"] = n(len(train)), n(len(test))
T["COL_TRAIN"], T["COL_TEST"] = n(train["collection"].nunique()), n(test["collection"].nunique())
T["TEST_DESDE"] = f"{test['collection_release'].min():%m/%Y}"

T["H2_D"], T["H2_PARES"], T["H2_MED"] = n(H2["d"], 2), n(H2["pares"]), n(H2["mediana"], 0)
T["H3_RHO"], T["H3_SKINS"], T["H3_BAJA"] = n(H3["rho"], 2), n(H3["skins"]), n(H3["baja"])
# Tres decimales: con dos, 0,1990 se leería «0,20» y no se vería por qué cae debajo del umbral de 0,2.
T["H4_R"], T["H4_FILAS"] = n(H4["r"], 3), n(H4["filas"])
T["MED_COVERT"], T["MED_CONSUMER"] = n(MEDIANA_RAREZA[5]), n(MEDIANA_RAREZA[0])
T["MED_MILSPEC"], T["MED_CLASSIFIED"] = n(MEDIANA_RAREZA[2]), n(MEDIANA_RAREZA[4])
T["SPEARMAN_RAREZA"] = n(SPEARMAN_RAREZA, 2)
T["MED_MAPA_CONS"], T["MED_CAJA_MIL"] = n(MED_RO[("drop de mapa", 0)]), n(MED_RO[("caja", 2)])
T["MED_CAJA_COV"], T["MED_SOUV_COV"] = n(MED_RO[("caja", 5)]), n(MED_RO[("paquete souvenir", 5)])
T["MED_TODOS_CONS"], T["MED_TODOS_MIL"] = n(MEDIANA_RAREZA[0]), n(MEDIANA_RAREZA[2])
T["H4_SEMAFORO"] = ("rojo en el semáforo (menos de 0,2): no hay efecto que afirmar" if abs(H4["r"]) < 0.2 else
                    "amarillo en el semáforo (entre 0,2 y 0,6): no alcanza para afirmarla" if abs(H4["r"]) < 0.6 else
                    "verde en el semáforo")

T["ELEGIDO"] = ELEGIDO
T["R2_CV"], T["DESVIO_CV"] = n(R2_CV, 2), n(DESVIO_CV, 2)
T["R2_CV3"], T["DESVIO_CV3"] = n(R2_CV, 3), n(DESVIO_CV, 3)
T["R2_CV_PCT"] = pct(100 * R2_CV)
T["R2_PISO"] = n(PISO["r2"], 2)
T["R2_TABLA"] = n(MEJOR_PISO["r2"], 2)
T["R2_TABLA_PCT"] = pct(100 * MEJOR_PISO["r2"])
T["R2_TEST"] = n(R2_TEST, 2)
T["R2_TABLA_TEST"] = n(PRUEBA[1]["prueba"], 2)
T["R2_PISO_TEST"] = n(PRUEBA[0]["prueba"], 2)
T["MAE_CV"], T["MAE_TEST"] = n(MAE_CV, 1), n(MAE_TEST, 1)
T["SESGO_CV"], T["SESGO_TEST"] = n(SESGO_CV, 1), n(SESGO_TEST, 0)
T["SESGO_TEST1"] = n(SESGO_TEST, 1)
T["SESGO_PISO_TEST"] = n(PRUEBA[0]["sesgo_prueba"], 1)
T["COL_SESGO_POS"] = f"{COLECCIONES_SESGO_POS} de {test['collection'].nunique()}"
T["MAS_CARO"] = pct(USD["mas_caro"])
T["USD_MED"], T["USD_MEDIO"] = n(USD["abs_mediano"], 2), n(USD["abs_medio"], 2)
T["USD_VECES"], T["USD_X2"] = n(USD["veces_mediana"], 1), pct(USD["dentro_x2"])
T["USD_REAL"], T["USD_PRED"] = n(USD["real_mediano"], 2), n(USD["pred_mediano"], 2)
T["MAPA_TR"], T["MAPA_TE"] = n(next(c["mediana"] for c in corrimiento if c["origen"] == "drop de mapa" and c["lado"] == "entrenamiento"), 1), \
    n(next(c["mediana"] for c in corrimiento if c["origen"] == "drop de mapa" and c["lado"] == "prueba"), 1)
T["MAPA_SESGO"] = n(next(c["sesgo"] for c in corrimiento if c["origen"] == "drop de mapa" and c["lado"] == "prueba"), 1)
T["MAPA_COL_TR"], T["MAPA_COL_TE"] = str(MAPA["col_tr"]), str(MAPA["col_te"])
T["MAPA_ULTIMA"], T["MAPA_PRIMERA"] = MAPA["ultima_tr"], MAPA["primera_te"]
T["MEDIANA_TEST"] = n(test["percentil"].median(), 0)
T["DUELOS"] = "".join(fila_tabla((d["retador"], ("+" if d["dif"] >= 0 else "−") + n(abs(d["dif"]), 3),
                                  n(d["desvio_dif"], 3), f"{d['gana']} de {len(PEDAZOS)}",
                                  "<b>lo reemplaza</b>" if d["supera"] and d["retador"] == ELEGIDO else "no alcanza"))
                      for d in duelos)
T["COMPARACION"] = "".join(fila_tabla((r["modelo"], n(r["r2"], 3), n(r["desvio"], 3), n(r["mae"], 1), "+" + n(r["sesgo"], 1)))
                           for r in sorted(pisos + comparacion, key=lambda r: -r["r2"]))
T["MEJORES"] = ", ".join(f"<code>{k}={v}</code>" for k, v in MEJORES.items())
T["N_GRILLA"] = str(N_GRILLA)
T["AJUSTE_DIF"], T["AJUSTE_DESV"] = n(AJUSTE["dif"], 3), n(AJUSTE["desvio_dif"], 3)
T["AJUSTE_LECTURA"] = ("sube por más que el desvío de la diferencia, así que vale la pena"
                       if AJUSTE["dif"] > AJUSTE["desvio_dif"] else "queda dentro del ruido")
T["TRAMOS"] = "".join(fila_tabla((str(t.tramo), n(t.filas), n(t.mae, 1), ("+" if t.sesgo >= 0 else "−") + n(abs(t.sesgo), 1),
                                  "USD " + n(t.real, 2), "USD " + n(t.pred, 2), "USD " + n(t.usd, 2)))
                      for t in por_tramo.itertuples())
T["PEORES"] = "".join(fila_tabla((r.market_hash_name, r.collection.replace("The ", "").replace(" Collection", ""), r.origen,
                                  n(r.percentil, 0), n(r.predicho, 0), "USD " + n(r.sell_price_usd, 2),
                                  "USD " + n(r.precio_predicho, 2)))
                      for r in peores.itertuples())
T["IMP_1"], T["IMP_1_V"] = IMPORTANCIA[0]["columna"], n(IMPORTANCIA[0]["baja"], 2)
T["IMP_2"], T["IMP_2_V"] = IMPORTANCIA[1]["columna"], n(IMPORTANCIA[1]["baja"], 2)
T["R2_OOF"] = n(R2_OOF, 2)
T["VERIFICADO"] = VERIFICADO
T["SKLEARN"] = sklearn.__version__
T["N_GRILLA_APP"] = n(len(grilla))

# el precio y la rareza, en dólares
T["PRECIO_P90"] = usd(p90_precio)
cr = {r["rareza"]: r for r in CAJA_RAREZA}
T["CAJA_MIL"], T["CAJA_COV"] = usd(cr["Mil-Spec Grade"]["med"]), usd(cr["Covert"]["med"])
T["CAJA_VECES"] = n(cr["Covert"]["med"] / cr["Mil-Spec Grade"]["med"])
T["CAJA_PASO_MIN"] = n(min(r["paso"] for r in CAJA_RAREZA if "paso" in r))
T["CAJA_PASO_MAX"] = n(max(r["paso"] for r in CAJA_RAREZA if "paso" in r))
# el Armory
T["ARM_FECHA"] = ARMORY_DESDE.strftime("%d/%m/%Y")
T["ARM_COL"], T["ARM_COL_PRE"] = str(ARM["col_arm"]), str(ARM["col_pre"])
T["ARM_USD_PRE"], T["ARM_USD"] = usd(ARM["usd_pre"]), usd(ARM["usd_arm"])
T["ARM_Q_PRE"], T["ARM_Q"] = n(ARM["q_pre"]), n(ARM["q_arm"])
T["ARM_VECES"] = n(ARM["veces"])
T["ARM_MIL_PRE"], T["ARM_MIL"] = usd(MILSPEC_MAPA["antes"]), usd(MILSPEC_MAPA["armory"])
T["ARM_MIL_VECES"] = n(MILSPEC_MAPA["veces"])
T["ARM_TEST_COL"], T["ARM_TRAIN_COL"] = str(ARM["test_col_arm"]), str(ARM["train_col_arm"])
T["ARM_TEST_PCT"] = pct(ARM["test_pct_mapa_arm"])
T["ARM_SESGO"], T["ARM_SESGO_RESTO"] = n(ARM["sesgo_mapa_arm"]), n(ARM["sesgo_resto"])
T["ARM_CARO"] = pct(ARM["caro_mapa_arm"])
T["ARM_Q_MAX"] = n(col_mapa.loc[col_mapa["armory"], "med"].max())
# colores que la plantilla toma de las mismas constantes que los gráficos
def texto_sobre(hexa):
    """Texto oscuro sobre colores claros y blanco sobre oscuros (luminancia relativa)."""
    r, g, b = (int(hexa[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = lambda v: v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return "#14181d" if 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b) > 0.2 else "#ffffff"


cuadradito = lambda texto, hexa: f'<span class="cuadradito" style="--c:{hexa};--t:{texto_sobre(hexa)}">{texto}</span>'
T["CHIPS_RAREZA"] = "".join(cuadradito(r.split(" ")[0], c) for r, c in zip(RAREZAS, COL_RAREZA))
for clave, i in (("CAJA", 0), ("MAPA", 1), ("SOUVENIR", 2), ("SIN_ID", 3)):
    T[f"CHIP_{clave}"] = cuadradito(ORIGENES[i][0].upper() + ORIGENES[i][1:], COL_ORIGEN[i])
T["COL_COVERT"], T["COL_ARMORY"], T["COL_MODELO"], T["COL_GRIS"] = COL_RAREZA[5], ARMORY, AMBAR, GRIS
MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
T["ARM_MES"] = f"{MESES_ES[ARMORY_DESDE.month - 1]} de {ARMORY_DESDE.year}"
T["IMP_1_PCT"] = pct(100 * IMPORTANCIA[0]["baja"] / sum(max(0, r["baja"]) for r in IMPORTANCIA))
T["H3_MEDIA"] = n(H3["media"], 2)
T["GRADIENTE_RAREZA"] = ", ".join(COL_RAREZA)
T["SESGO_TEST0"] = n(SESGO_TEST, 0)

# el modelo en el navegador: la alternativa que el instructivo admite (a menos de un desvío)
T["R2_RIDGE"], T["DESVIO_ELEGIDO"] = n(res[REFERENCIA]["r2"], 3), n(res[ELEGIDO]["desvio"], 3)
T["R2_ELEGIDO_DEF"] = n(res[ELEGIDO]["r2"], 3)
T["DIF_RIDGE"] = n(res[ELEGIDO]["r2"] - res[REFERENCIA]["r2"], 3)

# la prueba de la causa (sección 8c)
defe = {(d["modelo"], d["version"]): d for d in DEFENSA}
gb_sin, gb_con = defe[(ELEGIDO, "sin")], defe[(ELEGIDO, "con")]
T["DEF_SIN"], T["DEF_CON"] = n(gb_sin["r2"], 2), n(gb_con["r2"], 2)
T["DEF_SESGO_SIN"], T["DEF_SESGO_CON"] = n(gb_sin["sesgo_mapa"], 1), n(gb_con["sesgo_mapa"], 1)
T["DEF_TABLA"] = n(DEFENSA_TABLA, 2)
T["DEF_CORTE"] = f"{CORTE_EXP:%m/%Y}"
T["DEF_TRAIN"], T["DEF_TEST"], T["DEF_COL_TEST"] = n(DEF_INFO["train"]), n(DEF_INFO["test"]), n(DEF_INFO["col_test"])
T["DEF_ARM_TRAIN"] = n(DEF_INFO["arm_train"])
signo = lambda x, d=1: ("+" if x >= 0 else "−") + n(abs(x), d)
T["DEF_TABLA_FILAS"] = "".join(fila_tabla((m, n(defe[(m, "sin")]["r2"], 2), n(defe[(m, "con")]["r2"], 2),
                                           signo(defe[(m, "sin")]["sesgo_mapa"]), signo(defe[(m, "con")]["sesgo_mapa"])))
                               for m in orden_def)

# las demos del juego, el hallazgo en tres pasos y la nube con dos vistas
T["VECES_ST"], T["VECES_SV"] = n(VECES_ST, 1), n(VECES_SV, 1)
T["MED_CLAS_MIL"], T["MED_ARM_MIL"] = n(med_mil_clas), n(med_mil_arm)
T["NUBE_TR_R2"], T["NUBE_TR_ARRIBA"] = n(NUBE["tr_r2"], 2), pct(NUBE["tr_arriba"])
T["NUBE_TR_SESGO"] = signo(NUBE["tr_sesgo"])
T["NUBE_TE_R2"], T["NUBE_TE_ARRIBA"] = n(NUBE["te_r2"], 2), pct(NUBE["te_arriba"])
T["CAJA_DEMO"], T["CAJA_DEMO_COL"] = CAJA["nombre"], CAJA["col"]

T["N_COEF"] = str(N_COEF)
T["R2_RIDGE2"] = n(res[REFERENCIA]["r2"], 2)

JS = {"SKINS": SKINS, "ESPECS": ESPECS, "INTRO": INTRO, "FORMULA": FORMULA, "NL_NIVEL": NL_NIVEL,
      "SERIE": {k: [SERIE[k], COLOR_SERIE[k]] for k in SERIE}, "COL_ARMORY": ARMORY, "COL_GRIS": GRIS, "GRILLA": GRILLA, "VISTOS": VISTOS, "ARMAS": ARMAS, "ORIGENES": ORIGENES,
      "CUANTILES": CUANTILES_PRECIO, "MAE_CV": round(MAE_CV, 2), "SESGO_MAPA": round(float(T["MAPA_SESGO"].replace(",", ".")), 1),
      "MAE_TEST": round(MAE_TEST, 2), "SESGO_TEST": round(SESGO_TEST, 1), "GANCHO": GANCHO,
      "COL_RAREZA": COL_RAREZA, "COL_ORIGEN": COL_ORIGEN, "ARMORY_DESDE": ARMORY_DESDE.strftime("%Y-%m-%d"),
      "IMAGENES": IMAGENES, "DISENOS": DISENOS}

html = Path(args.plantilla).read_text(encoding="utf-8")
for k, v in T.items():
    html = html.replace("{{" + k + "}}", str(v))
faltan = re.findall(r"\{\{([A-Z0-9_]+)\}\}", html)
assert not faltan, f"marcadores sin reemplazar: {faltan}"
datos_js = "const APP = " + json.dumps(JS, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";"
html = html.replace("/*__APP__*/", datos_js.replace("</", "<\\/"))
salida = Path(args.salida)
salida.parent.mkdir(parents=True, exist_ok=True)
salida.write_text(html, encoding="utf-8")
print(f"aplicación escrita en {salida} ({salida.stat().st_size / 1e6:.1f} MB)")
print(json.dumps({"r2_cv": R2_CV, "desvio": DESVIO_CV, "r2_test": R2_TEST, "mae_test": MAE_TEST, "sesgo_test": SESGO_TEST,
                  "elegido": ELEGIDO, "mejores": MEJORES, "usd": USD, "h2": H2, "h3": H3, "h4": H4,
                  "r2_oof": R2_OOF, "mediana_rareza": MEDIANA_RAREZA,
                  "defensa": DEFENSA, "defensa_tabla": DEFENSA_TABLA}, default=str, indent=1))
