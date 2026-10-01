import time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, GroupKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

warnings.filterwarnings("ignore")
SEED = 42
FIG, RES = Path("figs"), Path("results"); FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
C_BASE, C_NO, C_PCA = "#9aa0a6", "#2b6cb0", "#dd6b20"
t0_all = time.time()

TARGET = "Y1"
FEATS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
df = pd.read_csv("energy_efficiency_Y1.csv")
X = df[FEATS]; y = df[TARGET]
print("Kích thước:", df.shape, "| thiếu:", int(df.isna().sum().sum()), "| trùng X:", int(X.duplicated().sum()))
print(y.describe().round(2))
print("Số mức của từng đặc trưng:", X.nunique().to_dict())

shape_id = X["X1"].rank(method="dense").astype(int)
print("Số hình dạng:", shape_id.nunique(), "| mỗi hình dạng có", shape_id.value_counts().unique(), "mẫu")
print("X2 == X3 + 2*X4 với mọi mẫu:", bool(np.allclose(X.X2, X.X3 + 2 * X.X4)))
print("Mỗi hình dạng có đúng 1 giá trị X2..X5:", bool((df.groupby(shape_id)[["X2", "X3", "X4", "X5"]].nunique() == 1).all().all()))
print("Hạng ma trận X1..X5:", np.linalg.matrix_rank(X[["X1", "X2", "X3", "X4", "X5"]].values), "/ 5")
print(df.groupby("X5")[TARGET].describe().round(2))
print(df.groupby("X7")[TARGET].mean().round(2))
print(df.groupby("X6")[TARGET].mean().round(2))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
axes[0].hist(y, bins=40, color=C_PCA); axes[0].set_xlabel("Heating Load (kWh/m²)"); axes[0].set_ylabel("Số mẫu")
axes[0].set_title("Phân phối target: 2 cụm")
for h, c in [(3.5, "#38a169"), (7.0, C_NO)]:
    axes[1].hist(y[X.X5 == h], bins=25, alpha=.7, color=c, label=f"Chiều cao {h} m")
axes[1].legend(); axes[1].set_xlabel("Heating Load"); axes[1].set_title("Hai cụm = 1 tầng và 2 tầng")
g = df.groupby("X7")[TARGET].mean()
axes[2].bar([f"{int(v*100)}%" for v in g.index], g.values, color=C_NO); axes[2].set_xlabel("Glazing Area (X7)")
axes[2].set_ylabel("Heating Load TB"); axes[2].set_title("Nhiều kính hơn → tải sưởi cao hơn")
fig.tight_layout(); fig.savefig(FIG / "01_target.png"); plt.close(fig)

corr = df[FEATS + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
ax.set_xticks(range(9)); ax.set_xticklabels(corr.columns); ax.set_yticks(range(9)); ax.set_yticklabels(corr.columns)
for i in range(9):
    for j in range(9):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(corr.iloc[i, j]) > .6 else "black")
fig.colorbar(im, fraction=.046); ax.set_title("Ma trận tương quan"); fig.tight_layout(); fig.savefig(FIG / "02_corr.png"); plt.close(fig)
corr.round(3).to_csv(RES / "corr.csv"); print(corr.round(2))

X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=X["X5"])
print("Dev:", X_dev.shape, "Test:", X_test.shape)
cv_tune = KFold(5, shuffle=True, random_state=SEED)
cv_stab = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)

NUM = ["X1", "X2", "X3", "X4", "X5", "X7"]; CAT = ["X6", "X8"]
def make_pre(enc):
    if enc == "numeric":
        return ColumnTransformer([("num", StandardScaler(), FEATS)])
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])

def make_pipe(model, pca=False, poly=False, enc="onehot"):
    steps = [("pre", make_pre(enc))]
    if enc == "onehot": steps.append(("scale_all", StandardScaler()))
    if pca: steps.append(("pca", PCA(random_state=SEED)))
    if poly: steps += [("poly", PolynomialFeatures(2, include_bias=False)), ("scale2", StandardScaler())]
    steps.append(("model", model))
    return Pipeline(steps)

def cv_score(pipe, cv=cv_tune, groups=None):
    s = cross_validate(pipe, X_dev, y_dev, cv=cv, groups=groups, scoring="neg_root_mean_squared_error", n_jobs=-1)
    return -s["test_score"].mean(), s["test_score"].std(ddof=1)

HGB = lambda: HistGradientBoostingRegressor(random_state=SEED, max_iter=500, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=5)
enc_rows = []
for enc in ["numeric", "onehot"]:
    for nm, mdl in [("Ridge", Ridge(1.0)), ("Poly2 + Ridge", None), ("KNN", KNeighborsRegressor(5, weights="distance")), ("HistGB", HGB())]:
        p = make_pipe(Ridge(1.0), poly=True, enc=enc) if mdl is None else make_pipe(mdl, enc=enc)
        m, s = cv_score(p); enc_rows.append({"biểu diễn": enc, "model": nm, "CV_RMSE": m, "std": s})
enc_df = pd.DataFrame(enc_rows); enc_df.to_csv(RES / "encoding_compare.csv", index=False)
print(enc_df.pivot(index="model", columns="biểu diễn", values="CV_RMSE").round(3))
ENC = enc_df.groupby("biểu diễn").CV_RMSE.mean().idxmin(); print("==> Chốt biểu diễn:", ENC)

pre_fit = Pipeline(make_pipe(LinearRegression(), enc=ENC).steps[:-1]).fit(X_dev)
Z = pre_fit.transform(X_dev); D = Z.shape[1]
pca_full = PCA().fit(Z); evr = pca_full.explained_variance_ratio_; cum = evr.cumsum()
K = {t: int(np.argmax(cum >= t - 1e-12) + 1) for t in (0.80, 0.90, 0.95, 0.99)}
print(f"d = {D} | số PC cần giữ 80/90/95/99 %:", K)
print("EVR:", np.round(evr * 100, 3))
n_zero = int((evr < 1e-6).sum()); print("Số PC có phương sai ≈ 0:", n_zero)
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.bar(range(1, D + 1), evr * 100, color=C_PCA, alpha=.85, label="EVR (%)")
ax.plot(range(1, D + 1), cum * 100, "o-", color=C_NO, label="Tích luỹ (%)")
for t in (80, 95): ax.axhline(t, ls="--", lw=.7, color="grey")
ax.set_xticks(range(1, D + 1)); ax.set_xlabel("Thành phần chính"); ax.set_ylabel("% phương sai"); ax.legend()
ax.set_title(f"Scree plot – d = {D} ({ENC})"); fig.tight_layout(); fig.savefig(FIG / "03_scree.png"); plt.close(fig)
pd.DataFrame({"PC": range(1, D + 1), "EVR": evr, "cum": cum}).to_csv(RES / "pca_evr.csv", index=False)

Zg = StandardScaler().fit_transform(X_dev[["X1", "X2", "X3", "X4", "X5"]])
pg = PCA().fit(Zg); print("PCA trên X1..X5 – EVR (%):", np.round(pg.explained_variance_ratio_ * 100, 3))
geo = pd.DataFrame(pg.components_[:2].T, index=["X1", "X2", "X3", "X4", "X5"], columns=["PC1", "PC2"]).round(3)
geo.to_csv(RES / "pca_geometry_loadings.csv"); print(geo)
json.dump({"geo_evr": pg.explained_variance_ratio_.tolist()}, open(RES / "pca_geometry.json", "w"))

names_out = pre_fit.named_steps["pre"].get_feature_names_out()
load = pd.DataFrame(pca_full.components_[:3].T, index=names_out, columns=["PC1", "PC2", "PC3"]).round(3)
load.to_csv(RES / "pca_loadings.csv"); print(load)
S = pca_full.transform(Z)
pc_y = pd.DataFrame({"PC": range(1, D + 1), "EVR_%": evr * 100,
                     "abs_corr_y": [abs(np.corrcoef(S[:, j], y_dev)[0, 1]) if S[:, j].std() > 1e-9 else 0 for j in range(D)]}).round(4)
pc_y.to_csv(RES / "pc_target_corr.csv", index=False); print(pc_y.to_string(index=False))
fig, ax = plt.subplots(figsize=(6.5, 5))
sc_ = ax.scatter(S[:, 0], S[:, 1], c=y_dev, s=10, cmap="plasma")
fig.colorbar(sc_, label="Heating Load"); ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
ax.set_title("Chiếu lên PC1–PC2, tô màu theo tải sưởi"); fig.tight_layout(); fig.savefig(FIG / "04_pc12.png"); plt.close(fig)

K_GRID = [1, 2, 3, 4, 5, 6, 8]
MODELS = {
    "Baseline (mean)":   (DummyRegressor(strategy="mean"), {}, False),
    "Baseline (median)": (DummyRegressor(strategy="median"), {}, False),
    "Linear":            (LinearRegression(), {}, False),
    "Ridge":             (Ridge(), {"model__alpha": [0.01, 0.1, 1, 10]}, False),
    "Lasso":             (Lasso(max_iter=20000), {"model__alpha": [0.001, 0.01, 0.1]}, False),
    "Poly2 + Ridge":     (Ridge(), {"model__alpha": [0.01, 0.1, 1, 10]}, True),
    "KNN":               (KNeighborsRegressor(weights="distance"), {"model__n_neighbors": [3, 5, 10]}, False),
    "RandomForest":      (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1), {"model__min_samples_leaf": [1, 3]}, False),
    "HistGB":            (HistGradientBoostingRegressor(random_state=SEED, max_iter=500, min_samples_leaf=5),
                          {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [7, 15]}, False),
}
SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
rows, best, folds = [], {}, {}
for name, (model, grid, poly) in MODELS.items():
    for use_pca in ([False] if name.startswith("Baseline") else [False, True]):
        label = name + (" + PCA" if use_pca else "")
        g = dict(grid)
        if use_pca: g["pca__n_components"] = K_GRID
        pipe = make_pipe(model, pca=use_pca, poly=poly, enc=ENC); t0 = time.time(); params = {}
        if g:
            gs = GridSearchCV(pipe, g, cv=cv_tune, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X_dev, y_dev)
            pipe, params = gs.best_estimator_, gs.best_params_
        r = cross_validate(pipe, X_dev, y_dev, cv=cv_stab, scoring=SCORING, return_train_score=True, n_jobs=-1)
        rm = -r["test_rmse"]; folds[label] = rm; best[label] = pipe
        rows.append({"model": label, "family": name, "pca": use_pca,
                     "params": ", ".join(f"{k.split('__')[-1]}={v}" for k, v in params.items()),
                     "CV_RMSE": rm.mean(), "CV_RMSE_std": rm.std(ddof=1), "CV_RMSE_SE": rm.std(ddof=1) / np.sqrt(len(rm)),
                     "CV_MAE": -r["test_mae"].mean(), "CV_MAE_std": r["test_mae"].std(ddof=1),
                     "CV_R2": r["test_r2"].mean(), "CV_R2_std": r["test_r2"].std(ddof=1),
                     "Train_RMSE": -r["train_rmse"].mean(), "time_s": time.time() - t0})
        print(f"{label:24s} RMSE={rm.mean():6.3f}±{rm.std(ddof=1):5.3f} MAE={-r['test_mae'].mean():.3f} R2={r['test_r2'].mean():.4f} {rows[-1]['params']}", flush=True)
res = pd.DataFrame(rows); res.to_csv(RES / "cv_results.csv", index=False)

order = res.sort_values("CV_RMSE", ascending=False)
fig, ax = plt.subplots(figsize=(9, 6.5))
cols = [C_BASE if m.startswith("Baseline") else (C_PCA if p else C_NO) for m, p in zip(order.model, order.pca)]
ax.barh(order.model, order.CV_RMSE, xerr=order.CV_RMSE_std, color=cols, capsize=3)
for i, v in enumerate(order.CV_RMSE): ax.text(v + .15, i, f"{v:.2f}", va="center", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("RMSE CV (kWh/m², thang log) – 25 fold; thanh lỗi = std")
ax.set_title("Baseline và các mô hình (xanh: không PCA, cam: có PCA)"); fig.tight_layout(); fig.savefig(FIG / "05_cv_bar.png"); plt.close(fig)
nb = [m for m in res.model if not m.startswith("Baseline")]
fig, ax = plt.subplots(figsize=(10.5, 4.3))
bp = ax.boxplot([folds[m] for m in nb], patch_artist=True, widths=.6)
for b_, m in zip(bp["boxes"], nb): b_.set_facecolor(C_PCA if "PCA" in m else C_NO); b_.set_alpha(.7)
ax.set_xticks(range(1, len(nb) + 1)); ax.set_xticklabels(nb, rotation=35, ha="right", fontsize=8)
ax.set_yscale("log"); ax.set_ylabel("RMSE (thang log)"); ax.set_title("Độ ổn định qua 25 fold")
fig.tight_layout(); fig.savefig(FIG / "06_cv_box.png"); plt.close(fig)

curve = []
for name in ["Ridge", "Poly2 + Ridge", "KNN", "HistGB"]:
    mdl = best[name].named_steps["model"]
    for k in range(1, D + 1):
        p = make_pipe(mdl, pca=True, poly=(name == "Poly2 + Ridge"), enc=ENC); p.set_params(pca__n_components=k)
        m, s = cv_score(p); curve.append({"model": name, "k": k, "RMSE": m, "std": s})
curve = pd.DataFrame(curve); curve.to_csv(RES / "pca_k_curve.csv", index=False)
pal = {"Ridge": "#38a169", "Poly2 + Ridge": "#2b6cb0", "KNN": "#805ad5", "HistGB": C_PCA}
fig, ax = plt.subplots(figsize=(9, 4.8))
for n in pal:
    c = curve[curve.model == n]; ax.plot(c.k, c.RMSE, "o-", ms=4, color=pal[n], label=f"{n} + PCA(k)")
    ax.axhline(res.loc[res.model == n, "CV_RMSE"].iloc[0], ls=":", color=pal[n], lw=1)
ax.axvline(K[0.95], ls="--", color="grey", lw=.8); ax.text(K[0.95] + .08, 0.03, "95% phương sai", fontsize=8, color="grey", transform=ax.get_xaxis_transform())
ax.set_yscale("log"); ax.set_yticks([0.4, 0.6, 1, 2, 3, 5]); ax.set_yticklabels(["0,4", "0,6", "1", "2", "3", "5"]); ax.set_xticks(range(1, D + 1)); ax.set_xlabel("Số thành phần chính k"); ax.set_ylabel("RMSE CV (thang log)")
ax.set_title("RMSE theo k (nét chấm = cùng mô hình không PCA)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "07_pca_k.png"); plt.close(fig)
print(curve.pivot(index="k", columns="model", values="RMSE").round(3))

load_all = pd.DataFrame(pca_full.components_.T, index=FEATS, columns=[f"PC{i}" for i in range(1, D + 1)]).round(2)
load_all.to_csv(RES / "pca_loadings_all.csv"); print(load_all)
print("|corr(X6, y)| =", round(abs(np.corrcoef(X_dev.X6, y_dev)[0, 1]), 4))

g_dev = X_dev["X1"].rank(method="dense").astype(int)
cv_shape = GroupKFold(n_splits=6)
shape_rows = []
for m in ["Ridge", "Poly2 + Ridge", "KNN", "RandomForest", "HistGB", "HistGB + PCA"]:
    a, _ = cv_score(best[m]); b, _ = cv_score(best[m], cv=cv_shape, groups=g_dev)
    shape_rows.append({"model": m, "KFold ngẫu nhiên": a, "GroupKFold theo hình dạng": b})
shape_df = pd.DataFrame(shape_rows); shape_df.to_csv(RES / "shape_generalization.csv", index=False); print(shape_df.round(3))

cand = res[~res.model.str.startswith("Baseline")].sort_values("CV_RMSE")
top = cand.iloc[0]; thr = top.CV_RMSE + top.CV_RMSE_SE
within = cand[cand.CV_RMSE <= thr]; chosen = within.sort_values(["CV_RMSE_std", "CV_RMSE"]).iloc[0]["model"]
print(f"Tốt nhất: {top.model} {top.CV_RMSE:.3f} (SE {top.CV_RMSE_SE:.3f}) → ngưỡng {thr:.3f}; trong ngưỡng: {list(within.model)} → chọn {chosen}")
test_rows, preds = [], {}
for m in res.model:
    p = best[m].fit(X_dev, y_dev); yp = p.predict(X_test); preds[m] = yp
    test_rows.append({"model": m, "Test_MAE": mean_absolute_error(y_test, yp), "Test_RMSE": root_mean_squared_error(y_test, yp), "Test_R2": r2_score(y_test, yp)})
test = pd.DataFrame(test_rows).merge(res[["model", "CV_RMSE", "CV_RMSE_std"]], on="model")
b = test.loc[test.model == "Baseline (mean)", "Test_RMSE"].iloc[0]; test["giam_vs_baseline_%"] = (1 - test.Test_RMSE / b) * 100
test.to_csv(RES / "test_results.csv", index=False); print(test.round(4).to_string())

yp = preds[chosen]; e = y_test.values - yp
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(y_test, yp, s=12, alpha=.6, color=C_NO); lim = [5, 45]; axes[0].plot(lim, lim, "k--", lw=1)
axes[0].set_xlabel("Tải sưởi thực tế"); axes[0].set_ylabel("Dự báo"); axes[0].set_title(f"Thực tế vs dự báo – {chosen}")
axes[1].scatter(yp, e, s=12, alpha=.6, color=C_PCA); axes[1].axhline(0, color="k", lw=1)
axes[1].set_xlabel("Dự báo"); axes[1].set_ylabel("Phần dư y − ŷ"); axes[1].set_title("Phần dư theo giá trị dự báo")
fig.tight_layout(); fig.savefig(FIG / "08_residuals.png"); plt.close(fig)
err = X_test.assign(y=y_test.values, pred=yp, ae=np.abs(e))
for col, lab in [("X5", "Chiều cao"), ("X7", "Diện tích kính"), ("X6", "Hướng")]:
    t = err.groupby(col).agg(n=("ae", "size"), MAE=("ae", "mean")).round(3); t.to_csv(RES / f"error_by_{col}.csv"); print(lab, "\n", t)
top_e = err.sort_values("ae", ascending=False).head(8).round(3); top_e.to_csv(RES / "top_errors.csv"); print(top_e)

summary = {"D": D, "ENC": ENC, "K": {str(k): v for k, v in K.items()}, "n_zero_pc": n_zero, "chosen": chosen, "thr": float(thr),
           "within": list(within.model), "evr": evr.tolist(), "runtime_min": (time.time() - t0_all) / 60}
json.dump(summary, open(RES / "summary.json", "w"), ensure_ascii=False, indent=1); print(summary)
