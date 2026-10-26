#!/usr/bin/env python3
"""
MQTT tabanlı IoT ağlarında bilinen ve bilinmeyen saldırıların makine öğrenmesi ile tespiti.

Deney 1  Bilinen saldırılar (A veri seti, çok sınıflı): Decision Tree, Random Forest
         1b  Blok (zaman sıralı) bölme ile rastgele bölmenin karşılaştırması
         1c  Sızıntı sütunları (IP, port, zaman...) modele verilirse ne olur
Deney 2  Bilinmeyen saldırılar (A veri seti): Leave-One-Attack-Type-Out ile DT/RF,
         yalnızca normal trafikle eğitilen Isolation Forest
Deney 3  Veri setleri arası genelleme: A ile eğit, B'de test et (ve tersi)

A = MQTTEEB-D, B = MQTTset (varsayılan adlar). Kullanım için README.md dosyasına bak.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # grafikler isteğe bağlı
    plt = None

T0 = time.time()
LEGIT = "legitimate"

# Etiketleri iki veri setinde ortak adlara çevirir. Sıra önemli:
# "Slow DoS" önce slowite'a, "DoS flood" önce flood'a düşmeli.
LABEL_RULES = [
    ("legitimate", ("legit", "normal", "benign")),
    ("slowite", ("slowite", "slow")),
    ("bruteforce", ("brute",)),
    ("malformed", ("malform",)),
    ("flood", ("flood",)),
    ("dos", ("dos",)),
]
CANON = {c for c, _ in LABEL_RULES}
LABEL_COLUMN_CANDIDATES = ("target", "label", "class", "attack_type", "attack", "category", "type")

# Modelin saldırının ne olduğunu değil, kimden ve ne zaman geldiğini ezberlemesine yol açan sütunlar.
# Normalize edilmiş sütun adlarında aranır (ör. "ip.src" -> "ip_src").
LEAK_PATTERNS = [
    r"^ip_", r"^ipv6_", r"(^|_)(src|dst)_?(ip|host|addr)", r"port", r"^eth_", r"(^|_)mac(_|$)", r"^arp_",
    r"timestamp", r"epoch", r"time_relative", r"^frame_time$", r"sniff_time", r"^time$", r"(^|_)date",
    r"frame_number", r"^no$", r"^unnamed", r"^index$", r"^id$",
    r"stream", r"tcp_(seq|ack|nxtseq)(_raw)?$", r"checksum", r"(^|_)ttl$",
    r"client_?id", r"username", r"msgid",
]
# İşlenmiş sürümler (SMOTE / normalize / augmented) test sonuçlarını şişirir
RISKY_FILE_HINTS = ("smote", "normaliz", "standard", "augment", "balanced", "oversampl")
TIME_COLUMN = re.compile(r"(frame_time_epoch|time_epoch|timestamp|sniff_time|^frame_time$|^time$)")
NULL_TOKENS = {"", "nan", "none", "null", "na", "-"}
MISSING = -1.0  # pakette bulunmayan alan (tshark boş bırakır) = -1


@dataclass
class Dataset:
    name: str
    y: np.ndarray            # ortak etiket adları (legitimate, dos, ...)
    raw_labels: pd.Series    # veri setindeki orijinal etiketler
    X_all: pd.DataFrame      # sızıntı dışı tüm sayısal özellikler
    leak: pd.DataFrame       # sızıntı sütunları, sayıya çevrilmiş (deney 1c için)
    text_cols: list
    X: pd.DataFrame | None = None     # ortak özellikler
    train: np.ndarray | None = None   # blok bölmede eğitim maskesi


def log(msg: str = "") -> None:
    print(f"[{time.time() - T0:7.1f} sn] {msg}", flush=True)


# --------------------------------------------------------------------------- okuma

def norm_col(c) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_")


def canon_label(raw) -> str:
    low = str(raw).strip().lower()
    for canon, keys in LABEL_RULES:
        if any(k in low for k in keys):
            return canon
    return low


def is_leak(col: str, extra: list[str]) -> bool:
    return any(re.search(p, col) for p in LEAK_PATTERNS + extra)


def expand_paths(paths: list[str]) -> list[Path]:
    files = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            found = sorted(p.rglob("*.csv"))
            names = " ".join(f.name.lower() for f in found)
            if len(found) > 1 and "raw" in names and "clean" in names:
                sys.exit(f"[DUR] {p} klasöründe hem ham (raw) hem temizlenmiş (cleaned) sürüm var; aynı veri iki kez okunur.\n"
                         "      Klasör yerine kullanacağın dosyayı ver.")
            files += found
        elif p.exists():
            files.append(p)
        else:
            sys.exit(f"[HATA] Bulunamadı: {p}")
    if not files:
        sys.exit(f"[HATA] CSV bulunamadı: {paths}")
    return files


def find_label_col(df: pd.DataFrame, wanted: str | None) -> str | None:
    if wanted:
        w = norm_col(wanted)
        if w not in df.columns:
            sys.exit(f"[HATA] Etiket sütunu '{wanted}' bulunamadı. Sütunlar: {list(df.columns)}")
        return w
    return next((c for c in LABEL_COLUMN_CANDIDATES if c in df.columns), None)


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Zaman sütunu varsa satırları zamana göre sıralar; blok bölme bu sıraya dayanır."""
    for c in df.columns:
        if not TIME_COLUMN.search(c):
            continue
        t = pd.to_numeric(df[c], errors="coerce")
        if t.notna().mean() < 0.9:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = pd.to_datetime(df[c], errors="coerce", utc=True, format="mixed")
            if d.notna().mean() < 0.9:
                continue
            t = (d - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
        log(f"    satırlar '{c}' sütununa göre zaman sırasına dizildi")
        return df.iloc[np.argsort(t.to_numpy(), kind="stable")].reset_index(drop=True)
    return df


def to_numeric_col(s: pd.Series) -> pd.Series | None:
    """Sütunu sayıya çevirir: '0x18' -> 24, 'True' -> 1, '3,3' -> 3. Metin sütunlarında None döner."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype("float64")
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    st = s.astype(str).str.strip().str.split(",", n=1).str[0].str.strip()  # tshark çoklu değer
    low = st.str.lower()
    present = s.notna() & ~low.isin(NULL_TOKENS)
    out = pd.to_numeric(st.where(present), errors="coerce")
    hexm = present & low.str.match(r"^0x[0-9a-f]+$")
    if hexm.any():
        out[hexm] = low[hexm].map(lambda v: float(int(v, 16)))
    boolm = present & low.isin(["true", "false"])
    if boolm.any():
        out[boolm] = (low[boolm] == "true").astype("float64")
    if present.sum() and out[present].notna().mean() < 0.95:
        return None
    return out


def cap_per_class(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Her sınıftan en fazla `cap` satır alır; zaman sırası korunur."""
    if not cap:
        return df
    parts = [g.sample(cap, random_state=seed) if len(g) > cap else g for _, g in df.groupby("_label", sort=False)]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def load_dataset(name: str, paths: list[str], label_col: str | None, args, rename: dict | None = None) -> Dataset:
    log(f"{name} okunuyor")
    frames = []
    for f in expand_paths(paths):
        if any(h in f.name.lower() for h in RISKY_FILE_HINTS) and not args.zorla:
            sys.exit(f"[DUR] {f.name}: işlenmiş bir sürüm gibi görünüyor (SMOTE/normalize/augmented).\n"
                     "      Bu sürümler test sonuçlarını şişirir; ham ya da cleaned CSV kullan.\n"
                     "      Yine de kullanmak istiyorsan --zorla ekle.")
        log(f"  {f.name}")
        df = pd.read_csv(f, low_memory=False)
        df.columns = [norm_col(c) for c in df.columns]
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]
        if rename:
            df = df.rename(columns=rename)
        lc = find_label_col(df, label_col)
        if lc is not None:
            raw = df.pop(lc).astype(str)
        elif canon_label(f.stem) in CANON:
            raw = pd.Series(f.stem, index=df.index)  # etiket dosya adından
        else:
            sys.exit(f"[HATA] {f.name}: etiket sütunu bulunamadı. --etiket-a / --etiket-b ile belirt.\n"
                     f"       Sütunlar: {list(df.columns)}")
        df["_raw_label"] = raw.to_numpy()
        frames.append(sort_by_time(df))

    df = pd.concat(frames, ignore_index=True, sort=False)
    df["_label"] = df["_raw_label"].map(canon_label)
    for (raw, canon), n in df.groupby(["_raw_label", "_label"]).size().items():
        log(f"    etiket {raw!r:32s} -> {canon:11s} {n:>10,} satır")
    unknown = sorted(set(df["_label"]) - CANON)
    if unknown:
        log(f"  [UYARI] tanınmayan etiketler, olduğu gibi bırakıldı: {unknown}")
    if LEGIT not in set(df["_label"]):
        sys.exit(f"[HATA] {name}: normal trafik etiketi bulunamadı (legitimate/normal/benign).")
    df = cap_per_class(df, args.sinif_basina_max, args.tohum)

    feats, leak, text_cols = {}, {}, []
    for c in df.columns:
        if c in ("_raw_label", "_label"):
            continue
        num = to_numeric_col(df[c])
        if c not in args.tut and is_leak(c, args.ekstra_sizinti):
            leak[c] = num if num is not None else pd.Series(pd.factorize(df[c].astype(str))[0], index=df.index)
        elif num is None:
            text_cols.append(c)
        else:
            feats[c] = num
    X = pd.DataFrame(feats, index=df.index).fillna(MISSING).astype("float32")
    L = pd.DataFrame(leak, index=df.index).fillna(MISSING).astype("float32")
    log(f"  {len(df):,} satır | {X.shape[1]} sayısal özellik | {L.shape[1]} sızıntı sütunu | {len(text_cols)} metin sütunu atıldı")
    return Dataset(name, df["_label"].to_numpy(dtype=str), df["_raw_label"], X, L, text_cols)


def align_features(A: Dataset, B: Dataset | None, out: Path) -> list[str]:
    """İki veri setinde de bulunan sayısal özellikleri seçer ve karşılaştırma tablosunu yazar."""
    sets = [D for D in (A, B) if D is not None]
    common = set.intersection(*(set(D.X_all.columns) for D in sets))
    common = sorted(c for c in common if any(D.X_all[c].nunique() > 1 for D in sets))

    def status(D: Dataset, c: str) -> str:
        if c in D.X_all.columns:
            return "özellik (sabit)" if D.X_all[c].nunique() <= 1 else "özellik"
        if c in D.leak.columns:
            return "sızıntı, atıldı"
        if c in D.text_cols:
            return "metin, atıldı"
        return "yok"

    cols = sorted(set().union(*(set(D.X_all.columns) | set(D.leak.columns) | set(D.text_cols) for D in sets)))
    table = pd.DataFrame({"sutun": cols})
    for D in sets:
        table[D.name] = [status(D, c) for c in cols]
    table["ortak_kullaniliyor"] = table["sutun"].isin(common)
    table.sort_values(["ortak_kullaniliyor", "sutun"], ascending=[False, True]).to_csv(
        out / "ozellik_karsilastirma.csv", index=False)
    return common


# --------------------------------------------------------------------------- yardımcılar

def block_split(y: np.ndarray, test_frac: float) -> np.ndarray:
    """Her sınıfın ilk (1 - test_frac) kısmı eğitim, sonu test. Zaman sırası korunur."""
    train = np.zeros(len(y), dtype=bool)
    for lab in np.unique(y):
        idx = np.flatnonzero(y == lab)
        train[idx[: int(round(len(idx) * (1 - test_frac)))]] = True
    return train


def random_split(y: np.ndarray, test_frac: float, seed: int) -> np.ndarray:
    idx = np.arange(len(y))
    tr, _ = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    train = np.zeros(len(y), dtype=bool)
    train[tr] = True
    return train


def make_models(args) -> dict:
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=args.tohum, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=args.agac, random_state=args.tohum,
                                               n_jobs=args.is_sayisi, class_weight="balanced_subsample"),
    }


def binary_metrics(y_true: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    y_true, pred = y_true.astype(int), pred.astype(int)
    both = len(np.unique(y_true)) == 2
    return {
        "yakalama_orani": pred[y_true == 1].mean() if (y_true == 1).any() else np.nan,
        "yanlis_alarm_orani": pred[y_true == 0].mean() if (y_true == 0).any() else np.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score) if both else np.nan,
        "pr_auc": average_precision_score(y_true, score) if both else np.nan,
    }


_IF_CACHE: dict = {}


def fit_isolation_forest(D: Dataset, args):
    """Yalnızca eğitim kısmındaki normal trafikle eğitilir. Eşik, normal trafiğin ayrılmış son %20'sinde
    hedef yanlış alarm oranını verecek şekilde seçilir; saldırı etiketlerine hiç bakılmaz."""
    if D.name not in _IF_CACHE:
        legit = np.flatnonzero((D.y == LEGIT) & D.train)
        cut = int(len(legit) * 0.8)
        if len(legit) - cut < 20:
            sys.exit(f"[HATA] {D.name}: Isolation Forest için yeterli normal trafik yok.")
        X = D.X.to_numpy()
        iso = IsolationForest(n_estimators=200, random_state=args.tohum, n_jobs=args.is_sayisi).fit(X[legit[:cut]])
        thr = np.quantile(-iso.score_samples(X[legit[cut:]]), 1 - args.hedef_fpr)
        _IF_CACHE[D.name] = (iso, thr)
    return _IF_CACHE[D.name]


def row_hashes(X: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    if plt is None:
        return
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2.5, 1.0 * len(labels) + 2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else "black", fontsize=8)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grouped(df: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    if plt is None or df.empty:
        return
    ax = (df * 100).plot.bar(figsize=(max(6, 1.8 * len(df) + 2), 4.2), rot=0, width=0.8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def md_table(df: pd.DataFrame, pct: tuple = (), digits: int = 3) -> str:
    """tabulate gerektirmeden markdown tablo."""
    df = df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df

    def fmt(col, v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "-"
            return f"%{v * 100:.1f}" if col in pct else f"{v:.{digits}f}"
        return str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# --------------------------------------------------------------------------- deneyler

def diagnostics(D: Dataset, args) -> dict:
    """Paket düzeyindeki verinin üst sınırlarını gösterir: aynı özellik vektörü farklı etiketlerde
    görünüyorsa hiçbir model onları ayıramaz; eğitimdeki satırın aynısı testte de varsa model ezberden geçer."""
    h = row_hashes(D.X)
    labels_per_vec = pd.Series(D.y).groupby(h).nunique()
    conflict = np.isin(h, labels_per_vec.index[labels_per_vec > 1]).mean()
    rnd = random_split(D.y, args.test_orani, args.tohum)
    return {
        "veri_seti": D.name,
        "satir": len(h),
        "tekrar_eden_satir_orani": 1 - len(np.unique(h)) / len(h),
        "celiskili_satir_orani": conflict,
        "test_egitimde_aynen_var_blok": np.isin(h[~D.train], h[D.train]).mean(),
        "test_egitimde_aynen_var_rastgele": np.isin(h[~rnd], h[rnd]).mean(),
    }


def run_exp1(D: Dataset, args, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log(f"Deney 1: bilinen saldırılar ({D.name}, çok sınıflı)")
    X, y = D.X.to_numpy(), D.y
    labels = sorted(np.unique(y))
    rows, per_class, importance = [], [], pd.DataFrame()
    splits = {"blok": D.train, "rastgele": random_split(y, args.test_orani, args.tohum)}
    for split_name, tr in splits.items():
        for mname, model in make_models(args).items():
            t = time.time()
            model.fit(X[tr], y[tr])
            pred = model.predict(X[~tr])
            yt = y[~tr]
            rows.append({"bolme": split_name, "model": mname, "dogruluk": accuracy_score(yt, pred),
                         "macro_f1": f1_score(yt, pred, labels=labels, average="macro", zero_division=0),
                         "sure_sn": round(time.time() - t, 1)})
            p, r, f, s = precision_recall_fscore_support(yt, pred, labels=labels, zero_division=0)
            per_class += [{"bolme": split_name, "model": mname, "sinif": c, "kesinlik": p[i], "duyarlilik": r[i],
                           "f1": f[i], "destek": int(s[i])} for i, c in enumerate(labels)]
            if split_name == "blok":
                cm = confusion_matrix(yt, pred, labels=labels)
                pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / f"deney1_karisiklik_{mname}.csv")
                plot_confusion(cm, labels, f"Deney 1 · {D.name} · {mname} (blok bölme)", out / f"deney1_karisiklik_{mname}.png")
                if mname == "RandomForest":
                    importance = (pd.Series(model.feature_importances_, index=D.X.columns, name="onem")
                                  .sort_values(ascending=False).rename_axis("ozellik").reset_index())
                    importance.to_csv(out / "deney1_ozellik_onemi.csv", index=False)
            log(f"  {split_name:8s} {mname:12s} macro-F1={rows[-1]['macro_f1']:.3f}")

    if D.leak.shape[1]:  # 1c: sızıntı sütunları eklenirse
        XL = pd.concat([D.X, D.leak], axis=1)
        model = make_models(args)["RandomForest"]
        model.fit(XL.to_numpy()[D.train], y[D.train])
        pred = model.predict(XL.to_numpy()[~D.train])
        rows.append({"bolme": "blok", "model": "RandomForest + sızıntı sütunları",
                     "dogruluk": accuracy_score(y[~D.train], pred),
                     "macro_f1": f1_score(y[~D.train], pred, labels=labels, average="macro", zero_division=0),
                     "sure_sn": np.nan})
        (pd.Series(model.feature_importances_, index=XL.columns, name="onem").sort_values(ascending=False)
         .rename_axis("ozellik").reset_index().assign(sizinti=lambda d: d["ozellik"].isin(D.leak.columns))
         .to_csv(out / "deney1c_sizinti_ozellik_onemi.csv", index=False))
        log(f"  sızıntı sütunlarıyla RandomForest macro-F1={rows[-1]['macro_f1']:.3f}")

    summary, per_class = pd.DataFrame(rows), pd.DataFrame(per_class)
    summary.to_csv(out / "deney1_ozet.csv", index=False)
    per_class.to_csv(out / "deney1_sinif_bazinda.csv", index=False)
    return summary, per_class, importance


def run_exp2(D: Dataset, args, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log(f"Deney 2: bilinmeyen saldırılar ({D.name}, Leave-One-Attack-Type-Out)")
    X, y = D.X.to_numpy(), D.y
    legit_test = np.flatnonzero((y == LEGIT) & ~D.train)
    iso, thr = fit_isolation_forest(D, args)
    rows = []
    for held in sorted(set(y) - {LEGIT}):
        held_idx = np.flatnonzero(y == held)          # model bu saldırının hiçbir örneğini görmedi
        tr = np.flatnonzero(D.train & (y != held))   # normal trafik + diğer saldırılar
        te = np.concatenate([legit_test, held_idx])
        y_te = np.r_[np.zeros(len(legit_test)), np.ones(len(held_idx))]
        for mname, model in make_models(args).items():
            model.fit(X[tr], (y[tr] != LEGIT).astype(int))
            score = model.predict_proba(X[te])[:, 1]
            rows.append({"disarida_birakilan": held, "model": mname, **binary_metrics(y_te, score >= 0.5, score)})
        score = -iso.score_samples(X[te])
        rows.append({"disarida_birakilan": held, "model": "IsolationForest", **binary_metrics(y_te, score >= thr, score)})
        log(f"  {held:11s} yakalama: " + ", ".join(f"{r['model']}={r['yakalama_orani']:.1%}" for r in rows[-3:]))

    long = pd.DataFrame(rows)
    detect = long.pivot(index="disarida_birakilan", columns="model", values="yakalama_orani")
    fpr = long.pivot(index="disarida_birakilan", columns="model", values="yanlis_alarm_orani")
    long.to_csv(out / "deney2_loao.csv", index=False)
    detect.to_csv(out / "deney2_ana_tablo_yakalama.csv")
    fpr.to_csv(out / "deney2_yanlis_alarm.csv")
    plot_grouped(detect, f"Deney 2 · {D.name} · Görülmeyen saldırıyı yakalama oranı", "Yakalama oranı (%)",
                 out / "deney2_yakalama.png")
    return long, detect, fpr


def run_exp3(src: Dataset, dst: Dataset, args, out: Path) -> tuple[list, list, list]:
    direction = f"{src.name} -> {dst.name}"
    log(f"Deney 3: veri setleri arası ({direction})")
    classes = sorted(set(src.y) & set(dst.y))
    Xs, Xd = src.X.to_numpy(), dst.X.to_numpy()
    s_tr = src.train & np.isin(src.y, classes)
    d_tr = dst.train & np.isin(dst.y, classes)
    d_te = ~dst.train & np.isin(dst.y, classes)
    y_te = dst.y[d_te]
    settings = {f"alan içi ({dst.name} -> {dst.name})": (Xd[d_tr], dst.y[d_tr], dst),
                f"alanlar arası ({direction})": (Xs[s_tr], src.y[s_tr], src)}
    rows, per_class, per_attack = [], [], []

    for setting, (Xtr, ytr, train_set) in settings.items():
        for mname, model in make_models(args).items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xd[d_te])
            rows.append({"yon": direction, "gorev": "çok sınıflı", "ayar": setting, "model": mname,
                         "macro_f1": f1_score(y_te, pred, labels=classes, average="macro", zero_division=0),
                         "dogruluk": accuracy_score(y_te, pred)})
            _, r, _, _ = precision_recall_fscore_support(y_te, pred, labels=classes, zero_division=0)
            per_class += [{"yon": direction, "ayar": setting, "model": mname, "sinif": c, "duyarlilik": r[i]}
                          for i, c in enumerate(classes)]
            if train_set is src and mname == "RandomForest":
                cm = confusion_matrix(y_te, pred, labels=classes)
                tag = f"{src.name}_to_{dst.name}".replace("-", "")
                pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / f"deney3_karisiklik_{tag}.csv")
                plot_confusion(cm, classes, f"Deney 3 · RandomForest · {direction}", out / f"deney3_karisiklik_{tag}.png")

        ybin_te = (y_te != LEGIT).astype(int)
        scored = {}
        for mname, model in make_models(args).items():
            model.fit(Xtr, (ytr != LEGIT).astype(int))
            score = model.predict_proba(Xd[d_te])[:, 1]
            scored[mname] = (score >= 0.5, score)
        iso, thr = fit_isolation_forest(train_set, args)
        score = -iso.score_samples(Xd[d_te])
        scored["IsolationForest"] = (score >= thr, score)
        for mname, (pred, score) in scored.items():
            rows.append({"yon": direction, "gorev": "ikili (saldırı mı?)", "ayar": setting, "model": mname,
                         **binary_metrics(ybin_te, pred, score)})
            per_attack += [{"yon": direction, "ayar": setting, "model": mname, "saldiri": c,
                            "yakalama_orani": pred[y_te == c].mean()} for c in classes if c != LEGIT]
        last = rows[-len(scored) - 2:]
        log(f"  {setting}: " + ", ".join(
            f"{r['model']} {'F1' if r['gorev'] == 'çok sınıflı' else 'yakalama'}="
            f"{r['macro_f1'] if r['gorev'] == 'çok sınıflı' else r['yakalama_orani']:.3f}" for r in last))
    return rows, per_class, per_attack


# --------------------------------------------------------------------------- rapor

def write_report(out: Path, A: Dataset, B: Dataset | None, common: list[str], diag: pd.DataFrame,
                 e1: tuple, e2: tuple, e3: pd.DataFrame | None, args) -> None:
    s1, pc1, imp = e1
    _, detect, fpr = e2
    sets = [D for D in (A, B) if D is not None]
    counts = pd.concat([pd.Series(D.y).value_counts().rename(D.name) for D in sets], axis=1).fillna(0).astype(int)
    mapping = pd.concat([pd.DataFrame({"veri_seti": D.name, "orijinal_etiket": D.raw_labels.astype(str),
                                       "ortak_etiket": D.y}) for D in sets])
    mapping = mapping.value_counts().rename("satir").reset_index().sort_values(["veri_seti", "ortak_etiket"])

    L = [f"# MQTT saldırı tespiti deney raporu\n",
         f"Oluşturulma: {time.strftime('%d.%m.%Y %H:%M')} · sınıf başına en fazla {args.sinif_basina_max or 'tüm'} satır · "
         f"test oranı %{args.test_orani * 100:.0f} · RF ağaç sayısı {args.agac} · IF hedef yanlış alarm %{args.hedef_fpr * 100:g}\n",
         "## Veri setleri\n", md_table(counts.rename_axis("sinif")), "",
         "Etiket eşlemesi (yanlış eşleme varsa `LABEL_RULES` listesini düzelt):\n", md_table(mapping), "",
         f"**Ortak özellikler ({len(common)}):** " + ", ".join(f"`{c}`" for c in common), "",
         "Tüm sütunların durumu: `ozellik_karsilastirma.csv`\n",
         "## Teşhis: paket düzeyindeki verinin sınırları\n", md_table(diag, pct=tuple(c for c in diag.columns if "oran" in c or "var" in c)), "",
         "- **çelişkili satır oranı:** Aynı özellik vektörü birden fazla etiketle görünüyor. Bu satırları hiçbir model ayıramaz.",
         "- **test eğitimde aynen var:** Test satırının birebir aynısı eğitimde de var. Oran yüksekse başarı kısmen ezberden gelir. Blok ve rastgele bölmeyi karşılaştır.\n",
         f"## Deney 1 · Bilinen saldırılar ({A.name})\n", md_table(s1, pct=("dogruluk", "macro_f1"), digits=1), "",
         "Sınıf bazında (blok bölme):\n",
         md_table(pc1[pc1["bolme"] == "blok"].drop(columns="bolme"), pct=("kesinlik", "duyarlilik", "f1")), ""]
    if not imp.empty:
        L += ["En önemli 10 özellik (RandomForest):\n", md_table(imp.head(10)), ""]
    L += [f"## Deney 2 · Bilinmeyen saldırılar ({A.name}, Leave-One-Attack-Type-Out)\n",
          "Satırdaki saldırı eğitimde hiç gösterilmedi. Hücreler o saldırının **yakalanma oranı**.\n",
          md_table(detect, pct=tuple(detect.columns)), "",
          "Normal trafikte yanlış alarm oranı:\n", md_table(fpr, pct=tuple(fpr.columns)), "",
          f"Isolation Forest yalnızca normal trafikle eğitildi; eşiği normal trafikte %{args.hedef_fpr * 100:g} yanlış alarm verecek şekilde seçildi. "
          "DT ve RF 0,5 olasılık eşiğini kullanır; eşikten bağımsız karşılaştırma için `deney2_loao.csv` içindeki ROC-AUC ve PR-AUC değerlerine bak.\n"]
    if e3 is not None and not e3.empty:
        multi = e3[e3["gorev"] == "çok sınıflı"][["yon", "ayar", "model", "macro_f1"]]
        binary = e3[e3["gorev"] != "çok sınıflı"][["yon", "ayar", "model", "yakalama_orani", "yanlis_alarm_orani", "roc_auc"]]
        L += ["## Deney 3 · Veri setleri arası genelleme\n",
              "**Alan içi:** Hedef veri setinin kendi eğitim kısmıyla eğitildi. **Alanlar arası:** Diğer veri setiyle eğitildi. "
              "Test kümesi ikisinde de aynı: hedef veri setinin test kısmı.\n",
              "Çok sınıflı (macro-F1):\n", md_table(multi, pct=("macro_f1",)), "",
              "İkili, saldırı mı değil mi:\n", md_table(binary, pct=("yakalama_orani", "yanlis_alarm_orani")), "",
              "Saldırı ve sınıf bazındaki ayrıntılar: `deney3_saldiri_bazinda.csv`, `deney3_sinif_bazinda.csv`\n"]
    L += ["## Notlar\n",
          "- Veriler paket düzeyinde. Tek bir flood paketi normal bir PUBLISH paketinden farksız olabilir. Flood ve DoS saldırılarında düşük yakalama oranı bu yüzden beklenen bir bulgu.",
          "- Sızıntı sütunları (IP, port, zaman, client ID...) özellik olarak kullanılmadı. Kullanılsalardı ne olacağı Deney 1'deki `RandomForest + sızıntı sütunları` satırında.",
          "- Bu rapordaki sayılar yalnızca bu çalıştırmaya ait. Farklı tohum (`--tohum`) ile tekrar çalıştırıp sonuçların kararlılığına bak."]
    (out / "rapor.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- ana akış

def parse_args():
    p = argparse.ArgumentParser(
        description="MQTT saldırı tespiti: bilinen, bilinmeyen ve veri setleri arası deneyler",
        epilog="Örnek: python mqtt_deneyler.py --a veri/MQTTEEB-D/raw.csv "
               "--b veri/MQTTset/train70_reduced.csv veri/MQTTset/test30_reduced.csv")
    p.add_argument("--a", nargs="+", required=True, metavar="YOL", help="A veri seti (MQTTEEB-D): CSV dosya(lar)ı ya da klasör")
    p.add_argument("--b", nargs="+", metavar="YOL", help="B veri seti (MQTTset): CSV dosya(lar)ı ya da klasör. Verilmezse deney 3 atlanır")
    p.add_argument("--a-adi", default="MQTTEEB-D", help="Raporda A'nın adı")
    p.add_argument("--b-adi", default="MQTTset", help="Raporda B'nin adı")
    p.add_argument("--etiket-a", help="A'daki etiket sütununun adı (otomatik bulunamazsa)")
    p.add_argument("--etiket-b", help="B'deki etiket sütununun adı (otomatik bulunamazsa)")
    p.add_argument("--esle", nargs="*", default=[], metavar="B_SUTUNU=A_SUTUNU",
                   help="Aynı alanın adı iki veri setinde farklıysa B'dekini A'daki adla eşle")
    p.add_argument("--ekstra-sizinti", nargs="*", default=[], metavar="REGEX", help="Sızıntı sayılacak ek sütun kalıpları")
    p.add_argument("--tut", nargs="*", default=[], metavar="SUTUN", help="Sızıntı kalıbına uysa da özellik olarak tutulacak sütunlar")
    p.add_argument("--sinif-basina-max", type=int, default=50000, help="Her sınıftan en fazla kaç satır (0 = hepsi). Varsayılan 50000")
    p.add_argument("--test-orani", type=float, default=0.3)
    p.add_argument("--hedef-fpr", type=float, default=0.01, help="Isolation Forest eşiği için hedef yanlış alarm oranı")
    p.add_argument("--agac", type=int, default=100, help="Random Forest ağaç sayısı")
    p.add_argument("--tohum", type=int, default=42)
    p.add_argument("--is-sayisi", type=int, default=-1, help="Paralel iş sayısı (-1 = tüm çekirdekler)")
    p.add_argument("--cikti", default="sonuclar", help="Sonuç klasörü")
    p.add_argument("--tek-yon", action="store_true", help="Deney 3'te sadece A -> B yönünü çalıştır")
    p.add_argument("--hizli", action="store_true", help="Hızlı deneme: sınıf başına 20000 satır, 50 ağaç")
    p.add_argument("--sadece-ozellikler", action="store_true", help="Sadece ortak özellik karşılaştırmasını yap ve çık")
    p.add_argument("--zorla", action="store_true", help="SMOTE/normalize görünen dosyaları da kabul et (önerilmez)")
    args = p.parse_args()
    args.tut = [norm_col(c) for c in args.tut]
    if args.hizli:
        args.sinif_basina_max = min(args.sinif_basina_max or 20000, 20000)
        args.agac = min(args.agac, 50)
    return args


def main() -> None:
    args = parse_args()
    out = Path(args.cikti)
    out.mkdir(parents=True, exist_ok=True)

    rename = {}
    for pair in args.esle:
        if "=" not in pair:
            sys.exit(f"[HATA] --esle biçimi B_SUTUNU=A_SUTUNU olmalı: {pair}")
        b, a = pair.split("=", 1)
        rename[norm_col(b)] = norm_col(a)

    A = load_dataset(args.a_adi, args.a, args.etiket_a, args)
    B = load_dataset(args.b_adi, args.b, args.etiket_b, args, rename) if args.b else None
    common = align_features(A, B, out)
    log(f"Ortak özellik sayısı: {len(common)} -> {common}")
    if not common:
        sys.exit("[HATA] Ortak sayısal özellik yok. ozellik_karsilastirma.csv dosyasına bakıp --esle ile sütun adlarını eşle.")
    if len(common) < 5:
        log("[UYARI] Ortak özellik sayısı çok az; --esle ile adları farklı ama aynı anlamdaki sütunları eşlemeyi dene.")
    if args.sadece_ozellikler:
        log(f"Karşılaştırma tablosu: {out / 'ozellik_karsilastirma.csv'}")
        return

    for D in (A, B):
        if D is not None:
            D.X = D.X_all[common]
            D.train = block_split(D.y, args.test_orani)

    diag = pd.DataFrame([diagnostics(D, args) for D in (A, B) if D is not None])
    diag.to_csv(out / "teshis.csv", index=False)

    e1 = run_exp1(A, args, out)
    e2 = run_exp2(A, args, out)
    e3 = None
    if B is not None:
        rows, per_class, per_attack = run_exp3(A, B, args, out)
        if not args.tek_yon:
            r2, c2, a2 = run_exp3(B, A, args, out)
            rows, per_class, per_attack = rows + r2, per_class + c2, per_attack + a2
        e3 = pd.DataFrame(rows)
        e3.to_csv(out / "deney3_ozet.csv", index=False)
        pd.DataFrame(per_class).to_csv(out / "deney3_sinif_bazinda.csv", index=False)
        pd.DataFrame(per_attack).to_csv(out / "deney3_saldiri_bazinda.csv", index=False)
        multi = e3[e3["gorev"] == "çok sınıflı"].copy()
        multi["ayar"] = multi["ayar"].str.split(" (", regex=False).str[0]
        pv = multi.pivot_table(index=["yon", "ayar"], columns="model", values="macro_f1", sort=False)
        pv.index = [f"Test: {y.split(' -> ')[1]}\n{a}" for y, a in pv.index]
        plot_grouped(pv, "Deney 3 · Alan içi ve alanlar arası macro-F1", "macro-F1 (%)", out / "deney3_genelleme.png")

    write_report(out, A, B, common, diag, e1, e2, e3, args)
    log(f"Bitti. Rapor: {out / 'rapor.md'}")


if __name__ == "__main__":
    main()
