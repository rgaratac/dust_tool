"""Station pairing for the quantification of the net desert-dust load.

Implements Step 2 (quantification) and Step 3 (subtraction) of the
Commission technical guidance on natural contributions (desert dust, sec. 5.2):

* Option 1 - regional background (RB) subtraction
    The net dust load is computed at a rural/regional background site k as
    PMDD_ik = PMR_ik - P50(non-dust days, +/-15 calendar days, day 0 excluded)
    and subtracted from every target site j inside its representative area:
    PMUS_ij = PMU_ij - PMDD_ik.

* Option 2 - site-based "self-calibration"
    Background sites (urban, suburban, rural) use their own record with a
    +/-3 calendar-day window. Traffic and industrial (hotspot) sites never
    self-calibrate: they take the net dust load of the nearest suitable
    urban/suburban background site.

Within each option the reference is the NEAREST suitable station. Pairings can
be changed by hand (``PairingEditor`` map widget or an overrides CSV); see
``apply_overrides``.

* 'auto' (recommended in the guidance) - Option 1 whenever a suitable RB site
  is available that day, otherwise Option 2.

Per-day fallback: every target gets a ranked list of candidate references. On
each dust day the first candidate with a valid net dust value is used, so a
gap at the nearest RB station falls through to the next RB station and, in
'auto' mode, to Option 2.

Suitability of a reference (Box 5) is approximated by geographical distance,
altitude difference and station classification. Meteorological and temporal
coherence cannot be checked from metadata; review the pairing table.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


@dataclass
class PairingConfig:
    method: str = "auto"                 # 'option1', 'option2', 'auto'
    # 'auto' only: 'option1' = RB first whenever one is suitable (guidance);
    #              'nearest' = nearest suitable reference of any option
    auto_priority: str = "option1"
    # --- Option 1: regional background reference ---
    rb_areas: tuple = ("rural", "rural-regional", "rural-remote")
    rb_max_distance_km: float = 75.0     # RB representativeness ~100x100 km
    rb_max_altitude_diff_m: float = 500.0
    rb_half_window_days: int = 15        # +/-15 days, P50
    rb_min_clean_days: int = 10
    # --- Option 2: self-calibration / urban background reference ---
    ub_areas: tuple = ("urban", "suburban")
    selfcal_areas: tuple = ("urban", "suburban", "rural", "rural-regional",
                            "rural-remote", "rural-nearcity")
    ub_max_distance_km: float = 60.0     # traffic/industrial -> nearest UB/SUB background
    prefer_same_municipality: bool = False  # True: same municipality first, then distance
    ub_max_altitude_diff_m: float = 300.0
    ub_half_window_days: int = 3         # +/-3 days, P50
    ub_min_clean_days: int = 3
    # Option 2, consecutive dust days: use the N clean days before and after
    # the episode when the calendar window has too few clean days.
    episode_fallback_neighbors: int = 3
    # --- general ---
    percentile: float = 50.0
    max_candidates: int = 3              # ranked references kept per option
    combine: str = "nearest"             # 'nearest' or 'mean' of available refs
    same_country_only: bool = False
    apply_on: str = "all_dust_days"      # or 'dust_exceedance_days'

    def as_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Metadata and classification
# ---------------------------------------------------------------------------
def load_station_metadata(metadata_path: str) -> pd.DataFrame:
    """One row per Samplingpoint with type, area, municipality, coordinates."""
    meta = pd.read_csv(metadata_path, low_memory=False)
    meta["Samplingpoint"] = (
        meta["Air Quality Station EoI Code"].str[:2] + "/" + meta["Sampling Point Id"]
    )
    keep = {
        "Samplingpoint": "Samplingpoint",
        "Air Quality Station EoI Code": "station_code",
        "Air Quality Station Name": "station_name",
        "Air Quality Station Type": "station_type",
        "Air Quality Station Area": "station_area",
        "Municipality": "municipality",
        "Latitude": "Latitude",
        "Longitude": "Longitude",
        "Altitude": "Altitude",
    }
    cols = [c for c in keep if c in meta.columns]
    out = (meta[cols].rename(columns=keep)
           .drop_duplicates(subset="Samplingpoint", keep="last"))
    for c in ("station_type", "station_area"):
        if c in out:
            out[c] = out[c].astype(str).str.strip().str.lower()
    return out.reset_index(drop=True)


def classify_station(station_type: str, station_area: str, cfg: PairingConfig) -> str:
    """RB = regional background, UB = urban/suburban background,
    OB = other background (e.g. rural-nearcity), HS = traffic/industrial."""
    t = str(station_type).lower()
    a = str(station_area).lower()
    if t in ("traffic", "industrial"):
        return "HS"
    if t == "background":
        if a in cfg.rb_areas:
            return "RB"
        if a in cfg.ub_areas:
            return "UB"
        return "OB"
    return "unknown"


def build_station_table(df: pd.DataFrame, metadata: pd.DataFrame,
                        cfg: PairingConfig) -> pd.DataFrame:
    """Stations present in ``df`` with metadata and pairing class."""
    st = (df[["Samplingpoint"]].astype({"Samplingpoint": str})
          .drop_duplicates().reset_index(drop=True))
    meta = metadata.copy()
    meta["Samplingpoint"] = meta["Samplingpoint"].astype(str)
    # coordinates already on df take precedence (they were used for CAMS)
    coord_cols = [c for c in ("Latitude", "Longitude", "Altitude") if c in df.columns]
    if coord_cols:
        coords = (df[["Samplingpoint"] + coord_cols].astype({"Samplingpoint": str})
                  .drop_duplicates("Samplingpoint"))
        st = st.merge(coords, on="Samplingpoint", how="left")
        meta = meta.drop(columns=[c for c in coord_cols if c in meta.columns])
    st = st.merge(meta, on="Samplingpoint", how="left")
    st["country"] = st["Samplingpoint"].str[:2]
    st["station_class"] = [
        classify_station(t, a, cfg)
        for t, a in zip(st.get("station_type"), st.get("station_area"))
    ]
    st["municipality_key"] = st.get("municipality", pd.Series(index=st.index, dtype=str)) \
        .astype(str).str.strip().str.lower().replace({"nan": np.nan, "": np.nan})
    return st


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1)[:, None] * np.cos(lat2)[None, :] * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _candidates(targets: pd.DataFrame, refs: pd.DataFrame, max_km: float,
                max_alt: float, cfg: PairingConfig, prefer_same_municipality: bool = False,
                max_candidates: int | None = None) -> pd.DataFrame:
    """Long table target -> candidate reference, ranked."""
    if targets.empty or refs.empty:
        return pd.DataFrame(columns=["target", "reference", "distance_km",
                                     "altitude_diff_m", "same_municipality"])
    rows = []
    rlat = refs["Latitude"].to_numpy(float)
    rlon = refs["Longitude"].to_numpy(float)
    ralt = refs["Altitude"].to_numpy(float)
    rid = refs["Samplingpoint"].to_numpy()
    rcountry = refs["country"].to_numpy()
    rmun = refs["municipality_key"].to_numpy(object)
    chunk = 500
    for s in range(0, len(targets), chunk):
        t = targets.iloc[s:s + chunk]
        d = _haversine_km(t["Latitude"].to_numpy(float), t["Longitude"].to_numpy(float),
                          rlat, rlon)
        dalt = np.abs(t["Altitude"].to_numpy(float)[:, None] - ralt[None, :])
        for i, (_, trow) in enumerate(t.iterrows()):
            ok = (d[i] <= max_km) & ~np.isnan(d[i])
            # missing altitude -> do not reject on altitude
            ok &= np.isnan(dalt[i]) | (dalt[i] <= max_alt)
            ok &= rid != trow["Samplingpoint"]
            if cfg.same_country_only:
                ok &= rcountry == trow["country"]
            same_mun = (rmun == trow["municipality_key"]) & pd.notna(trow["municipality_key"])
            idx = np.where(ok)[0]
            if idx.size == 0:
                continue
            # rank: same municipality first (only relevant for UB), then distance
            key = np.lexsort((d[i][idx], ~same_mun[idx] if prefer_same_municipality
                              else np.zeros(idx.size, bool)))
            idx = idx[key][: (max_candidates or cfg.max_candidates)]
            for j in idx:
                rows.append((trow["Samplingpoint"], rid[j], float(d[i][j]),
                             float(dalt[i][j]), bool(same_mun[j])))
    return pd.DataFrame(rows, columns=["target", "reference", "distance_km",
                                       "altitude_diff_m", "same_municipality"])


def pair_stations(stations: pd.DataFrame, cfg: PairingConfig) -> pd.DataFrame:
    """Ranked reference list per target station.

    Columns: target, rank, reference, option, distance_km, altitude_diff_m,
    same_municipality, target_class, reference_class.
    """
    if cfg.method not in ("option1", "option2", "auto"):
        raise ValueError("method must be 'option1', 'option2' or 'auto'")
    st = stations.dropna(subset=["Latitude", "Longitude"])
    cls = st.set_index("Samplingpoint")["station_class"]
    parts = []

    if cfg.method in ("option1", "auto"):
        rb = st[st["station_class"] == "RB"]
        c1 = _candidates(st, rb, cfg.rb_max_distance_km, cfg.rb_max_altitude_diff_m, cfg)
        # an RB target is its own (best) regional reference
        own = st[st["station_class"] == "RB"]
        own_rows = pd.DataFrame({
            "target": own["Samplingpoint"], "reference": own["Samplingpoint"],
            "distance_km": 0.0, "altitude_diff_m": 0.0, "same_municipality": True})
        c1 = pd.concat([own_rows, c1], ignore_index=True)
        c1["option"] = "option1_RB"
        c1["order"] = 0
        parts.append(c1)

    if cfg.method in ("option2", "auto"):
        # background targets: self-calibration
        bg = st[st["station_class"].isin(["UB", "RB", "OB"])
                & st["station_area"].isin(cfg.selfcal_areas)]
        self_rows = pd.DataFrame({
            "target": bg["Samplingpoint"], "reference": bg["Samplingpoint"],
            "distance_km": 0.0, "altitude_diff_m": 0.0, "same_municipality": True})
        self_rows["option"] = "option2_self"
        self_rows["order"] = 1
        parts.append(self_rows)
        # hotspot targets: nearest suitable urban/suburban background (by distance)
        hs = st[st["station_class"] == "HS"]
        ub = st[st["station_class"] == "UB"]
        c_ub = _candidates(hs, ub, cfg.ub_max_distance_km, cfg.ub_max_altitude_diff_m,
                           cfg, prefer_same_municipality=cfg.prefer_same_municipality)
        c_ub["option"] = "option2_UB"
        c_ub["order"] = 2
        parts.append(c_ub)

    pairs = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else \
        pd.DataFrame(columns=["target", "reference", "option", "order"])
    if pairs.empty:
        return pairs
    pairs = pairs.drop_duplicates(subset=["target", "reference", "option"])
    pairs["manual"] = False
    if cfg.method == "auto" and cfg.auto_priority == "nearest":
        pairs = pairs.sort_values(["target", "distance_km", "order"])
    else:
        pairs = pairs.sort_values(["target", "order", "distance_km"])
    pairs["rank"] = pairs.groupby("target").cumcount()
    pairs["target_class"] = pairs["target"].map(cls)
    pairs["reference_class"] = pairs["reference"].map(cls)
    return pairs.drop(columns="order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Net dust load at a reference station
# ---------------------------------------------------------------------------
def _naive_day(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.floor("D")


def net_dust_series(values: np.ndarray, dust: np.ndarray, half_window: int,
                    min_clean: int, percentile: float = 50.0,
                    fallback_neighbors: int = 0, only: np.ndarray | None = None):
    """Background and net dust for every day of a gap-free daily series.

    Background on day i = P(percentile) of non-dust days in [i-W, i+W],
    day i always excluded. If fewer than ``min_clean`` such days exist and
    ``fallback_neighbors`` > 0, the N clean days before and after the episode
    are used instead (guidance, Option 2, consecutive dust days).
    ``only`` (bool mask) restricts the computation to the days that are needed.
    """
    n = values.size
    clean = (~dust) & ~np.isnan(values)
    clean_idx = np.flatnonzero(clean)
    bg = np.full(n, np.nan)
    n_used = np.zeros(n, dtype=int)
    used_fallback = np.zeros(n, dtype=bool)
    days = range(n) if only is None else np.flatnonzero(only)
    for i in days:
        lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
        m = clean[lo:hi].copy()
        if lo <= i < hi:
            m[i - lo] = False
        w = values[lo:hi][m]
        if w.size >= min_clean:
            bg[i] = np.percentile(w, percentile)
            n_used[i] = w.size
        elif fallback_neighbors > 0 and clean_idx.size:
            p = np.searchsorted(clean_idx, i, side="left")
            before = clean_idx[max(0, p - fallback_neighbors):p]
            after = clean_idx[p:p + fallback_neighbors]
            after = after[after != i]
            sel = np.concatenate([before, after])
            if sel.size >= min(min_clean, 2 * fallback_neighbors):
                bg[i] = np.percentile(values[sel], percentile)
                n_used[i] = sel.size
                used_fallback[i] = True
    net = np.clip(values - bg, 0, None)
    return bg, net, n_used, used_fallback


def reference_net_dust(df: pd.DataFrame, references, cfg: PairingConfig,
                       option_by_ref: dict, time_col="day", value_col="daily_mean",
                       needed_days=None) -> pd.DataFrame:
    """Net dust per (reference, option, day). RB refs: +/-15 d, others +/-3 d."""
    out = []
    sub = df[df["Samplingpoint"].astype(str).isin(set(references))]
    for (ref), g in sub.groupby(sub["Samplingpoint"].astype(str), observed=True):
        g = g.assign(_d=_naive_day(g[time_col])).dropna(subset=["_d"])
        g = g.groupby("_d").agg(v=(value_col, "mean"), f=("dust_flag", "max"))
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(idx)
        vals = g["v"].to_numpy(float)
        dust = g["f"].fillna(False).to_numpy(bool)
        only = None if needed_days is None else idx.isin(needed_days)
        for opt in option_by_ref.get(ref, ()):
            if opt == "option1_RB":
                W, mn, fb = cfg.rb_half_window_days, cfg.rb_min_clean_days, 0
            else:
                W, mn, fb = (cfg.ub_half_window_days, cfg.ub_min_clean_days,
                             cfg.episode_fallback_neighbors)
            bg, net, nu, ufb = net_dust_series(vals, dust, W, mn, cfg.percentile, fb, only)
            out.append(pd.DataFrame({
                "reference": ref, "option": opt, "_d": idx,
                "ref_value": vals, "ref_dust_flag": dust, "ref_background": bg,
                "ref_net_dust": net, "ref_n_clean_days": nu,
                "ref_episode_fallback": ufb}))
    if not out:
        return pd.DataFrame(columns=["reference", "option", "_d", "ref_net_dust"])
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Full deduction with pairing
# ---------------------------------------------------------------------------
def apply_paired_deduction(df_all: pd.DataFrame, df_year: pd.DataFrame,
                           pairs: pd.DataFrame, cfg: PairingConfig,
                           time_col="day", value_col="daily_mean") -> pd.DataFrame:
    """Return ``df_year`` with paired net dust and corrected concentration.

    New / overwritten columns:
      Dust_contribution   net dust subtracted (0 on non-dust days)
      corrected_pollutant daily_mean - Dust_contribution
      pollutant_median    background (P50) at the reference used
      dust_method         option used that day ('none' = no valid reference)
      dust_reference      reference Samplingpoint (or 'mean of n' if combine='mean')
      dust_ref_distance_km, dust_ref_manual (pairing set by hand),
      dust_day_confirmed (net dust > 0)
    """
    out = df_year.copy()
    out["_d"] = _naive_day(out[time_col])
    sp = out["Samplingpoint"].astype(str)
    dust = out["dust_flag"].fillna(False).astype(bool)
    if cfg.apply_on == "dust_exceedance_days":
        dust &= out["Exceedance"].fillna(False).astype(bool)

    option_by_ref = pairs.groupby("reference")["option"].agg(lambda x: set(
        "option1_RB" if o == "option1_RB" else "option2" for o in x)).to_dict()
    needed = pd.DatetimeIndex(out.loc[dust, "_d"].unique())
    net = reference_net_dust(df_all, pairs["reference"].unique(), cfg,
                             option_by_ref, time_col, value_col, needed_days=needed)
    pairs2 = pairs.assign(opt_key=np.where(pairs["option"] == "option1_RB",
                                           "option1_RB", "option2"))

    tgt = out.loc[dust, ["_d"]].assign(target=sp[dust].values, _row=out.index[dust])
    m = (tgt.merge(pairs2, on="target", how="inner")
            .merge(net.rename(columns={"option": "opt_key"}),
                   on=["reference", "opt_key", "_d"], how="left"))
    m = m[m["ref_net_dust"].notna()]

    if cfg.combine == "mean" and not m.empty:
        # best option group with data that day, then mean over its references
        m["group"] = m["option"].map({"option1_RB": 0, "option2_self": 1, "option2_UB": 2})
        m.loc[m["manual"].astype(bool), "group"] = -1
        best = m.groupby("_row")["group"].transform("min")
        m = m[m["group"] == best]
        chosen = m.groupby("_row").agg(
            ref_net_dust=("ref_net_dust", "mean"),
            ref_background=("ref_background", "mean"),
            option=("option", "first"),
            manual=("manual", "first"),
            reference=("reference", lambda r: r.iloc[0] if len(r) == 1 else f"mean of {len(r)}"),
            distance_km=("distance_km", "mean"))
    else:
        chosen = (m.sort_values(["_row", "rank"]).groupby("_row").first()
                  [["ref_net_dust", "ref_background", "option", "reference", "distance_km", "manual"]])

    out["Dust_contribution"] = np.float32(0.0)
    out["pollutant_median"] = np.float32(np.nan)
    out["dust_method"] = np.where(dust, "none", "")
    out["dust_reference"] = ""
    out["dust_ref_distance_km"] = np.nan
    out["dust_ref_manual"] = False
    if not chosen.empty:
        rows = chosen.index
        val = out.loc[rows, value_col].astype(float)
        # never subtract more than was measured at the target
        out.loc[rows, "Dust_contribution"] = np.minimum(chosen["ref_net_dust"], val).astype("float32")
        out.loc[rows, "pollutant_median"] = chosen["ref_background"].astype("float32")
        out.loc[rows, "dust_method"] = chosen["option"].values
        out.loc[rows, "dust_reference"] = chosen["reference"].astype(str).values
        out.loc[rows, "dust_ref_distance_km"] = chosen["distance_km"].values
        out.loc[rows, "dust_ref_manual"] = chosen["manual"].astype(bool).values
    out["Dust_contribution"] = out["Dust_contribution"].fillna(0).astype("float32")
    out["corrected_pollutant"] = (out[value_col].astype(float)
                                  - out["Dust_contribution"]).astype("float32")
    out["dust_day_confirmed"] = out["Dust_contribution"] > 0
    return out.drop(columns="_d")


def pairing_summary(pairs: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """One row per station: class, first-choice reference and option."""
    first = pairs[pairs["rank"] == 0].set_index("target")
    s = stations.set_index("Samplingpoint")[
        [c for c in ("station_name", "station_type", "station_area", "municipality",
                     "station_class", "Latitude", "Longitude", "Altitude") if c in stations]]
    s = s.join(first[["reference", "option", "distance_km", "altitude_diff_m", "manual"]]
               .rename(columns={"reference": "primary_reference", "option": "primary_option"}))
    s["n_candidates"] = pairs.groupby("target").size().reindex(s.index).fillna(0).astype(int)
    s["primary_option"] = s["primary_option"].fillna("no_reference")
    s["manual"] = s["manual"].fillna(False).astype(bool)
    return s.reset_index()


# ---------------------------------------------------------------------------
# Manual changes to the pairing
# ---------------------------------------------------------------------------
OVERRIDE_COLUMNS = ["Samplingpoint", "reference", "action"]
NO_REFERENCE = "none"   # reference value meaning: do not correct this station


def empty_overrides() -> pd.DataFrame:
    return pd.DataFrame(columns=OVERRIDE_COLUMNS)


def load_overrides(path: str | None) -> pd.DataFrame:
    """Read a pairing overrides CSV (Samplingpoint, reference, action).

    action = 'assign'            -> Samplingpoint uses ``reference``
                                    (``reference = 'none'``: not corrected)
    action = 'exclude_reference' -> station ``Samplingpoint`` is not used as a
                                    reference for other stations
    """
    import os
    if not path or not os.path.exists(path):
        return empty_overrides()
    ov = pd.read_csv(path, dtype=str).fillna("")
    if "action" not in ov:
        ov["action"] = "assign"
    return ov[OVERRIDE_COLUMNS]


def _manual_option(target: str, ref: str, cls: pd.Series) -> str:
    if cls.get(ref) == "RB":
        return "option1_RB"
    return "option2_self" if ref == target else "option2_UB"


def apply_overrides(pairs: pd.DataFrame, stations: pd.DataFrame,
                    overrides: pd.DataFrame | None) -> pd.DataFrame:
    """Pairs table with the manual changes applied and ranks recomputed.

    A manual reference becomes rank 0; the automatic candidates stay behind it
    as back-ups for days on which the manual reference has no data.
    """
    if overrides is None or len(overrides) == 0:
        return pairs.copy()
    out = pairs.copy()
    st = stations.set_index("Samplingpoint")
    cls = st["station_class"]
    excl = set(overrides.loc[overrides["action"] == "exclude_reference", "Samplingpoint"])
    if excl:
        out = out[~(out["reference"].isin(excl) & (out["reference"] != out["target"]))]
    assign = overrides[overrides["action"] == "assign"].drop_duplicates("Samplingpoint", keep="last")
    new_rows = []
    for tgt, ref in zip(assign["Samplingpoint"], assign["reference"]):
        if tgt not in st.index:
            continue
        if ref == NO_REFERENCE:
            out = out[out["target"] != tgt]
            continue
        if ref not in st.index:
            continue
        out = out[~((out["target"] == tgt) & (out["reference"] == ref))]
        d = float(_haversine_km(np.array([st.at[tgt, "Latitude"]]), np.array([st.at[tgt, "Longitude"]]),
                                np.array([st.at[ref, "Latitude"]]), np.array([st.at[ref, "Longitude"]]))[0, 0])
        new_rows.append(dict(target=tgt, reference=ref, option=_manual_option(tgt, ref, cls),
                             distance_km=d,
                             altitude_diff_m=abs(float(st.at[tgt, "Altitude"]) - float(st.at[ref, "Altitude"])),
                             same_municipality=bool(st.at[tgt, "municipality_key"] == st.at[ref, "municipality_key"]),
                             manual=True, rank=-1,
                             target_class=cls.get(tgt), reference_class=cls.get(ref)))
    if new_rows:
        out = pd.concat([pd.DataFrame(new_rows), out], ignore_index=True)
    out = out.sort_values(["target", "rank"]).reset_index(drop=True)
    out["rank"] = out.groupby("target").cumcount()
    return out


def current_assignment(pairs: pd.DataFrame) -> pd.DataFrame:
    """First-choice reference per target (rank 0)."""
    return pairs[pairs["rank"] == 0].set_index("target")


# ---------------------------------------------------------------------------
# Interactive map editor (Jupyter / Colab)
# ---------------------------------------------------------------------------
# Station colours by type (classification from DataExtract.csv)
_TYPE_STYLE = {
    "regional_bg": ("#1a9850", "Regional background (RB)"),
    "urban_bg":    ("#2c7fb8", "Urban background"),
    "suburban_bg": ("#41b6c4", "Suburban background"),
    "other_bg":    ("#a6d96a", "Other background (rural near city)"),
    "traffic":     ("#d7301f", "Traffic"),
    "industrial":  ("#8c6bb1", "Industrial"),
    "unknown":     ("#969696", "Unknown type"),
}
_LINK_AUTO = "rgba(60,60,60,0.75)"
_LINK_MANUAL = "#ff7f00"
_LINK_SELECTED = "#000000"
_CANDIDATE = "#ffd700"


def _type_category(station_class, station_type, station_area) -> str:
    t, a = str(station_type).lower(), str(station_area).lower()
    if t == "traffic":
        return "traffic"
    if t == "industrial":
        return "industrial"
    if station_class == "RB":
        return "regional_bg"
    if station_class == "UB":
        return "suburban_bg" if a == "suburban" else "urban_bg"
    if station_class == "OB":
        return "other_bg"
    return "unknown"


def _arrow_geometry(links, pos, where=0.55, frac=0.15, min_km=0.2, max_km=5.0, angle_deg=26):
    """Line coordinates and filled arrowheads for (start, end) links.

    The arrowhead sits at ``where`` (fraction of the link) and points to the
    end station, so it is never hidden under a marker.
    """
    lat_l, lon_l, lat_a, lon_a = [], [], [], []
    ang = np.radians(angle_deg)
    for a, b in links:
        la0, lo0 = pos.at[a, "Latitude"], pos.at[a, "Longitude"]
        la1, lo1 = pos.at[b, "Latitude"], pos.at[b, "Longitude"]
        lat_l += [la0, la1, None]
        lon_l += [lo0, lo1, None]
        kx = 111.32 * np.cos(np.radians((la0 + la1) / 2))
        ky = 110.57
        dx, dy = (lo1 - lo0) * kx, (la1 - la0) * ky
        L = float(np.hypot(dx, dy))
        if L == 0:
            continue
        ux, uy = dx / L, dy / L
        tx, ty = dx * where, dy * where                       # tip (km from start)
        size = min(max(frac * L, min_km), max_km)
        pts = [(tx, ty)]
        for sgn in (1, -1):
            c, s_ = np.cos(sgn * ang), np.sin(sgn * ang)
            bx, by = -(ux * c - uy * s_), -(ux * s_ + uy * c)  # rotated backwards
            pts.append((tx + bx * size, ty + by * size))
        pts.append((tx, ty))
        for x, y in pts:
            lat_a.append(la0 + y / ky)
            lon_a.append(lo0 + x / kx)
        lat_a.append(None)
        lon_a.append(None)
    return (lat_l, lon_l), (lat_a, lon_a)


def _map_figure_widget_class():
    import plotly.graph_objects as go

    class _MapFigureWidget(go.FigureWidget):
        """Ignore read-only viewport updates sent by Plotly when panning."""
        def _perform_plotly_relayout(self, relayout_data):
            if isinstance(relayout_data, dict):
                relayout_data = {k: v for k, v in relayout_data.items()
                                 if not (k.split(".", 1)[0].startswith("map")
                                         and k.endswith("._derived"))}
            return super()._perform_plotly_relayout(relayout_data)
    return _MapFigureWidget


class PairingEditor:
    """Map to inspect and change which reference station each station uses.

    Stations are coloured by type. Arrows go from each station to the
    reference whose net dust is subtracted from it (orange = changed by hand).

    Changing a link (two clicks):
      1. click a station  -> its link turns black, possible references get a
         yellow ring;
      2. click one of the yellow-ringed stations -> it becomes the reference.
    Click the selected station again to deselect. The list under the map, the
    buttons (undo, back to nearest, do not correct, exclude as reference,
    reset) and the methods ``set_reference``, ``reset``, ``exclude_reference``,
    ``reset_all`` do the same. With ``save_path`` every change is written to
    the overrides CSV at once.
    """

    def __init__(self, stations: pd.DataFrame, pairs: pd.DataFrame, cfg: PairingConfig,
                 overrides: pd.DataFrame | None = None, save_path: str | None = None,
                 candidate_max_km: float = 200.0, n_candidates: int = 15,
                 arrow_to: str = "reference"):
        self.stations = stations.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
        self.stations["type_category"] = [
            _type_category(c, t, a) for c, t, a in zip(
                self.stations["station_class"], self.stations.get("station_type"),
                self.stations.get("station_area"))]
        self.st = self.stations.set_index("Samplingpoint")
        self.base_pairs = pairs
        self.cfg = cfg
        self.overrides = (overrides.copy() if overrides is not None else empty_overrides())
        self.save_path = save_path
        self.candidate_max_km = candidate_max_km
        self.n_candidates = n_candidates
        self.arrow_to = arrow_to            # 'reference' or 'station'
        self.selected = None
        self._refcap = self.stations[self.stations["station_class"].isin(["RB", "UB", "OB"])]
        self._busy = False
        self._history = []
        self._message = ""
        self.widget = None

    # ----- state ----------------------------------------------------------
    @property
    def pairs(self) -> pd.DataFrame:
        return apply_overrides(self.base_pairs, self.stations, self.overrides)

    def _drop(self, sp, action):
        o = self.overrides
        self.overrides = o[~((o["Samplingpoint"] == sp) & (o["action"] == action))].reset_index(drop=True)

    def _push(self):
        self._history.append(self.overrides.copy())
        self._history = self._history[-50:]

    def set_reference(self, target: str, reference: str | None):
        """reference: a Samplingpoint, 'none' (do not correct) or None (back to nearest)."""
        self._push()
        self._drop(target, "assign")
        if reference is not None:
            self.overrides = pd.concat([self.overrides, pd.DataFrame(
                [[target, reference, "assign"]], columns=OVERRIDE_COLUMNS)], ignore_index=True)
        self._changed()

    def reset(self, target: str):
        self.set_reference(target, None)

    def exclude_reference(self, station: str, exclude: bool = True):
        self._push()
        self._drop(station, "exclude_reference")
        if exclude:
            self.overrides = pd.concat([self.overrides, pd.DataFrame(
                [[station, "", "exclude_reference"]], columns=OVERRIDE_COLUMNS)], ignore_index=True)
        self._changed()

    def reset_all(self):
        self._push()
        self.overrides = empty_overrides()
        self._changed()

    def undo(self):
        if self._history:
            self.overrides = self._history.pop()
            self._message = "Last change undone."
            self._changed()

    def save(self, path: str | None = None):
        path = path or self.save_path
        if path:
            self.overrides.to_csv(path, index=False)
        return path

    def _changed(self):
        if self.save_path:
            self.save()
        if self.widget is not None:
            self._refresh()

    def candidates(self, target: str, n: int | None = None) -> pd.DataFrame:
        """Possible references for ``target`` within ``candidate_max_km``, nearest first."""
        t = self.st.loc[target]
        r = self._refcap
        d = _haversine_km(np.array([t["Latitude"]]), np.array([t["Longitude"]]),
                          r["Latitude"].to_numpy(float), r["Longitude"].to_numpy(float))[0]
        c = r.assign(distance_km=d,
                     altitude_diff_m=np.abs(r["Altitude"].to_numpy(float) - float(t["Altitude"])))
        is_self = c["Samplingpoint"] == target
        # traffic/industrial sites do not self-calibrate (guidance, Option 2)
        if t["station_class"] == "HS":
            c = c[~is_self]
        c = c[(c["distance_km"] <= self.candidate_max_km)].sort_values("distance_km")
        if n:
            c = c.head(n)
        c = c.copy()
        lim_km = np.where(c["station_class"] == "RB", self.cfg.rb_max_distance_km, self.cfg.ub_max_distance_km)
        lim_alt = np.where(c["station_class"] == "RB", self.cfg.rb_max_altitude_diff_m,
                           self.cfg.ub_max_altitude_diff_m)
        c["within_limits"] = (c["distance_km"] <= lim_km) & ~(c["altitude_diff_m"] > lim_alt)
        c["option"] = [_manual_option(target, ref, self.st["station_class"]) for ref in c["Samplingpoint"]]
        return c

    # ----- widget ---------------------------------------------------------
    def _name(self, sp):
        n = self.st.at[sp, "station_name"] if "station_name" in self.st else ""
        return n if isinstance(n, str) and n else sp

    def _label(self, sp):
        r = self.st.loc[sp]
        return f"{self._name(sp)} ({r.get('station_type', '?')}/{r.get('station_area', '?')}) – {sp}"

    def show(self, zoom=None, height=650):
        import plotly.graph_objects as go
        from ipywidgets import HTML, Dropdown, Button, HBox, VBox, Layout, Checkbox
        try:
            from google.colab import output as colab_output
            colab_output.enable_custom_widget_manager()
        except ImportError:
            pass
        W = _map_figure_widget_class()
        lat, lon = self.stations["Latitude"], self.stations["Longitude"]
        if zoom is None:
            span = max(lat.max() - lat.min(), lon.max() - lon.min())
            zoom = 3.5 if span > 15 else 5 if span > 6 else 6.5
        S = go.Scattermap
        traces = [
            # 0-3 links and arrowheads
            S(lat=[], lon=[], mode="lines", line=dict(width=1.3, color=_LINK_AUTO),
              hoverinfo="skip", name="Link: nearest (automatic)", legendgroup="la"),
            S(lat=[], lon=[], mode="lines", fill="toself", fillcolor=_LINK_AUTO,
              line=dict(width=1, color=_LINK_AUTO), hoverinfo="skip", showlegend=False, legendgroup="la"),
            S(lat=[], lon=[], mode="lines", line=dict(width=2.2, color=_LINK_MANUAL),
              hoverinfo="skip", name="Link: changed by hand", legendgroup="lm"),
            S(lat=[], lon=[], mode="lines", fill="toself", fillcolor=_LINK_MANUAL,
              line=dict(width=1, color=_LINK_MANUAL), hoverinfo="skip", showlegend=False, legendgroup="lm"),
            # 4-5 selected links
            S(lat=[], lon=[], mode="lines", line=dict(width=4, color=_LINK_SELECTED),
              hoverinfo="skip", name="Selected station", legendgroup="ls"),
            S(lat=[], lon=[], mode="lines", fill="toself", fillcolor=_LINK_SELECTED,
              line=dict(width=1, color=_LINK_SELECTED), hoverinfo="skip", showlegend=False, legendgroup="ls"),
            # 6 candidate rings, 7 selection ring
            S(lat=[], lon=[], mode="markers", marker=dict(size=[], color=_CANDIDATE, opacity=0.85),
              hoverinfo="skip", name="Possible new reference (click)"),
            S(lat=[], lon=[], mode="markers", marker=dict(size=[], color=_LINK_SELECTED, opacity=0.9),
              hoverinfo="skip", showlegend=False),
        ]
        self._type_keys = list(_TYPE_STYLE)
        for key in self._type_keys:
            col, lab = _TYPE_STYLE[key]
            traces.append(S(lat=[], lon=[], mode="markers", name=lab,
                            marker=dict(size=[], color=col, opacity=1.0),
                            customdata=[], text=[], hovertemplate="%{text}<extra></extra>"))
        # last: black centre dot = not corrected
        traces.append(S(lat=[], lon=[], mode="markers", marker=dict(size=5, color="black"),
                        name="● not corrected (no reference)", hoverinfo="skip"))
        fig = go.Figure(traces)
        fig.update_layout(
            map=dict(style="carto-positron", zoom=zoom,
                     center=dict(lat=float(lat.median()), lon=float(lon.median()))),
            height=height, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                        bgcolor="rgba(255,255,255,0.85)", font=dict(size=11)),
            clickmode="event")
        self.fig = W(fig)
        self._first_type_trace = 8

        self.w_station = Dropdown(
            options=[("Click a station on the map, or choose it here", None)]
                    + sorted((self._label(sp), sp) for sp in self.stations["Samplingpoint"]),
            value=None, description="Station:", layout=Layout(width="780px"))
        self.w_ref = Dropdown(options=[], description="Reference:", layout=Layout(width="780px"),
                              disabled=True)
        self.w_click = Checkbox(value=True, indent=False,
                                description="Change a link by clicking: 1st click = station, 2nd click = new reference")
        self.w_undo = Button(description="Undo", icon="undo")
        self.w_reset = Button(description="Back to nearest", icon="refresh", disabled=True)
        self.w_none = Button(description="Do not correct", icon="times", disabled=True)
        self.w_excl = Button(description="Do not use as reference", icon="ban", disabled=True)
        self.w_reset_all = Button(description="Reset all changes", icon="trash", button_style="warning")
        self.w_info = HTML()
        self.w_status = HTML()

        self.w_station.observe(lambda ch: self._select(ch["new"]), names="value")
        self.w_ref.observe(self._on_ref, names="value")
        self.w_undo.on_click(lambda _: self.undo())
        self.w_reset.on_click(lambda _: self.selected and self._assign(self.selected, None))
        self.w_none.on_click(lambda _: self.selected and self._assign(self.selected, NO_REFERENCE))
        self.w_excl.on_click(self._on_exclude)
        self.w_reset_all.on_click(lambda _: self.reset_all())

        def on_click(trace, points, state):
            if points.point_inds:
                self.click(trace.customdata[points.point_inds[0]])
        for tr in self.fig.data[self._first_type_trace:self._first_type_trace + len(self._type_keys)]:
            tr.on_click(on_click)

        arrow_txt = ("station → reference whose net dust is subtracted from it"
                     if self.arrow_to == "reference" else "reference → station it represents")
        self.widget = VBox([
            HTML(f"<b>Station pairing.</b> Colour = station type. Arrow = {arrow_txt}. "
                 "Larger dots represent more stations. Hover a station for details."),
            self.fig, self.w_click, self.w_info, self.w_station, self.w_ref,
            HBox([self.w_undo, self.w_reset, self.w_none, self.w_excl, self.w_reset_all]),
            self.w_status])
        self._refresh()
        return self.widget

    # ----- interaction ----------------------------------------------------
    def click(self, sp):
        """Map click: select a station, or (2nd click) make ``sp`` its reference."""
        sel = self.selected
        if sel is not None and sp == sel:
            self.w_station.value = None
            return
        if (sel is not None and self.w_click.value
                and sp in set(self.candidates(sel)["Samplingpoint"])):
            self._assign(sel, sp)
            return
        self.w_station.value = sp

    def _assign(self, target, ref):
        self.set_reference(target, ref)
        if ref is None:
            self._message = f"<b>{self._name(target)}</b> is back to its nearest reference."
        elif ref == NO_REFERENCE:
            self._message = f"<b>{self._name(target)}</b> will not be corrected."
        else:
            self._message = f"<b>{self._name(target)}</b> now uses <b>{self._name(ref)}</b>."
        if self.widget is not None:
            self._refresh()

    def _on_ref(self, ch):
        if self._busy or self.selected is None or ch["new"] is None:
            return
        val = ch["new"]
        self._assign(self.selected, None if val == "__default__" else val)

    def _on_exclude(self, _):
        if self.selected is None:
            return
        o = self.overrides
        now = ((o["Samplingpoint"] == self.selected) & (o["action"] == "exclude_reference")).any()
        self.exclude_reference(self.selected, not now)

    def _select(self, sp):
        self.selected = sp
        self._message = ""
        self._refresh()

    def _refresh(self):
        asg = current_assignment(self.pairs)
        st = self.stations
        sp = st["Samplingpoint"].to_numpy()
        ref = asg["reference"].reindex(sp)
        opt = asg["option"].reindex(sp)
        man = asg["manual"].reindex(sp).fillna(False).astype(bool)
        dist = asg["distance_km"].reindex(sp)
        others = ref[(ref.notna()) & (ref.index != ref.values)]
        n_rep = others.value_counts()
        size = 9 + np.minimum(3 * np.sqrt(n_rep.reindex(sp).fillna(0).to_numpy()), 16)
        size_s = pd.Series(size, index=sp)
        pos = st.set_index("Samplingpoint")[["Latitude", "Longitude"]]

        def oriented(items):
            return [(a, b) if self.arrow_to == "reference" else (b, a) for a, b in items]

        auto_links = [(t, r) for t, r in others.items() if not man.get(t)]
        man_links = [(t, r) for t, r in others.items() if man.get(t)]
        la, aa = _arrow_geometry(oriented(auto_links), pos)
        lm, am = _arrow_geometry(oriented(man_links), pos)

        sel = self.selected
        sel_links, ring = [], []
        cand = pd.DataFrame()
        if sel is not None:
            r0 = ref.get(sel)
            if isinstance(r0, str) and r0 != sel:
                sel_links.append((sel, r0))
            rep = others[others == sel].index.tolist()
            sel_links += [(t, sel) for t in rep]
            ring = [sel]
            if self.w_click.value:
                cand = self.candidates(sel)
                cand = cand[cand["Samplingpoint"] != r0]
        ls, as_ = _arrow_geometry(oriented(sel_links), pos)

        hover = []
        for s_, r_, d_, m_ in zip(sp, ref, dist, man):
            n = int(n_rep.get(s_, 0))
            rtxt = "<i>not corrected</i>" if not isinstance(r_, str) else (
                "own record" if r_ == s_ else f"{self._name(r_)} ({d_:.1f} km)")
            hover.append(f"<b>{self._name(s_)}</b><br>{s_}<br>"
                         f"{self.st.at[s_, 'station_type']} / {self.st.at[s_, 'station_area']}<br>"
                         f"reference: {rtxt}{' (by hand)' if m_ else ''}<br>"
                         f"represents {n} other station(s)")
        hover = np.array(hover, dtype=object)
        tcat = st["type_category"].to_numpy()
        lat_a, lon_a = st["Latitude"].to_numpy(float), st["Longitude"].to_numpy(float)
        nc = ref.isna().to_numpy()

        D = self.fig.data
        with self.fig.batch_update():
            D[0].lat, D[0].lon = la
            D[1].lat, D[1].lon = aa
            D[2].lat, D[2].lon = lm
            D[3].lat, D[3].lon = am
            D[4].lat, D[4].lon = ls
            D[5].lat, D[5].lon = as_
            D[6].lat = cand["Latitude"].tolist() if len(cand) else []
            D[6].lon = cand["Longitude"].tolist() if len(cand) else []
            D[6].marker.size = (size_s.reindex(cand["Samplingpoint"]).to_numpy() + 9).tolist() if len(cand) else []
            D[7].lat = [pos.at[h, "Latitude"] for h in ring]
            D[7].lon = [pos.at[h, "Longitude"] for h in ring]
            D[7].marker.size = [float(size_s[h]) + 12 for h in ring]
            for k, key in enumerate(self._type_keys, start=self._first_type_trace):
                m = tcat == key
                tr = D[k]
                tr.lat = lat_a[m].tolist()
                tr.lon = lon_a[m].tolist()
                tr.customdata = sp[m].tolist()
                tr.text = hover[m].tolist()
                tr.marker.size = size[m].tolist()
                tr.visible = True if m.any() else "legendonly"
            D[-1].lat = lat_a[nc].tolist()
            D[-1].lon = lon_a[nc].tolist()

        n_man = int((self.overrides["action"] == "assign").sum())
        n_exc = int((self.overrides["action"] == "exclude_reference").sum())
        saved = f" Saved to <code>{self.save_path}</code>." if self.save_path else ""
        self.w_status.value = (f"<span style='color:#555'>{n_man} station(s) changed by hand, "
                               f"{n_exc} reference(s) excluded.{saved}</span>")
        self.w_undo.disabled = not self._history

        if sel is None:
            self.w_info.value = (f"<div style='padding:4px 0'>{self._message}<br>"
                                 "Click a station to see its link and change it.</div>")
            self.w_ref.disabled = self.w_reset.disabled = self.w_none.disabled = self.w_excl.disabled = True
            self._busy = True
            self.w_ref.options = []
            self._busy = False
            return
        cands = self.candidates(sel, self.n_candidates)
        opts = [("Default: nearest suitable reference", "__default__"),
                ("Do not correct this station", NO_REFERENCE)]
        for _, c in cands.iterrows():
            who = "own record" if c["Samplingpoint"] == sel else self._name(c["Samplingpoint"])
            warn = "  ⚠ outside distance/altitude limits" if not c["within_limits"] else ""
            opts.append((f"{who} | {c['station_type']}/{c['station_area']} | {c['distance_km']:.1f} km"
                         f" | Δalt {c['altitude_diff_m']:.0f} m | {c['option']}{warn}",
                         c["Samplingpoint"]))
        o = self.overrides
        cur = o[(o["Samplingpoint"] == sel) & (o["action"] == "assign")]
        value = cur["reference"].iloc[-1] if len(cur) else "__default__"
        if value not in [v for _, v in opts]:
            opts.append((f"{self._name(value)} (set by hand)", value))
        self._busy = True
        self.w_ref.options = opts
        self.w_ref.value = value
        self._busy = False
        self.w_ref.disabled = self.w_reset.disabled = self.w_none.disabled = False
        excluded = ((o["Samplingpoint"] == sel) & (o["action"] == "exclude_reference")).any()
        self.w_excl.disabled = self.st.at[sel, "station_class"] == "HS"
        self.w_excl.description = "Use again as reference" if excluded else "Do not use as reference"

        r0 = ref.get(sel)
        rep = others[others == sel].index.tolist()
        rtxt = ("<b>not corrected</b>" if not isinstance(r0, str) else
                "its <b>own record</b>" if r0 == sel else
                f"<b>{self._name(r0)}</b> ({dist.get(sel):.1f} km)")
        how = "set by hand" if man.get(sel) else "nearest suitable"
        lst = ", ".join(self._name(t) for t in rep[:15]) + (" …" if len(rep) > 15 else "")
        hint = ("<br><span style='color:#8a6d00'>➜ Click a station with a <b>yellow ring</b> "
                "to make it the new reference.</span>") if self.w_click.value else ""
        msg = f"{self._message}<br>" if self._message else ""
        self.w_info.value = (f"<div style='padding:4px 0'>{msg}<b>{self._label(sel)}</b><br>"
                             f"Uses {rtxt} – {opt.get(sel, 'none')}, {how}."
                             f"{' <i>Excluded as reference.</i>' if excluded else ''}<br>"
                             f"Represents <b>{len(rep)}</b> other station(s){': ' + lst if rep else ''}"
                             f"{hint}</div>")
