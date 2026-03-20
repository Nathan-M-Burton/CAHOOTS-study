"""Difference-in-Differences functions for the SDR/PDR analysis."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2


def ensure_complete_data(df_agg, grouping_cols, year_range_tuple, specific_type=None):
    """Ensure all year x hour combinations are present, filling gaps with zero."""
    year_col = "year"
    hr_col = "hr"
    volume_col = "Call Volume"
    incident_col = "InitialIncidentTypeDescription"

    years_in_range = list(range(year_range_tuple[0], year_range_tuple[1] + 1))
    hours_in_day = list(range(24))

    base_df = pd.DataFrame(
        [(y, hour) for y in years_in_range for hour in hours_in_day],
        columns=[year_col, hr_col],
    )
    grouping_cols_corrected = [year_col if c == "Year" else c for c in grouping_cols]

    if df_agg is None:
        df_agg = pd.DataFrame(columns=grouping_cols_corrected + [hr_col, volume_col])

    if "Year" in df_agg.columns and year_col not in df_agg.columns:
        df_agg.rename(columns={"Year": year_col}, inplace=True)

    current_merge_on_cols = [year_col, hr_col]
    if incident_col in grouping_cols_corrected and specific_type:
        base_df[incident_col] = specific_type
        current_merge_on_cols.append(incident_col)

    for col in current_merge_on_cols + [volume_col]:
        if col not in df_agg.columns:
            if col == volume_col:
                df_agg[col] = 0
            elif col == incident_col and specific_type:
                df_agg[col] = specific_type
            elif col in (year_col, hr_col):
                df_agg[col] = np.nan

    base_df[year_col] = base_df[year_col].astype(int)
    if not df_agg.empty:
        if year_col in df_agg.columns:
            df_agg[year_col] = pd.to_numeric(df_agg[year_col], errors="coerce").fillna(0).astype(int)
        if incident_col in df_agg.columns and specific_type:
            df_agg[incident_col] = df_agg[incident_col].astype(str)
            if incident_col in base_df.columns:
                base_df[incident_col] = base_df[incident_col].astype(str)

    try:
        if not df_agg.empty and all(c in df_agg.columns for c in current_merge_on_cols):
            merged = pd.merge(base_df, df_agg, on=current_merge_on_cols, how="left")
        else:
            merged = base_df.copy()
            merged[volume_col] = 0
    except Exception:
        merged = base_df.copy()
        merged[volume_col] = 0

    merged[volume_col] = merged[volume_col].fillna(0).astype(int)
    merged[year_col] = merged[year_col].astype(int)

    if incident_col in grouping_cols_corrected and specific_type:
        merged[incident_col] = merged[incident_col].fillna(specific_type).astype(str)

    final_cols_expected = grouping_cols_corrected + [hr_col, volume_col]
    final_cols_present = [c for c in final_cols_expected if c in merged.columns]
    return merged[final_cols_present]


def _estimate_nb_alpha(formula, data, offset):
    """Estimate NB dispersion alpha via Poisson auxiliary regression."""
    try:
        pois_res = smf.glm(formula, data=data, family=sm.families.Poisson(),
                           offset=offset).fit()
        mu = pois_res.fittedvalues
        y = data["count"]
        if np.all(np.isfinite(mu)) and np.all(mu > 0):
            denom = (mu ** 2).sum()
            if denom > 1e-9:
                return max(((y - mu) ** 2 - mu).sum() / denom, 1e-9)
    except Exception:
        pass
    return np.nan


def _fit_nb_model(formula, data, offset, alpha, cluster_se):
    """Fit a NB GLM with optional clustered standard errors on monthnum."""
    nb_fam = sm.families.NegativeBinomial(alpha=alpha)
    cov_type, cov_kwds = "HC1", None
    if cluster_se and "monthnum" in data.columns and data["monthnum"].nunique() >= 2:
        cov_type = "cluster"
        cov_kwds = {"groups": data["monthnum"]}
    return smf.glm(formula, data=data, family=nb_fam,
                   offset=offset).fit(cov_type=cov_type, cov_kwds=cov_kwds,
                                      maxiter=100)


def _run_parallel_trends_test(cdf, treatment_indicator, cluster_se):
    """Pre-treatment parallel-trends test. Returns (p_value, slope)."""
    pre_full = cdf[cdf["full_post"] == 0].copy()

    if (pre_full.empty
            or pre_full[treatment_indicator].nunique() != 2
            or pre_full.groupby(treatment_indicator)["monthnum"].nunique().min() < 2):
        return np.nan, np.nan

    pt_rhs = [treatment_indicator, "monthnum", f"{treatment_indicator}:monthnum"]
    if "log_monthly_volume" in pre_full.columns and pre_full["log_monthly_volume"].nunique() > 1:
        pt_rhs.append("log_monthly_volume")
    pt_rhs = sorted(set(pt_rhs))
    pt_formula = "count ~ " + " + ".join(pt_rhs)
    pt_interaction = f"{treatment_indicator}:monthnum"

    try:
        alpha_pt = _estimate_nb_alpha(pt_formula, pre_full, pre_full["offset"])
        if pd.isna(alpha_pt):
            return np.nan, np.nan

        nb_res = _fit_nb_model(pt_formula, pre_full, pre_full["offset"],
                               alpha_pt, cluster_se)
        if nb_res.converged and pt_interaction in nb_res.pvalues:
            return nb_res.pvalues[pt_interaction], nb_res.params[pt_interaction]
    except Exception:
        pass
    return np.nan, np.nan


def _make_skipped_result(ctype, method, raw_counts, pt_pvalue, pt_slope):
    """Build a result dict for a skipped call type."""
    return {
        "call_type": ctype, "analysis_method": method,
        "did_coefficient": np.nan, "p_value": np.nan, "std_error": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "effect_size": np.nan, "theta_low": np.nan, "theta_high": np.nan,
        **raw_counts,
        "alpha_nb": np.nan, "converged": False,
        "pt_pvalue": pt_pvalue, "pt_slope": pt_slope,
    }


def _compute_counterfactual_proportions(results_df):
    """Add proportion-added / proportion-prevented columns to results."""
    out_cols = [
        "prop_added", "prop_added_low", "prop_added_high",
        "prop_prevented", "prop_prev_low", "prop_prev_high",
        "calls_added_or_prevented", "calls_added_low", "calls_added_high",
    ]
    for col in out_cols:
        results_df[col] = 0.0

    req = ["effect_size", "theta_low", "theta_high", "treatment_post_raw", "significant"]
    if results_df.empty or not all(c in results_df.columns for c in req):
        return

    calc = results_df.dropna(subset=req).copy()
    if calc.empty:
        return

    sig_mask = calc["significant"].astype(bool)
    theta = calc["effect_size"]
    theta_L = calc["theta_low"]
    theta_H = calc["theta_high"]
    Y = calc["treatment_post_raw"]

    eps = 1e-9
    denom = np.maximum(eps, 1 + theta)
    denom_L = np.maximum(eps, 1 + theta_L)
    denom_H = np.maximum(eps, 1 + theta_H)

    delta = np.zeros_like(theta)
    delta_low = np.zeros_like(theta)
    delta_high = np.zeros_like(theta)

    if sig_mask.any():
        delta[sig_mask] = Y.loc[sig_mask] * (theta.loc[sig_mask] / denom.loc[sig_mask])
        ci_lo = Y.loc[sig_mask] * (theta_L.loc[sig_mask] / denom_L.loc[sig_mask])
        ci_hi = Y.loc[sig_mask] * (theta_H.loc[sig_mask] / denom_H.loc[sig_mask])
        delta_low[sig_mask] = np.minimum(ci_lo, ci_hi)
        delta_high[sig_mask] = np.maximum(ci_lo, ci_hi)

    results_df.loc[calc.index, "calls_added_or_prevented"] = delta
    results_df.loc[calc.index, "calls_added_low"] = delta_low
    results_df.loc[calc.index, "calls_added_high"] = delta_high

    inc = sig_mask & (theta > 0)
    dec = sig_mask & (theta < 0)

    if inc.any():
        prop_pt = theta.loc[inc] / denom.loc[inc]
        prop_lo = np.where(theta_L.loc[inc] > 0, theta_L.loc[inc] / denom_L.loc[inc], 0.0)
        prop_hi = theta_H.loc[inc] / denom_H.loc[inc]
        results_df.loc[calc.index[inc], ["prop_added", "prop_added_low", "prop_added_high"]] = np.vstack(
            [prop_pt, np.minimum(prop_lo, prop_hi), np.maximum(prop_lo, prop_hi)]
        ).T

    if dec.any():
        prop_pt = np.abs(theta.loc[dec])
        prop_lo_c = np.abs(theta_L.loc[dec])
        prop_hi_c = np.abs(theta_H.loc[dec])
        results_df.loc[calc.index[dec], ["prop_prevented", "prop_prev_low", "prop_prev_high"]] = np.vstack(
            [prop_pt, np.minimum(prop_lo_c, prop_hi_c), np.maximum(prop_lo_c, prop_hi_c)]
        ).T


def analyze_did_offset_nb_canonical_monthly(
    data,
    treatment_indicator,
    cutoff_year=2017,
    cutoff_month=1,
    partial_year=2016,
    partial_month=11,
    p=0.05,
    min_calls=0,
    min_total_calls=1000,
    return_monthly_data=True,
    cluster_se=False,
    return_models=False,
):
    """
    Difference-in-Differences analysis using a Negative Binomial model.

    The outcome variable is the combined EPD + CAHOOTS call volume,
    aggregated at the monthly level.  Includes a parallel-trends test
    for each call type.
    """
    df = data.copy()
    if "cahoots_handled" not in df.columns:
        raise ValueError("Input data must contain a 'cahoots_handled' column.")

    df["monthnum"] = 12 * df["year"] + df["month"]

    cut_full = 12 * cutoff_year + cutoff_month
    cut_part = 12 * partial_year + partial_month

    df["full_post"] = (df["monthnum"] >= cut_full).astype(int)
    df["partial_post"] = ((df["monthnum"] >= cut_part) & (df["monthnum"] < cut_full)).astype(int)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

    grp_cols = [
        "InitialIncidentTypeDescription", treatment_indicator,
        "year", "month", "monthnum",
        "partial_post", "full_post", "date",
    ]
    monthly = (
        df.groupby(grp_cols, observed=True)
        .agg(count=("InitialIncidentTypeDescription", "size"),
             cahoots_count=("cahoots_handled", "sum"))
        .reset_index()
    )

    vol_cols = [
        treatment_indicator, "year", "month", "monthnum",
        "partial_post", "full_post", "date",
    ]
    group_vol = (
        df.groupby(vol_cols, observed=True).size().reset_index(name="group_volume")
    )
    monthly = monthly.merge(group_vol, on=vol_cols, how="left")

    monthly["days_in_month"] = monthly["date"].dt.days_in_month
    monthly["hours_per_day"] = np.where(monthly[treatment_indicator] == 1, 4, 20)
    monthly["exposure_hours"] = monthly["hours_per_day"] * monthly["days_in_month"]
    monthly["offset"] = np.log(monthly["exposure_hours"].clip(lower=1))
    monthly["log_monthly_volume"] = np.log(monthly["group_volume"].clip(lower=1))
    monthly["call_rate"] = monthly["count"] / monthly["exposure_hours"]

    # Model per call type
    results, monthly_out = [], []
    fitted_models = {}
    all_call_types = monthly["InitialIncidentTypeDescription"].unique()
    print(f"Analyzing {len(all_call_types)} call types...")
    processed_count = 0

    for ctype, cdf in monthly.groupby("InitialIncidentTypeDescription", observed=True):
        tmask = cdf[treatment_indicator] == 1
        cmask = ~tmask
        if not tmask.any() or not cmask.any():
            continue

        treat_pre_series = cdf.loc[tmask & (cdf["full_post"] == 0) & (cdf["partial_post"] == 0), "count"]
        treat_post_total_system_series = cdf.loc[tmask & (cdf["full_post"] == 1), "count"]
        ctrl_pre_series = cdf.loc[cmask & (cdf["full_post"] == 0) & (cdf["partial_post"] == 0), "count"]
        ctrl_post_total_system_series = cdf.loc[cmask & (cdf["full_post"] == 1), "count"]
        cahoots_treat_post_series = cdf.loc[tmask & (cdf["full_post"] == 1), "cahoots_count"]
        cahoots_treat_post_raw = cahoots_treat_post_series.sum() if not cahoots_treat_post_series.empty else 0

        treat_pre_total_system = treat_pre_series.sum() if not treat_pre_series.empty else 0
        treat_post_total_system = treat_post_total_system_series.sum() if not treat_post_total_system_series.empty else 0
        ctrl_pre_total_system = ctrl_pre_series.sum() if not ctrl_pre_series.empty else 0
        ctrl_post_total_system = ctrl_post_total_system_series.sum() if not ctrl_post_total_system_series.empty else 0

        if (treat_pre_total_system + treat_post_total_system) < min_calls:
            continue
        if (treat_pre_total_system + treat_post_total_system + ctrl_pre_total_system + ctrl_post_total_system) < min_total_calls:
            continue

        raw_counts = {
            "treatment_post_raw": treat_post_total_system,
            "treatment_pre_raw": treat_pre_total_system,
            "control_post_raw": ctrl_post_total_system,
            "control_pre_raw": ctrl_pre_total_system,
            "cahoots_treat_post_raw": cahoots_treat_post_raw,
        }

        if return_monthly_data:
            tmp = cdf.copy()
            tmp["call_type"] = ctype
            tmp["group"] = np.where(tmp[treatment_indicator] == 1, "Treatment", "Control")
            monthly_out.append(tmp)

        # Parallel-trends test
        pt_pvalue, pt_slope = _run_parallel_trends_test(cdf, treatment_indicator, cluster_se)

        # Main DiD model
        main_rhs = sorted({
            treatment_indicator, "partial_post", "full_post",
            f"{treatment_indicator}:partial_post",
            f"{treatment_indicator}:full_post",
            "log_monthly_volume",
        })
        main_formula = "count ~ " + " + ".join(main_rhs)
        main_inter = f"{treatment_indicator}:full_post"

        if cdf["count"].nunique() <= 1 or cdf[treatment_indicator].nunique() <= 1 or cdf["full_post"].nunique() <= 1:
            results.append(_make_skipped_result(
                ctype, "Skipped - Insufficient Variation", raw_counts, pt_pvalue, pt_slope))
            processed_count += 1
            continue

        # Estimate alpha via Poisson
        alpha_main = _estimate_nb_alpha(main_formula, cdf, cdf["offset"])

        if pd.isna(alpha_main):
            results.append(_make_skipped_result(
                ctype, "Skipped - Alpha Estimation Error", raw_counts, pt_pvalue, pt_slope))
            processed_count += 1
            continue

        nb_res_main = None
        cov_kwds = None
        try:
            nb_res_main = _fit_nb_model(main_formula, cdf, cdf["offset"],
                                        alpha_main, cluster_se)
            if nb_res_main.converged and return_models:
                fitted_models[ctype] = nb_res_main
            # Track whether clustering was applied for the method label
            if cluster_se and "monthnum" in cdf.columns and cdf["monthnum"].nunique() >= 2:
                cov_kwds = True
        except Exception:
            pass

        coef, p_val_main, std_err, ci_low, ci_up = [np.nan] * 5
        theta_hat, theta_lo, theta_hi = [np.nan] * 3
        converged_status = False

        if nb_res_main is not None and nb_res_main.converged:
            converged_status = True
            if main_inter in nb_res_main.params.index:
                try:
                    coef = nb_res_main.params[main_inter]
                    p_val_main = nb_res_main.pvalues[main_inter]
                    std_err = nb_res_main.bse[main_inter]
                    ci_bounds = nb_res_main.conf_int(alpha=p).loc[main_inter]
                    ci_low, ci_up = ci_bounds[0], ci_bounds[1]
                    if np.all(np.isfinite([coef, ci_low, ci_up])):
                        theta_hat = np.exp(coef) - 1
                        theta_lo = np.exp(ci_low) - 1
                        theta_hi = np.exp(ci_up) - 1
                except Exception:
                    pass

        results.append({
            "call_type": ctype,
            "analysis_method": ("Monthly NB DiD (step-wise alpha, clustered)"
                                if cluster_se and cov_kwds is not None
                                else "Monthly NB DiD (step-wise alpha)"),
            "did_coefficient": coef, "p_value": p_val_main, "std_error": std_err,
            "ci_lower": ci_low, "ci_upper": ci_up,
            "effect_size": theta_hat, "theta_low": theta_lo, "theta_high": theta_hi,
            **raw_counts,
            "alpha_nb": alpha_main if pd.notna(alpha_main) else np.nan,
            "converged": converged_status,
            "pt_pvalue": pt_pvalue, "pt_slope": pt_slope,
        })
        processed_count += 1

    print(f"Analyzed {processed_count} / {len(all_call_types)} call types")

    results_df = pd.DataFrame(results)
    if results_df.empty:
        empty = (results_df, pd.DataFrame())
        if return_models:
            empty += ({},)
        return empty if return_monthly_data else (empty[0] if not return_models else (empty[0], empty[2]))

    # Multiple-testing correction
    if "p_value" in results_df.columns:
        valid = results_df["p_value"].dropna()
        if not valid.empty:
            results_df.loc[valid.index, "p_value_corrected"] = multipletests(valid, method="holm")[1]
        else:
            results_df["p_value_corrected"] = np.nan
    else:
        results_df["p_value_corrected"] = np.nan

    if "effect_size" not in results_df.columns:
        results_df["effect_size"] = np.nan

    sig = (
        (results_df["p_value_corrected"] < p)
        & (results_df["effect_size"].abs() >= 0.05)
        & results_df["effect_size"].notna()
    )
    results_df["significant"] = sig.fillna(False)

    results_df["impact_classification"] = np.select(
        [~results_df["significant"],
         results_df["significant"] & (results_df["effect_size"] > 0),
         results_df["significant"] & (results_df["effect_size"] < 0)],
        ["No significant change", "Significant increase", "Significant decrease"],
        default="Insufficient data",
    )

    _compute_counterfactual_proportions(results_df)

    # Sort by |effect_size|
    if "effect_size" in results_df.columns and results_df["effect_size"].notna().any():
        results_df["_abs"] = results_df["effect_size"].abs()
        results_df = results_df.sort_values("_abs", ascending=False, na_position="last").drop(columns=["_abs"])

    monthly_df_out = pd.DataFrame()
    if return_monthly_data:
        if monthly_out:
            monthly_df_out = pd.concat(monthly_out, ignore_index=True)
            if "significant" in results_df.columns:
                sig_types = set(results_df.loc[results_df["significant"].fillna(False), "call_type"])
                monthly_df_out["is_significant_did"] = monthly_df_out["call_type"].isin(sig_types)
            else:
                monthly_df_out["is_significant_did"] = False

    if return_monthly_data and return_models:
        return results_df, monthly_df_out, fitted_models
    elif return_monthly_data:
        return results_df, monthly_df_out
    elif return_models:
        return results_df, fitted_models
    else:
        return results_df


def test_within_treatment_heterogeneity(
    data,
    cutoff_year=2017,
    cutoff_month=1,
    partial_year=2016,
    partial_month=11,
    split_hour=5,
    call_types_to_test=None,
    results_df_main=None,
    cluster_se=True,
    p_threshold=0.05,
):
    """
    Test for effect heterogeneity within the treatment window.

    Estimates separate DiD effects for early (3-split_hour) vs. late
    (split_hour-7) sub-windows and tests H0: beta_early = beta_late
    using a Wald test.
    """
    df = data.copy()
    df["monthnum"] = 12 * df["year"] + df["month"]
    cut_full = 12 * cutoff_year + cutoff_month
    cut_part = 12 * partial_year + partial_month

    df["full_post"] = (df["monthnum"] >= cut_full).astype(int)
    df["partial_post"] = ((df["monthnum"] >= cut_part) & (df["monthnum"] < cut_full)).astype(int)
    df["treat_early"] = ((df["hour"] >= 3) & (df["hour"] < split_hour)).astype(int)
    df["treat_late"] = ((df["hour"] >= split_hour) & (df["hour"] < 7)).astype(int)

    if call_types_to_test is None:
        if results_df_main is not None and "significant" in results_df_main.columns:
            call_types_to_test = results_df_main.loc[
                results_df_main["significant"], "call_type"
            ].tolist()
        else:
            call_types_to_test = df["InitialIncidentTypeDescription"].unique().tolist()

    print(f"Testing within-treatment heterogeneity for {len(call_types_to_test)} call types")
    print(f"Early window: 3am-{split_hour}am | Late window: {split_hour}am-7am")

    grp_cols = [
        "InitialIncidentTypeDescription", "treat_early", "treat_late",
        "year", "month", "monthnum", "partial_post", "full_post",
    ]
    monthly = df.groupby(grp_cols, observed=True).size().reset_index(name="count")
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    monthly["days_in_month"] = monthly["date"].dt.days_in_month

    def calc_hours(row):
        if row["treat_early"] == 1:
            return split_hour - 3
        elif row["treat_late"] == 1:
            return 7 - split_hour
        return 20

    monthly["hours_per_day"] = monthly.apply(calc_hours, axis=1)
    monthly["exposure_hours"] = monthly["hours_per_day"] * monthly["days_in_month"]
    monthly["offset"] = np.log(monthly["exposure_hours"].clip(lower=1))

    vol_cols = ["treat_early", "treat_late", "year", "month", "monthnum"]
    group_vol = df.groupby(vol_cols, observed=True).size().reset_index(name="group_volume")
    monthly = monthly.merge(group_vol, on=vol_cols, how="left")
    monthly["log_monthly_volume"] = np.log(monthly["group_volume"].clip(lower=1))

    results = []

    for ctype in call_types_to_test:
        cdf = monthly[monthly["InitialIncidentTypeDescription"] == ctype].copy()

        if cdf.empty or cdf["count"].sum() < 100:
            results.append({
                "call_type": ctype,
                "beta_early": np.nan, "se_early": np.nan,
                "beta_late": np.nan, "se_late": np.nan,
                "wald_statistic": np.nan, "heterogeneity_pvalue": np.nan,
                "interpretation": "Insufficient data", "n_obs": len(cdf),
            })
            continue

        has_early = cdf["treat_early"].sum() > 0
        has_late = cdf["treat_late"].sum() > 0
        has_control = ((cdf["treat_early"] == 0) & (cdf["treat_late"] == 0)).sum() > 0
        has_post_var = cdf["full_post"].nunique() == 2

        if not (has_early and has_late and has_control and has_post_var):
            results.append({
                "call_type": ctype,
                "beta_early": np.nan, "se_early": np.nan,
                "beta_late": np.nan, "se_late": np.nan,
                "wald_statistic": np.nan, "heterogeneity_pvalue": np.nan,
                "interpretation": "Insufficient variation in groups", "n_obs": len(cdf),
            })
            continue

        formula = (
            "count ~ "
            "treat_early + treat_late + partial_post + full_post + "
            "treat_early:partial_post + treat_early:full_post + "
            "treat_late:partial_post + treat_late:full_post + "
            "log_monthly_volume"
        )

        try:
            alpha = _estimate_nb_alpha(formula, cdf, cdf["offset"])
            if pd.isna(alpha):
                alpha = 0.1

            nb_fam = sm.families.NegativeBinomial(alpha=alpha)
            cov_type, cov_kwds = "HC1", None
            if cluster_se and "monthnum" in cdf.columns and cdf["monthnum"].nunique() >= 2:
                cov_type = "cluster"
                cov_kwds = {"groups": cdf["monthnum"]}

            nb_res = smf.glm(formula, data=cdf, family=nb_fam, offset=cdf["offset"]).fit(
                cov_type=cov_type, cov_kwds=cov_kwds, maxiter=100
            )

            if not nb_res.converged:
                results.append({
                    "call_type": ctype,
                    "beta_early": np.nan, "se_early": np.nan,
                    "beta_late": np.nan, "se_late": np.nan,
                    "wald_statistic": np.nan, "heterogeneity_pvalue": np.nan,
                    "interpretation": "Model did not converge", "n_obs": len(cdf),
                })
                continue

            early_term, late_term = "treat_early:full_post", "treat_late:full_post"
            if early_term not in nb_res.params.index or late_term not in nb_res.params.index:
                results.append({
                    "call_type": ctype,
                    "beta_early": np.nan, "se_early": np.nan,
                    "beta_late": np.nan, "se_late": np.nan,
                    "wald_statistic": np.nan, "heterogeneity_pvalue": np.nan,
                    "interpretation": "Required terms missing", "n_obs": len(cdf),
                })
                continue

            beta_early = nb_res.params[early_term]
            beta_late = nb_res.params[late_term]
            se_early = nb_res.bse[early_term]
            se_late = nb_res.bse[late_term]

            cov_matrix = nb_res.cov_params()
            ei = list(nb_res.params.index).index(early_term)
            li = list(nb_res.params.index).index(late_term)
            var_diff = cov_matrix.iloc[ei, ei] + cov_matrix.iloc[li, li] - 2 * cov_matrix.iloc[ei, li]
            se_diff = np.sqrt(var_diff) if var_diff > 0 else np.nan

            if pd.notna(se_diff) and se_diff > 0:
                wald_stat = ((beta_early - beta_late) / se_diff) ** 2
                wald_pval = 1 - chi2.cdf(wald_stat, df=1)
            else:
                wald_stat, wald_pval = np.nan, np.nan

            if pd.isna(wald_pval):
                interp = "Could not compute test"
            elif wald_pval >= p_threshold:
                interp = "No significant heterogeneity (supports stability)"
            elif beta_early > beta_late:
                interp = "Significant heterogeneity: larger effect in early window"
            else:
                interp = "Significant heterogeneity: larger effect in late window"

            results.append({
                "call_type": ctype,
                "beta_early": beta_early, "se_early": se_early,
                "effect_size_early": np.exp(beta_early) - 1,
                "beta_late": beta_late, "se_late": se_late,
                "effect_size_late": np.exp(beta_late) - 1,
                "beta_difference": beta_early - beta_late,
                "se_difference": se_diff,
                "wald_statistic": wald_stat, "heterogeneity_pvalue": wald_pval,
                "interpretation": interp, "n_obs": len(cdf), "converged": True,
            })

        except Exception as e:
            results.append({
                "call_type": ctype,
                "beta_early": np.nan, "se_early": np.nan,
                "beta_late": np.nan, "se_late": np.nan,
                "wald_statistic": np.nan, "heterogeneity_pvalue": np.nan,
                "interpretation": f"Model error: {str(e)[:50]}", "n_obs": len(cdf),
            })

    results_df = pd.DataFrame(results)

    valid = results_df.dropna(subset=["heterogeneity_pvalue"])
    if not valid.empty:
        n_stable = (valid["heterogeneity_pvalue"] >= p_threshold).sum()
        n_het = (valid["heterogeneity_pvalue"] < p_threshold).sum()
        print(f"\nSummary:")
        print(f"  Stable effects (p >= {p_threshold}): {n_stable}")
        print(f"  Heterogeneous effects (p < {p_threshold}): {n_het}")
        print(f"  Estimation issues: {len(results_df) - len(valid)}")

    return results_df
