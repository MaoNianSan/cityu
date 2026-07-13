"""Result rows, aggregation, relative widths, and endpoint stability."""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import MAIN_CONFIDENCE_LEVEL

RESULT_COLUMNS=[
    "experiment","replicate","target","parameter","learner","learner_label","learner_model",
    "method","method_label","confidence_level","estimate","truth","ci_low","ci_high","ci_width","covered",
    "quality_metric","quality_value","selected_lambda","lambda_source","backend","package_version","status","warning",
    "tuned_params","validation_score","n_total","n_labelled","n_unlabelled","n_train","n_inference",
]
WARNING_COLUMNS=["experiment","replicate","learner","method","parameter","confidence_level","warning","traceback"]


def aggregate_results(rows):
    df=pd.DataFrame(rows,columns=RESULT_COLUMNS)
    ok=df[(df.status=="ok") & df.ci_width.notna()].copy()
    classic_rep=(ok[ok.method=="classic"][["experiment","replicate","parameter","confidence_level","ci_width"]]
                 .rename(columns={"ci_width":"classic_width_rep"}))
    ok=ok.merge(classic_rep,on=["experiment","replicate","parameter","confidence_level"],how="left")
    ok["relative_width_rep"]=ok["ci_width"]/ok["classic_width_rep"]
    df=df.merge(ok[["experiment","replicate","parameter","learner","method","confidence_level","relative_width_rep"]],
                on=["experiment","replicate","parameter","learner","method","confidence_level"],how="left")
    group=["experiment","target","parameter","learner","learner_label","learner_model","quality_metric","method","method_label","confidence_level"]
    summary=(ok.groupby(group,dropna=False,as_index=False)
        .agg(coverage=("covered","mean"),
             avg_width=("ci_width","mean"),sd_ci_low=("ci_low","std"),sd_ci_high=("ci_high","std"),
             quality_value_mean=("quality_value","mean"),quality_value_sd=("quality_value","std"),
             relative_width_rep_mean=("relative_width_rep","mean"),relative_width_rep_sd=("relative_width_rep","std"),
             relative_width_rep_median=("relative_width_rep","median"),
             relative_width_rep_q25=("relative_width_rep",lambda x:x.quantile(.25)),
             relative_width_rep_q75=("relative_width_rep",lambda x:x.quantile(.75)),
             median_lambda=("selected_lambda","median"),n_success=("status","size")))
    summary["coverage_error"]=summary["coverage"]-summary["confidence_level"]
    total=(df.groupby(group,dropna=False).size().rename("n_total_rows").reset_index())
    summary=summary.merge(total,on=group,how="left")
    summary["n_failed"]=summary["n_total_rows"]-summary["n_success"]
    summary=summary.drop(columns="n_total_rows")
    classic=(summary[summary.method=="classic"][["experiment","parameter","confidence_level","avg_width"]]
             .rename(columns={"avg_width":"classic_avg_width"}))
    summary=summary.merge(classic,on=["experiment","parameter","confidence_level"],how="left")
    summary["relative_width"]=summary["avg_width"]/summary["classic_avg_width"]
    main=summary[np.isclose(summary.confidence_level,MAIN_CONFIDENCE_LEVEL)].copy()
    return df,summary,main

def summarize_ppiv2_tuning_effect(replicate_results, diagnostics, tolerance=1e-10):
    """Summarize selected V2 lambda and paired 95% V2/PPI interval changes."""
    grid = diagnostics[diagnostics["diagnostic"].eq("lambda_grid")].copy()
    keys = ["experiment", "replicate", "learner"]
    explicit = (replicate_results.loc[replicate_results.method.eq("ppi_plus_plus_v2"), keys+["selected_lambda"]]
                .dropna(subset=["selected_lambda"]).drop_duplicates(keys))
    if len(explicit): selected = explicit.rename(columns={"selected_lambda":"lambda_selected"})
    elif len(grid):
        selected = grid.loc[grid.groupby(keys)["value"].idxmin(), keys+["lambda"]].rename(columns={"lambda":"lambda_selected"})
    else: selected = pd.DataFrame(columns=keys+["lambda_selected"])
    lam = selected.groupby(["experiment","learner"]).lambda_selected.agg(
        lambda_median="median", lambda_q25=lambda x:x.quantile(.25), lambda_q75=lambda x:x.quantile(.75),
        lambda_equal_1_rate=lambda x:np.mean(np.isclose(x,1,atol=tolerance)),
        lambda_below_1_rate=lambda x:np.mean(x < 1-tolerance)).reset_index()
    ok=replicate_results[(replicate_results.status=="ok") & np.isclose(replicate_results.confidence_level,.95)]
    a=ok[ok.method=="ppi"][["experiment","replicate","learner","learner_label","parameter","ci_width","covered"]].rename(columns={"ci_width":"ppi_width","covered":"ppi_covered"})
    b=ok[ok.method=="ppi_plus_plus_v2"][["experiment","replicate","learner","parameter","ci_width","covered"]].rename(columns={"ci_width":"v2_width","covered":"v2_covered"})
    paired=a.merge(b,on=["experiment","replicate","learner","parameter"]); paired["ratio"]=paired.v2_width/paired.ppi_width
    out=paired.groupby(["experiment","learner","learner_label","parameter"],as_index=False).agg(
        v2_to_ppi_width_ratio_median=("ratio","median"),v2_to_ppi_width_ratio_q25=("ratio",lambda x:x.quantile(.25)),v2_to_ppi_width_ratio_q75=("ratio",lambda x:x.quantile(.75)),
        ppi_coverage_95=("ppi_covered","mean"),v2_coverage_95=("v2_covered","mean"),ppi_avg_width_95=("ppi_width","mean"),v2_avg_width_95=("v2_width","mean"))
    out["coverage_difference"]=out.v2_coverage_95-out.ppi_coverage_95
    out=out.merge(lam,on=["experiment","learner"],how="left")
    cols=["experiment","learner","learner_label","parameter","lambda_median","lambda_q25","lambda_q75","lambda_equal_1_rate","lambda_below_1_rate","v2_to_ppi_width_ratio_median","v2_to_ppi_width_ratio_q25","v2_to_ppi_width_ratio_q75","ppi_coverage_95","v2_coverage_95","coverage_difference","ppi_avg_width_95","v2_avg_width_95"]
    return out[cols].sort_values(["experiment","learner","parameter"]).reset_index(drop=True)
