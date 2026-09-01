# Methodology

This note documents the modelling choices and the reasoning behind them. The goal
is not novelty for its own sake; it is to build a nowcast that is honest and that a
desk would actually trust.

## 1. The data problem comes first

Most macro and ML finance projects fail on data, not models. Two specific traps:

**Look-ahead through revisions.** The payrolls figure you download today is not the
figure that printed in that month. Statistical agencies revise heavily. If you train
or standardize on revised data, your historical signal silently knows things it could
not have known, and the backtest is fiction.

**Look-ahead through publication lag.** On the last day of a month you do not yet
have that month's industrial production; it publishes mid-next-month. A panel that
assumes every series is available immediately overstates real-time skill.

The fix is point-in-time data. For each as-of date the panel is reconstructed using
only values that were knowable then. For monthly series the full ALFRED revision
history is downloaded once, cached, and sliced in memory for every as-of date, taking
the latest revision published on or before that date. Caching the history once rather
than re-querying it for every step is what makes a multi-decade replay feasible
without overwhelming the FRED endpoint. Daily and weekly financial series (the yield
curve, the VIX, credit spreads) are not revised, so they use a publication-lag proxy
instead: the final series shifted by the indicator's typical release delay. Any series
whose vintage history cannot be fetched falls back to the same lag proxy, which
captures the timing trap even if not the revision trap. A timing-only mode (release
lags for every series) is available via an environment switch for fast iteration when
the full vintage download is not warranted.

## 2. Stationarity and standardization

Each indicator is transformed to a stationary form: year-over-year percent change for
level series (payrolls, production, sales), year-over-year difference for rates
(unemployment), and raw level for series that are already stationary spreads or
indices (the yield curve, credit spreads, the VIX, financial conditions). Every
series is then z-scored and sign-aligned so that a positive standardized value always
means a stronger economy.

Standardization has two modes. The live snapshot uses full-sample statistics for
interpretability and stability. The pseudo-real-time backtest uses an expanding
window, so the z-score at each date reflects only the distribution observed up to that
date. Mixing these up is a common and subtle source of leakage.

## 3. The composite: a dynamic factor model on a monthly panel

The composite activity index is the common factor extracted by a dynamic factor model
estimated with the Kalman filter and the EM algorithm (`statsmodels`
`DynamicFactorMQ`, the Banbura-Modugno specification). This is the academic standard
for nowcasting and it matters for three reasons:

- It handles the ragged edge natively. Different series end on different dates because
  of publication lags; the Kalman filter treats the missing recent cells as states to
  be estimated rather than requiring imputation.
- It models dynamics. The factor follows an autoregressive process, so the filter
  distinguishes signal from month-to-month noise.
- It would support a principled news decomposition, because the filter gives the
  marginal impact of each new observation on the state. That is not what is
  implemented: the news module refits the factor with one cell knocked out at a
  time (see section 7), and it is not wired into the pipeline.

One thing this specification is not: every series is averaged to month-end before
estimation, so the state space is monthly. `DynamicFactorMQ` can carry quarterly
series alongside monthly ones; no quarterly series is passed to it here. "Mixed
frequency" in this project means the inputs arrive daily, weekly and monthly and
end on different dates, not that the model aggregates frequencies internally.

Series with very low coverage over the sample (for example a credit spread that only
begins in the late 1990s) are excluded from the state-space estimation, since a column
that is mostly missing destabilizes convergence; their loadings and contributions are
still computed against the fitted factor afterwards, so the dashboard keeps the full
indicator list. PCA on mean-imputed data is retained as a fallback and cross-check. If
the DFM fails to converge in some environment, the system degrades to PCA rather than
crashing, logs the reason, and reports which method produced the factor.

## 4. Recession probability

Two separate models, because they answer different questions.

The coincident nowcast is a probit of the NBER recession indicator on the activity
factor and the yield-curve slope. Its in-sample AUC is very high, but that number is
partly circular: the factor is built from the same activity data that defines a
recession, so a high coincident AUC is expected and is not evidence of foresight.

The leading model is a probit of a recession occurring within twelve months on the
10-year minus 3-month Treasury spread, the most robust single recession predictor in
the literature. This is the genuinely predictive object, and it carries a real lead
time rather than describing the present.

Probit can hit perfect separation when the factor is a very strong classifier, so a
regularized logistic regression is the fallback to keep estimation stable.

## 5. GDP nowcast

Two estimators map the monthly factor to quarterly GDP growth. The bridge equation
aggregates the monthly factor to a quarterly average and regresses GDP on it: simple
and transparent. The unrestricted MIDAS regression uses the three within-quarter
monthly factor readings as separate regressors, so a partially complete quarter still
yields a nowcast from whatever months have printed. Both report an in-sample residual
standard error, which becomes the uncertainty band. Two honest caveats: it is an
in-sample number, so it excludes parameter uncertainty and any out-of-sample
degradation, and it is constant across the quarter, where a real nowcast's
uncertainty should shrink as the quarter fills in. Its empirical coverage has not
been measured. Treat it as a rough scale, not a calibrated interval.

The GDP nowcast is also the one output with **no out-of-sample evaluation**: the
replay in section 8 scores the composite and the recession probability only.

## 6. Regimes

A three-state Gaussian HMM labels each month as Expansion, Slowdown, or Contraction
from the activity factor. States are relabelled by their mean activity after fitting,
so the output is interpretable regardless of the HMM's arbitrary initialization order.
The regime conditions the narrative and can gate the allocation rule.

## 7. News decomposition (experimental, not wired in)

Nothing in the pipeline, the API or the published site calls this module; it runs
only from its unit test, and no release attribution is published anywhere. What it
does when called: compare the factor at a target date under the old and new panels
and attribute the revision to the cells that changed or newly arrived. The
attribution is leave-one-out: knock each new observation out, refit, and measure
how much of the revision disappears. This produces the desk-friendly statement
"the nowcast rose because payrolls surprised," which is a flow of information rather
than a static contribution chart.

## 8. Honest evaluation

The pseudo-real-time replay is the centerpiece. For each month in the test window it
rebuilds the panel as of that date, standardizes on an expanding window, fits the
factor, and produces the nowcast, then scores it against final data. Two numbers
matter: the correlation between the real-time and final composite (how much
revisions move the signal) and the out-of-sample recession AUC (how well the
real-time probability anticipates realized recessions). The out-of-sample AUC is
always lower than the in-sample number, and reporting the lower one is the point.

`RESULTS.md` is the single source of truth for these figures, and it is not quoted
here so the two cannot drift apart. It is regenerated by `make backtest`, which is
run **by hand** rather than on a schedule, and each run stamps the file with the
commit, whether real ALFRED vintages or the release-lag proxy were used, the replay
window and the exact package versions that produced the numbers. Read the header
before quoting the numbers.

Three caveats belong with any figure from that file. The replay estimates the factor
with PCA rather than the DFM, for speed across hundreds of refits, so it evaluates a
close cousin of the model the site ships. The window contains three recessions
(2001, 2008-09, 2020), one of them a trivially detectable outlier, so an AUC computed
over 377 months has an effective sample size nearer three - the confidence interval
is wide whatever the third decimal says. And figures produced before the September
2026 alignment fix are not comparable: until then the replay read the last *fitted*
probability off a label-truncated training set, which lagged the as-of month by the
recognition lag.

The gap between the in-sample and out-of-sample AUC is the honest cost of scoring on
data the model has not seen; a result with no gap would be the warning sign, not a
triumph.

## 9. External benchmarks

A homemade index is only credible next to the public versions of the same thing, so
the dashboard pulls three benchmarks live from FRED at snapshot build time and plots
them alongside the model's own lines.

The composite activity index is compared to CFNAI-MA3, the Chicago Fed's three-month
moving average of its National Activity Index. The three-month average is the correct
comparison because the composite is itself a smoothed latent factor; correlating it
against raw monthly CFNAI would penalize it for filtering high-frequency noise. The
current correlation is roughly 0.64. The recession probability is shown next to the
Chauvet-Piger smoothed recession probabilities (FRED series `RECPROUSM156N`), a
dynamic-factor Markov-switching model on coincident variables, with a current
correlation near 0.67. The GDP nowcast is compared to the Atlanta Fed's GDPNow. The live gap is published
in `docs/latest.json` and on the dashboard rather than quoted here, because it moves:
on 1 September 2026 it was -1.8 pp, not the fraction of a point an earlier draft of
this note claimed. GDPNow is also a different object - a bridge-equation
reconstruction of the NIPA expenditure components, not a factor model - so the two
are not expected to agree closely.

These benchmarks are deliberately not tuned to match. The composite is a 30-series
dynamic factor model that includes seven financial series, and CFNAI-MA3 is an
85-series principal-component index that deliberately excludes financial variables,
so a moderate rather than near-perfect correlation is the expected and honest result.
Correlation is also not a skill test: this project does not yet compare its GDP
nowcast against an AR(1), a random walk, or the last published quarter. The
value of the comparison is not a high number; it is that the comparison bounds how far
the homemade lines stray from the institutional ones and makes any divergence visible
rather than hidden.

## 10. From signal to decision (experimental, not wired in)

This module runs only from its unit test, on random returns: no equity series is
configured and no backtest result exists in the repository. What it implements: the
allocation overlay maps the macro state to an equity weight: a base tilt that
rises with activity momentum (bounded by a hyperbolic tangent) and falls with
recession probability. The signal is lagged one month so the backtest only ever acts
on information available before the return is earned. Performance is measured net of a
simple turnover cost against a buy-and-hold benchmark on the same series, using
annualized return, volatility, Sharpe, and maximum drawdown. The rule is intentionally
simple; the claim is "the signal can drive a decision," not "this is a tuned strategy."

## 11. Operations

Validation gates reject a malformed panel before it reaches a model and warn on stale
or low-coverage data. Drift monitoring computes a population stability index per
indicator on the standardized panel, comparing the most recent seven years against the
earlier history in eight quantile bins. The conventional PSI alert thresholds of 0.1
and 0.25 are calibrated for short-horizon monitoring of a model's inputs against a
frozen training sample; this monitor instead asks whether a multi-year window differs
from fifty years of macro history, where broad secular drift is expected and is not a
model failure, so the thresholds are raised to a watch at 0.5 and an alert at 1.0.
Calibrated this way the monitor stays quiet on series near their historical norms and
flags the genuinely dislocated ones, such as the inverted-then-normalizing yield
curve, capacity utilization, and financial conditions, rather than lighting up every
row. A calibration report that checks whether predicted recession probabilities match
realized frequencies is implemented but never called. The pipeline produces a single
artifact that the API serves, and a scheduled GitHub Action rebuilds the static site
in `docs/` from the latest FRED data every day at 12:17 UTC. The older Streamlit
snapshot, `app/snapshot.json`, is no longer refreshed and should be treated as frozen.
