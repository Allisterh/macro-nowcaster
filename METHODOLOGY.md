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

The fix is point-in-time data. Each observation is stored with a `vintage_date` (when
it became knowable). The point-in-time store reconstructs the panel "as of" any date
using only vintages published on or before it. When true ALFRED vintages are not
available for a series, the client falls back to applying the indicator's typical
publication lag, which captures the timing trap even if not the revision trap.

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

## 3. The composite: a mixed-frequency dynamic factor model

The composite activity index is the common factor extracted by a dynamic factor model
estimated with the Kalman filter and the EM algorithm (`statsmodels`
`DynamicFactorMQ`, the Banbura-Modugno specification). This is the academic standard
for nowcasting and it matters for three reasons:

- It handles the ragged edge natively. Different series end on different dates because
  of publication lags; the Kalman filter treats the missing recent cells as states to
  be estimated rather than requiring imputation.
- It models dynamics. The factor follows an autoregressive process, so the filter
  distinguishes signal from month-to-month noise.
- It supports a principled news decomposition, because the filter gives the
  marginal impact of each new observation on the state.

PCA on mean-imputed data is retained as a fallback and cross-check. If the DFM fails
to converge in some environment, the system degrades to PCA rather than crashing, and
it reports which method produced the factor.

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
standard error, which becomes the uncertainty band. A point estimate with no error
bar reads as naive; the band communicates that a nowcast is a distribution.

## 6. Regimes

A three-state Gaussian HMM labels each month as Expansion, Slowdown, or Contraction
from the activity factor. States are relabelled by their mean activity after fitting,
so the output is interpretable regardless of the HMM's arbitrary initialization order.
The regime conditions the narrative and can gate the allocation rule.

## 7. News decomposition

When a new vintage arrives, the system compares the factor at a target date under the
old and new panels and attributes the revision to the cells that changed or newly
arrived. The attribution is leave-one-out: knock each new observation out, refit, and
measure how much of the revision disappears. This produces the desk-friendly statement
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

## 9. From signal to decision

The allocation overlay maps the macro state to an equity weight: a base tilt that
rises with activity momentum (bounded by a hyperbolic tangent) and falls with
recession probability. The signal is lagged one month so the backtest only ever acts
on information available before the return is earned. Performance is measured net of a
simple turnover cost against a buy-and-hold benchmark on the same series, using
annualized return, volatility, Sharpe, and maximum drawdown. The rule is intentionally
simple; the claim is "the signal can drive a decision," not "this is a tuned strategy."

## 10. Operations

Validation gates reject a malformed panel before it reaches a model and warn on stale
or low-coverage data. Drift monitoring computes a population stability index per
indicator and flags distribution shifts. Calibration checks whether predicted
recession probabilities match realized frequencies. The pipeline produces a single
artifact that the API serves and the frontend consumes, and a scheduled job rebuilds
it from the latest data so the dashboard is never stale.
