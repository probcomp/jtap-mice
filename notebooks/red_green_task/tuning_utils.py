import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# -------------------------------------------------
# ------- Comprehensive Hyperparameter Analysis & Multivariate Modeling ------
# -------------------------------------------------

def comprehensive_hyperparameter_analysis_multivar(
    hyperparam_dict, metric_values, metric_name="Metric", max_interactions=15, max_bins=10
):
    """
    Hyperparameter univariate, bivariate, and multivariate analysis for a chosen error metric.
    Adds: multivariate regression, partial dependence and a direct summary of tuning ranges.

    Args:
        hyperparam_dict: dict mapping hyperparameter name -> list of values
        metric_values: array-like metric values to analyze
        metric_name: str, metric description for labeling
        max_interactions: int, max hyperparam pairs for 2D map
        max_bins: int, maximum number of bins to use for binned analysis
    """
    print(f"\n{'='*60}\nCOMPREHENSIVE HYPERPARAMETER ANALYSIS FOR {metric_name}\n{'='*60}")

    # ------ Build DataFrame ------
    df_data = hyperparam_dict.copy()
    df_data['metric'] = metric_values
    df = pd.DataFrame(df_data)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Dataset shape: {df.shape}")
    print(f"{metric_name} statistics: mean={df['metric'].mean():.4f}, std={df['metric'].std():.4f}")
    print(f"{metric_name} range: [{df['metric'].min():.4f}, {df['metric'].max():.4f}]")

    hyperparams = [col for col in df.columns if col != 'metric']

    # 1. CORRELATION ANALYSIS
    print(f"\n{'-'*40}\n1. CORRELATION ANALYSIS\n{'-'*40}")
    correlations, p_values = {}, {}
    for param in hyperparams:
        try:
            corr, p_val = stats.pearsonr(df[param], df['metric'])
        except Exception:
            corr, p_val = np.nan, np.nan
        correlations[param] = corr
        p_values[param] = p_val
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{param:30s}: r={corr:7.4f}, p={p_val:.4f} {significance}")

    # Spearman correlation (optional, for non-linear relationships)
    print("\nSpearman correlation (may be more robust for non-linear/uniform quantized params):")
    for param in hyperparams:
        try:
            scorr, spval = stats.spearmanr(df[param], df['metric'])
        except Exception:
            scorr, spval = np.nan, np.nan
        ssig = "***" if spval < 0.001 else "**" if spval < 0.01 else "*" if spval < 0.05 else ""
        print(f"{param:30s}: ρ={scorr:7.4f}, p={spval:.4f} {ssig}")

    # Plots: Pearson bar + Spearman overlay
    fig, ax = plt.subplots(figsize=(14, 6))
    bar1 = ax.bar(range(len(hyperparams)), [correlations[p] for p in hyperparams], color='dodgerblue', label="Pearson r", alpha=0.6)
    bar2 = ax.bar(range(len(hyperparams)), [stats.spearmanr(df[p], df['metric']).correlation if len(np.unique(df[p])) > 2 else 0 for p in hyperparams], 
                  color='orange', alpha=0.4, label="Spearman ρ")
    ax.axhline(0, color='k', linestyle=':')
    ax.set_xticks(range(len(hyperparams)))
    ax.set_xticklabels(hyperparams, rotation=45, ha='right')
    ax.set_ylabel(f'Correlation with {metric_name}')
    ax.set_title(f'Pearson vs Spearman Correlations')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Correlation matrix heatmap (all input and metric)
    fig, ax = plt.subplots(figsize=(12, 9))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, square=True, ax=ax, fmt=".2f")
    plt.title(f'Correlation Matrix - {metric_name}')
    plt.tight_layout()
    plt.show()

    # 2. FEATURE IMPORTANCE (Random Forest)
    print(f"\n{'-'*40}\n2. RANDOM FOREST FEATURE IMPORTANCE\n{'-'*40}")
    from sklearn.inspection import permutation_importance

    X, y = df[hyperparams], df['metric']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    rf.fit(X_scaled, y)
    importance_rf = pd.DataFrame({
        'Parameter': hyperparams,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("Random Forest Feature Importance:")
    for i, (_, row) in enumerate(importance_rf.iterrows(), 1):
        print(f"{i:2d}. {row['Parameter']:30s}: {row['Importance']:.4f}")

    plt.figure(figsize=(14, 6))
    bars = plt.bar(importance_rf['Parameter'], importance_rf['Importance'], color='c', alpha=0.7)
    plt.ylabel('Random Forest Importance')
    plt.title(f'Random Forest Feature Importance - {metric_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    result_perm = permutation_importance(rf, X_scaled, y, n_repeats=20, random_state=42, scoring='neg_mean_squared_error')
    importances_perm = pd.Series(result_perm.importances_mean, index=hyperparams)
    fig, ax = plt.subplots(figsize=(14, 6))
    importances_perm.sort_values(ascending=False).plot.bar(ax=ax, color='salmon')
    plt.ylabel('Permuted Importance')
    plt.title('Permutation Feature Importances (Random Forest)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 3. INDIVIDUAL HYPERPARAMETER vs METRIC SCATTERS
    print(f"\n{'-'*40}\n3. INDIVIDUAL HYPERPARAMETER ANALYSIS\n{'-'*40}")
    n_params = len(hyperparams)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    for i, param in enumerate(hyperparams):
        ax = axes[i]
        ax.scatter(df[param], df['metric'], alpha=0.5, s=25)
        try:
            fit = np.polyfit(df[param], df['metric'], 1)
            ax.plot(np.sort(df[param]), np.poly1d(fit)(np.sort(df[param])), 'r--', alpha=0.85)
        except Exception:
            pass
        ax.set_xlabel(param)
        ax.set_ylabel(metric_name)
        ax.set_title(f'{param} vs {metric_name}\nr={correlations.get(param, np.nan):.3f}')
        ax.grid(True, alpha=0.3)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()

    # 4. BINNED ANALYSIS (NOW EQUAL POPULATION BINS WITH max_bins PARAM)
    print(f"\n{'-'*40}\n4. BINNED PERFORMANCE ANALYSIS (Equal Population Bins)\n{'-'*40}")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    for i, param in enumerate(hyperparams):
        ax = axes[i]
        param_values = df[param]
        n_unique = len(np.unique(param_values))
        n_bins = min(max_bins, n_unique)
        # Bin logic: always use pd.qcut for binning (equal population bins) if more than unique threshold
        if n_unique <= n_bins:
            bins = sorted(np.unique(param_values))
            bin_labels = [str(b) for b in bins]
            binned = df.groupby(param)[['metric']].agg(['mean', 'std', 'count'])
            means = binned['metric']['mean'].values
            stds = binned['metric']['std'].values
            counts = binned['metric']['count'].values
            x = np.arange(len(bins))
        else:
            # Use qcut for equal-population bins
            # Remove tied duplicates
            try:
                labels = pd.qcut(param_values, q=n_bins, duplicates='drop')
            except ValueError:
                # Fall back to cut if qcut fails
                labels = pd.cut(param_values, bins=n_bins)
            df_tmp = df.assign(_bin=labels)
            binned = df_tmp.groupby('_bin')[['metric']].agg(['mean', 'std', 'count'])
            means = binned['metric']['mean'].values
            stds = binned['metric']['std'].values
            counts = binned['metric']['count'].values
            bin_labels = [str(idx) for idx in binned.index]
            x = np.arange(len(bin_labels))
        bars = ax.bar(x, means, yerr=stds, alpha=0.75, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=35, ha='right')
        ax.set_xlabel(param)
        ax.set_ylabel(f'Mean {metric_name}')
        ax.set_title(f'{param}: Mean {metric_name} by bins')
        ax.grid(True, alpha=0.3)
        for xpos, count in zip(x, counts):
            ax.text(xpos, means[xpos] + (stds[xpos] if not np.isnan(stds[xpos]) else 0)*0.05, f"n={count}", ha='center', fontsize=8)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()

    # 5. TOP vs BOTTOM PERFORMERS
    print(f"\n{'-'*40}\n5. TOP vs BOTTOM PERFORMERS ANALYSIS\n{'-'*40}")
    # Lower RMSE is "better", so top=lowest decile (10%)
    top_10pct = df['metric'].quantile(0.1)
    bot_10pct = df['metric'].quantile(0.9)
    top = df[df['metric'] <= top_10pct]
    bottom = df[df['metric'] >= bot_10pct]
    print(f"Top 10% performers (lowest {metric_name} <= {top_10pct:.4f}): {len(top)} samples")
    print(f"Bottom 10% performers (highest {metric_name} >= {bot_10pct:.4f}): {len(bottom)} samples")
    diffs = []
    for param in hyperparams:
        top_mean = top[param].mean()
        bot_mean = bottom[param].mean()
        top_std = top[param].std()
        bot_std = bottom[param].std()
        try:
            tstat, tpval = stats.ttest_ind(top[param], bottom[param], nan_policy='omit')
        except Exception:
            tpval = np.nan
        sig = "***" if tpval < 0.001 else "**" if tpval < 0.01 else "*" if tpval < 0.05 else ""
        print(f"{param:30s}: Top={top_mean:.3f}±{top_std:.3f}, Bot={bot_mean:.3f}±{bot_std:.3f}, Diff={top_mean-bot_mean:.3f} {sig}")
        diffs.append(dict(Parameter=param, Top=top_mean, Bot=bot_mean, Difference=top_mean-bot_mean,
                          Top_Std=top_std, Bot_Std=bot_std, P=tpval, Significance=sig))
    df_compare = pd.DataFrame(diffs)
    x = np.arange(len(hyperparams))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    w = 0.35
    ax1.bar(x - w/2, df_compare['Top'], width=w, color='green', alpha=0.7, label='Top 10%')
    ax1.bar(x + w/2, df_compare['Bot'], width=w, color='crimson', alpha=0.7, label='Bottom 10%')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_compare['Parameter'], rotation=45, ha='right')
    ax1.set_ylabel('Hyperparameter Value')
    ax1.set_title('Mean Value for Top/Bottom 10% Performers')
    ax1.legend()
    for i, sig in enumerate(df_compare['Significance']):
        if sig:
            ax1.text(x[i], max(df_compare['Top'][i], df_compare['Bot'][i]), sig, ha='center', va='bottom', fontweight='bold')
    color_diff = ['green' if d > 0 else 'red' for d in df_compare['Difference']]
    bars = ax2.bar(x, df_compare['Difference'], color=color_diff, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_compare['Parameter'], rotation=45, ha='right')
    ax2.set_ylabel('Top - Bottom Value')
    ax2.set_title('Difference: Top10% - Bottom10%')
    ax2.axhline(0, color='k', linestyle=':')
    for i, sig in enumerate(df_compare['Significance']):
        if sig:
            ax2.text(x[i], df_compare['Difference'][i], sig, ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 6. HYPERPARAM INTERACTIONS (2D) - both axes now use qcut if possible, bins set by max_bins
    print(f"\n{'-'*40}\n6. HYPERPARAMETER INTERACTION ANALYSIS\n{'-'*40}")
    from itertools import combinations
    param_order = importance_rf['Parameter'].tolist()
    pair_list = list(combinations(param_order, 2))
    if max_interactions is not None:
        pair_list = pair_list[:max_interactions]
    print(f"Visualizing {len(pair_list)} 2D hyperparam interactions out of {len(hyperparams)*(len(hyperparams)-1)//2} possible pairs.")
    n_pairs = len(pair_list)
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten()
    for idx, (param1, param2) in enumerate(pair_list):
        ax = axes[idx]
        try:
            x, y = df[param1], df[param2]
            metric = df['metric']
            n_bins1 = min(max_bins, len(np.unique(x)))
            n_bins2 = min(max_bins, len(np.unique(y)))
            try:
                x_bins = pd.qcut(x, q=n_bins1, duplicates='drop', labels=False)
            except Exception:
                x_bins = pd.cut(x, bins=n_bins1, labels=False)
            try:
                y_bins = pd.qcut(y, q=n_bins2, duplicates='drop', labels=False)
            except Exception:
                y_bins = pd.cut(y, bins=n_bins2, labels=False)
            grid_shape = (len(np.unique(y_bins)), len(np.unique(x_bins)))
            means_grid = np.full(grid_shape, np.nan)
            for i in range(grid_shape[1]):
                for j in range(grid_shape[0]):
                    mask = (x_bins == i) & (y_bins == j)
                    if mask.any():
                        means_grid[j, i] = metric[mask].mean()
            # Calculate bin edges for display
            try:
                x_bins_edges = pd.qcut(x, q=n_bins1, retbins=True, duplicates='drop')[1]
            except Exception:
                x_bins_edges = np.histogram_bin_edges(x, bins=n_bins1)
            try:
                y_bins_edges = pd.qcut(y, q=n_bins2, retbins=True, duplicates='drop')[1]
            except Exception:
                y_bins_edges = np.histogram_bin_edges(y, bins=n_bins2)
            im = ax.imshow(means_grid, origin='lower', aspect='auto', interpolation='nearest',
                           cmap="RdYlGn_r", extent=[x_bins_edges[0], x_bins_edges[-1], y_bins_edges[0], y_bins_edges[-1]])
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_title(f"{param1} x {param2}\n{metric_name}")
            plt.colorbar(im, ax=ax)
        except Exception:
            ax.set_visible(False)
    for i in range(n_pairs, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.show()

    # 7. MULTIVARIATE REGRESSION
    print(f"\n{'-'*40}\n7. MULTIVARIATE ANALYSIS\n{'-'*40}")
    X = df[hyperparams]
    y = df['metric']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    results = {}
    models = {'LinearRegression': LinearRegression(), 'Ridge(alpha=1)': Ridge(alpha=1.0), 'Lasso(alpha=0.05)': Lasso(alpha=0.05)}
    for name, model in models.items():
        model.fit(Xs, y)
        y_pred = model.predict(Xs)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        results[name] = {'r2': r2, 'rmse': rmse, 'coeffs': model.coef_}
        print(f"{name:18s}: R^2={r2:.3f}, RMSE={rmse:.4f}")
        if hasattr(model, 'coef_'):
            message = f"  Top coefficients ({name}):"
            print(message)
            coefs = pd.Series(model.coef_, index=hyperparams)
            for (param, val) in coefs.abs().sort_values(ascending=False).head(5).items():
                print(f"    {param:>20s}: {coefs[param]:+.3f}")
    # Show coefficients barplot
    fig, ax = plt.subplots(figsize=(12,5))
    coefs = pd.Series(models['Ridge(alpha=1)'].coef_, index=hyperparams)
    coefs.plot(kind='bar', color=['green' if x<0 else 'coral' for x in coefs], ax=ax)
    ax.axhline(0, linestyle=':', color='k')
    plt.title("Ridge Regression Standardized Coefficients")
    plt.ylabel("Coefficient (standardized)")
    plt.tight_layout()
    plt.show()

    # Partial dependence line plots for top features
    print("\nPartial dependence (regression-based; 1D):")
    n_top = 3
    for param in coefs.abs().sort_values(ascending=False).head(n_top).index:
        idx = hyperparams.index(param)
        Xtmp = np.tile(np.median(Xs, axis=0), (50,1))
        rng = np.linspace(Xs[:,idx].min(), Xs[:,idx].max(), 50)
        Xtmp[:,idx] = rng
        preds = models['Ridge(alpha=1)'].predict(Xtmp)
        plt.plot(scaler.inverse_transform([np.eye(len(hyperparams))[idx]*val for val in rng])[:,idx], preds, label=param)
    plt.xlabel("Feature value")
    plt.ylabel(f'Predicted {metric_name} (Ridge)')
    plt.legend()
    plt.title("Regression Partial 1D Dependence Plots (Ridge, top features)")
    plt.tight_layout()
    plt.show()

    # 8. SUMMARY & NEXT-ROUND TUNING RECOMMENDATIONS
    print(f"\n{'-'*40}\n8. SUMMARY & RECOMMENDATIONS FOR NEXT TUNING ROUND\n{'-'*40}")

    print("\nSelecting hyperparameters using a COMBINATION of prior analyses (RandomForest importance, Ridge/Lasso coefficients, marginal plots, and separation of top- vs. bottom-performing runs)...")

    # 1. Features with high RandomForest importance
    strong_import_rf = importance_rf.head(7)['Parameter'].tolist()
    # 2. Features with large-magnitude Ridge coefficients
    strong_coef_ridge = coefs.abs().sort_values(ascending=False).head(7).index.tolist()
    # 3. Features with high separation in top vs. bottom decile plots
    separation = {}
    for p in hyperparams:
        tvals, bvals = top[p], bottom[p]
        # Take separation as the effect size (Cohen's d)
        pooled_sd = np.sqrt((tvals.var() + bvals.var())/2)
        if pooled_sd > 0:
            separation[p] = abs(tvals.mean()-bvals.mean())/pooled_sd
        else:
            separation[p] = 0
    separation_sorted = sorted(separation.items(), key=lambda x: -x[1])
    strong_sep = [p for p,sep in separation_sorted[:7]]

    # Automated inclusion: union of all above (these are considered "important")
    suggested_params = list(dict.fromkeys(strong_import_rf + strong_coef_ridge + strong_sep))
    suggested_params_set = set(suggested_params)

    print("- Key hyperparameters recommended for further attention (union of top 7 from RF importance, Ridge coeffs, and separation):")
    emoji_important = "⭐"
    emoji_regular = "•"
    for i, p in enumerate(suggested_params):
        print(f"  {i+1:2d}. {p} {emoji_important}")

    print("\nSuggested value ranges for NEXT ROUND (based on best 15% performers):")
    # Use a larger chunk of best runs for more stable estimate
    top_frac = 0.15
    ntop = max(1, int(len(df) * top_frac))
    topN = df.nsmallest(ntop, 'metric')

    # Provide recommended ranges for ALL hyperparams, with an emoji to indicate importance
    print("Legend: ⭐ = key/important for tuning, • = secondary/less critical")

    for param in hyperparams:
        best_vals = topN[param]
        boundary_status = ""
        # Heuristic: check if top values are at the edge of original search range
        search_min, search_max = df[param].min(), df[param].max()
        if abs(best_vals.min()-search_min) < 1e-6 or abs(best_vals.max() - search_max) < 1e-6:
            boundary_status = " [! at range boundary]"
        emoji = emoji_important if param in suggested_params_set else emoji_regular
        print(f" {emoji} {param:>25s}: best range = {best_vals.min():.3f} – {best_vals.max():.3f}  (mean={best_vals.mean():.3f} ± {best_vals.std():.3f}){boundary_status}")

    print("\nCROSS-CHECK: Features *not* marked as ⭐ but showing moderate importance in RF or significant in Lasso or separation may warrant some exploration (to avoid overfitting to your current data).")
    secondary_features = [p for p in hyperparams if (p not in suggested_params) and (
        p in importance_rf.head(12)['Parameter'].tolist() or
        abs(coefs[hyperparams.index(p)]) > 0.15 or
        separation[p] > 0.3
    )]
    if len(secondary_features) > 0:
        print("Also consider, for limited exploration:")
        for p in secondary_features:
            print(f"  – {p}")

    print("\nBALANCED TUNING ADVICE:")
    print("▶ Focus tuning compute on key (⭐) features above, but reserve ~15–25% of experiments to covering secondary/uncertain (•) features to avoid overfitting local minima.")
    print("▶ Reduce grid size for non-informative/flat-effect parameters.")
    print("▶ If a feature's best ranges always hug boundaries, expand the search.")
    print("▶ If distributions for top and bottom performers overlap greatly, de-prioritize that parameter.")

    print("\nGeneral practice: aim for a grid or sampler that oversamples the promising regions identified, but still ensures moderate global coverage for uncertainty quantification.")
    print("\n(Refer to coefficient plots, partial dependence, and top/bottom marginal plots for final tuning round design.)")
    return df, importance_rf, df_compare, coefs, suggested_params
