from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import pearsonr
from prefect import flow, task, get_run_logger


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "assignments" / "resources" / "happiness_project"
OUTPUT_DIR = BASE_DIR / "assignments_01" / "outputs"


def standardize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace("%", " percent ")
    name = name.replace("/", " ")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def rename_columns_to_standard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [standardize_column_name(col) for col in df.columns]

    rename_map = {}

    for col in df.columns:
        if col in {"ranking", "rank", "overall_rank"}:
            rename_map[col] = "ranking"
        elif col in {"country", "country_name", "country_or_region"}:
            rename_map[col] = "country"
        elif col in {"region", "regional_indicator"}:
            rename_map[col] = "region"
        elif col in {"happiness_score", "score", "ladder_score", "life_ladder"}:
            rename_map[col] = "happiness_score"
        elif col in {
            "gdp_per_capita",
            "logged_gdp_per_capita",
            "economy_gdp_per_capita",
            "economy_gdp_per_capita_",
        }:
            rename_map[col] = "gdp_per_capita"
        elif col in {"social_support", "family"}:
            rename_map[col] = "social_support"
        elif col in {
            "healthy_life_expectancy",
            "health_life_expectancy",
            "health_healthy_life_expectancy",
        }:
            rename_map[col] = "healthy_life_expectancy"
        elif col in {"freedom", "freedom_to_make_life_choices"}:
            rename_map[col] = "freedom_to_make_life_choices"
        elif col in {"generosity"}:
            rename_map[col] = "generosity"
        elif col in {
            "perceptions_of_corruption",
            "trust_government_corruption",
            "corruption",
        }:
            rename_map[col] = "perceptions_of_corruption"

    return df.rename(columns=rename_map)


def detect_csv_delimiter(sample_path: Path) -> str:
    text = sample_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    preview = "\n".join(text[:5])
    return ";" if preview.count(";") > preview.count(",") else ","


def to_numeric_series(series: pd.Series) -> pd.Series:
    """
    Convert string-looking numeric data into real numeric dtype.
    Handles:
    - surrounding spaces
    - commas used as decimal separators
    - blank strings
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


@task(retries=3, retry_delay_seconds=2)
def load_multiple_years() -> pd.DataFrame:
    logger = get_run_logger()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    delimiter = detect_csv_delimiter(csv_files[0])
    logger.info(f"Detected delimiter: {delimiter}")

    all_frames = []

    for csv_file in csv_files:
        match = re.search(r"(20\d{2})", csv_file.name)
        if not match:
            logger.info(f"Skipping file without year in name: {csv_file.name}")
            continue

        year = int(match.group(1))

        df = pd.read_csv(
            csv_file,
            sep=delimiter,
            decimal=".",
            engine="python",
        )

        df = rename_columns_to_standard(df)
        df["year"] = year

        numeric_columns = [
            "ranking",
            "happiness_score",
            "gdp_per_capita",
            "social_support",
            "healthy_life_expectancy",
            "freedom_to_make_life_choices",
            "generosity",
            "perceptions_of_corruption",
            "year",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = to_numeric_series(df[col])

        text_columns = ["country", "region"]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        all_frames.append(df)
        logger.info(f"Loaded {csv_file.name} with {len(df)} rows")

    if not all_frames:
        raise ValueError("No valid CSV files were loaded.")

    merged_df = pd.concat(all_frames, ignore_index=True)

    merged_path = OUTPUT_DIR / "merged_happiness.csv"
    merged_df.to_csv(merged_path, index=False)

    logger.info(f"Saved merged dataset to {merged_path}")
    logger.info(f"Merged shape: {merged_df.shape}")

    return merged_df


@task
def descriptive_statistics(df: pd.DataFrame) -> dict:
    logger = get_run_logger()

    overall_mean = df["happiness_score"].mean()
    overall_median = df["happiness_score"].median()
    overall_std = df["happiness_score"].std()

    logger.info(f"Overall mean happiness_score: {overall_mean:.4f}")
    logger.info(f"Overall median happiness_score: {overall_median:.4f}")
    logger.info(f"Overall std happiness_score: {overall_std:.4f}")

    mean_by_year = df.groupby("year")["happiness_score"].mean().sort_index()
    for year, value in mean_by_year.items():
        logger.info(f"Year {int(year)}: mean happiness_score = {value:.4f}")

    if "region" in df.columns:
        mean_by_region = (
            df.groupby("region")["happiness_score"]
            .mean()
            .sort_values(ascending=False)
        )
        for region, value in mean_by_region.items():
            logger.info(f"Region {region}: mean happiness_score = {value:.4f}")
    else:
        mean_by_region = pd.Series(dtype=float)
        logger.info("No region column found.")

    return {
        "overall_mean": float(overall_mean),
        "overall_median": float(overall_median),
        "overall_std": float(overall_std),
        "mean_by_year": mean_by_year.to_dict(),
        "mean_by_region": mean_by_region.to_dict(),
    }


@task
def create_visualizations(df: pd.DataFrame) -> None:
    logger = get_run_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(df["happiness_score"].dropna(), bins=20)
    plt.title("Distribution of Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "happiness_histogram.png")
    plt.close()
    logger.info("Saved happiness_histogram.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="year", y="happiness_score")
    plt.title("Happiness Score by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "happiness_by_year.png")
    plt.close()
    logger.info("Saved happiness_by_year.png")

    if "gdp_per_capita" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(df["gdp_per_capita"], df["happiness_score"])
        plt.title("GDP per Capita vs Happiness Score")
        plt.xlabel("GDP per Capita")
        plt.ylabel("Happiness Score")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "gdp_vs_happiness.png")
        plt.close()
        logger.info("Saved gdp_vs_happiness.png")
    else:
        logger.info("Skipping GDP scatter plot because gdp_per_capita column was not found.")

    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()
    logger.info("Saved correlation_heatmap.png")


@task
def hypothesis_testing(df: pd.DataFrame) -> dict:
    logger = get_run_logger()
    results = {}

    scores_2019 = df.loc[df["year"] == 2019, "happiness_score"].dropna()
    scores_2020 = df.loc[df["year"] == 2020, "happiness_score"].dropna()

    if len(scores_2019) > 1 and len(scores_2020) > 1:
        t_stat, p_value = stats.ttest_ind(scores_2019, scores_2020, equal_var=False)
        mean_2019 = scores_2019.mean()
        mean_2020 = scores_2020.mean()

        logger.info(f"2019 mean happiness: {mean_2019:.4f}")
        logger.info(f"2020 mean happiness: {mean_2020:.4f}")
        logger.info(f"2019 vs 2020 t-statistic: {t_stat:.4f}")
        logger.info(f"2019 vs 2020 p-value: {p_value:.6f}")

        if p_value < 0.05:
            interpretation = (
                "Average happiness changed between 2019 and 2020, "
                "and the difference is unlikely due to chance."
            )
        else:
            interpretation = (
                "There is not enough evidence to say average happiness "
                "changed between 2019 and 2020."
            )

        logger.info(interpretation)

        results["prepost_2020"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_2019": float(mean_2019),
            "mean_2020": float(mean_2020),
            "interpretation": interpretation,
        }

    if "region" in df.columns:
        region_means = (
            df.groupby("region")["happiness_score"]
            .mean()
            .sort_values(ascending=False)
        )
        if len(region_means) >= 2:
            region1 = region_means.index[0]
            region2 = region_means.index[1]

            region1_scores = df.loc[df["region"] == region1, "happiness_score"].dropna()
            region2_scores = df.loc[df["region"] == region2, "happiness_score"].dropna()

            if len(region1_scores) > 1 and len(region2_scores) > 1:
                t2, p2 = stats.ttest_ind(region1_scores, region2_scores, equal_var=False)
                logger.info(
                    f"Region comparison {region1} vs {region2}: "
                    f"t={t2:.4f}, p={p2:.6f}"
                )

                results["region_test"] = {
                    "region1": region1,
                    "region2": region2,
                    "t_stat": float(t2),
                    "p_value": float(p2),
                }

    return results


@task
def correlation_and_multiple_comparisons(df: pd.DataFrame) -> dict:
    logger = get_run_logger()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    explanatory_cols = [col for col in numeric_cols if col != "happiness_score"]

    results = {}
    valid_test_count = 0

    for col in explanatory_cols:
        subset = df[[col, "happiness_score"]].dropna()
        if len(subset) < 3:
            continue

        r, p = pearsonr(subset[col], subset["happiness_score"])
        results[col] = {"r": float(r), "p_value": float(p)}
        valid_test_count += 1

    if valid_test_count == 0:
        logger.info("No valid correlation tests were run.")
        return {"results": {}, "adjusted_alpha": None}

    adjusted_alpha = 0.05 / valid_test_count
    logger.info(f"Number of correlation tests: {valid_test_count}")
    logger.info(f"Bonferroni adjusted alpha: {adjusted_alpha:.6f}")

    for col, result in results.items():
        sig_05 = result["p_value"] < 0.05
        sig_bonf = result["p_value"] < adjusted_alpha

        logger.info(
            f"{col}: r={result['r']:.4f}, p={result['p_value']:.6f}, "
            f"significant@0.05={sig_05}, significant@bonferroni={sig_bonf}"
        )

        result["significant_original"] = sig_05
        result["significant_bonferroni"] = sig_bonf

    return {
        "results": results,
        "adjusted_alpha": adjusted_alpha,
    }


@task
def summary_report(
    df: pd.DataFrame,
    stats_result: dict,
    hypothesis_result: dict,
    corr_result: dict,
) -> None:
    logger = get_run_logger()

    total_countries = df["country"].nunique() if "country" in df.columns else "Unknown"
    total_years = df["year"].nunique() if "year" in df.columns else "Unknown"

    logger.info(f"Total number of countries: {total_countries}")
    logger.info(f"Total number of years: {total_years}")

    region_means = stats_result.get("mean_by_region", {})
    if region_means:
        region_series = pd.Series(region_means).sort_values(ascending=False)
        logger.info(f"Top 3 regions: {region_series.head(3).to_dict()}")
        logger.info(f"Bottom 3 regions: {region_series.tail(3).to_dict()}")
    else:
        logger.info("Region summary unavailable.")

    if "prepost_2020" in hypothesis_result:
        logger.info(
            f"2019 vs 2020 result: "
            f"{hypothesis_result['prepost_2020']['interpretation']}"
        )

    corr_results = corr_result.get("results", {})
    bonf = {k: v for k, v in corr_results.items() if v.get("significant_bonferroni")}

    if bonf:
        strongest = max(bonf.items(), key=lambda item: abs(item[1]["r"]))
        logger.info(
            f"Strongest Bonferroni-significant variable: {strongest[0]} "
            f"(r={strongest[1]['r']:.4f}, p={strongest[1]['p_value']:.6f})"
        )
    else:
        logger.info("No correlation remained significant after Bonferroni correction.")


@flow
def happiness_pipeline():
    df = load_multiple_years()
    stats_result = descriptive_statistics(df)
    create_visualizations(df)
    hypothesis_result = hypothesis_testing(df)
    corr_result = correlation_and_multiple_comparisons(df)
    summary_report(df, stats_result, hypothesis_result, corr_result)


if __name__ == "__main__":
    happiness_pipeline()