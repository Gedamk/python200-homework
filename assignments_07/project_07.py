# assignments_07/project_07.py

import os

# Use a non-GUI matplotlib backend.
# This prevents Windows/Tkinter thread crashes when the CodeAgent saves plots.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

DATA_PATH = "assignments_01/outputs/merged_happiness.csv"
OUTPUT_DIR = Path("assignments_07/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = None


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize common World Happiness column names for easier agent use."""
    dataframe = dataframe.copy()

    rename_map = {
        "Country": "country",
        "Country name": "country",
        "country_name": "country",
        "Regional indicator": "region",
        "Region": "region",
        "Happiness Score": "happiness_score",
        "Happiness.Score": "happiness_score",
        "Score": "happiness_score",
        "Ladder score": "happiness_score",
        "Economy (GDP per Capita)": "gdp_per_capita",
        "Economy..GDP.per.Capita.": "gdp_per_capita",
        "GDP per capita": "gdp_per_capita",
        "Logged GDP per capita": "gdp_per_capita",
        "Family": "social_support",
        "Social support": "social_support",
        "Health (Life Expectancy)": "healthy_life_expectancy",
        "Health..Life.Expectancy.": "healthy_life_expectancy",
        "Healthy life expectancy": "healthy_life_expectancy",
        "Freedom": "freedom_to_make_life_choices",
        "Freedom to make life choices": "freedom_to_make_life_choices",
        "Generosity": "generosity",
        "Trust (Government Corruption)": "perceptions_of_corruption",
        "Trust..Government.Corruption.": "perceptions_of_corruption",
        "Perceptions of corruption": "perceptions_of_corruption",
        "Year": "year",
    }

    dataframe = dataframe.rename(columns=rename_map)

    dataframe.columns = [
        str(col).strip().lower().replace(" ", "_").replace(".", "_")
        for col in dataframe.columns
    ]

    return dataframe


@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    This tool first tries to load the merged Week 1 dataset from
    assignments_01/outputs/merged_happiness.csv. If that file does not exist,
    it tries to load and merge all CSV files from
    assignments/resources/happiness_project/.

    Returns:
        dict: A dictionary containing the dataset shape, column names, and source.
        If the file cannot be loaded, returns a dictionary with an error message.
    """
    global df

    path = Path(DATA_PATH)

    if path.exists():
        df = pd.read_csv(path)
        df = normalize_columns(df)
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "source": str(path),
        }

    fallback_dir = Path("assignments/resources/happiness_project")

    if not fallback_dir.exists():
        return {
            "error": (
                f"Could not find {DATA_PATH} or fallback folder "
                f"{fallback_dir}."
            )
        }

    csv_files = sorted(fallback_dir.glob("*.csv"))

    if not csv_files:
        return {"error": f"No CSV files found in {fallback_dir}."}

    frames = []

    for csv_file in csv_files:
        temp_df = pd.read_csv(csv_file)

        if "year" not in [str(col).lower() for col in temp_df.columns]:
            digits = "".join(ch for ch in csv_file.stem if ch.isdigit())
            if digits:
                temp_df["year"] = int(digits[:4])

        frames.append(temp_df)

    df = pd.concat(frames, ignore_index=True)
    df = normalize_columns(df)

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "source": str(fallback_dir),
    }


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column.

    Args:
        column: The name of the column to summarize, such as happiness_score
            or gdp_per_capita.

    Returns:
        dict: Descriptive statistics from pandas describe(), including count,
        mean, standard deviation, minimum, quartiles, and maximum for numeric
        columns. Returns an error dictionary if no data is loaded or if the
        column does not exist.
    """
    global df

    if df is None:
        return {"error": "No data is loaded. Run load_happiness_data first."}

    if column not in df.columns:
        return {
            "error": f"Column '{column}' was not found.",
            "available_columns": list(df.columns),
        }

    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute Pearson correlation coefficient and p-value between two columns.

    Args:
        col1: The first numeric column name.
        col2: The second numeric column name.

    Returns:
        dict: A dictionary containing col1, col2, pearson_r, and p_value.
        The pearson_r value shows the strength and direction of the relationship.
        The p_value helps determine whether the relationship is statistically
        significant. Returns an error dictionary if no data is loaded, if a
        column does not exist, or if the values are not numeric.
    """
    global df

    if df is None:
        return {"error": "No data is loaded. Run load_happiness_data first."}

    if col1 not in df.columns:
        return {
            "error": f"Column '{col1}' was not found.",
            "available_columns": list(df.columns),
        }

    if col2 not in df.columns:
        return {
            "error": f"Column '{col2}' was not found.",
            "available_columns": list(df.columns),
        }

    clean_df = df[[col1, col2]].dropna()

    if clean_df.empty:
        return {"error": "No valid rows after removing missing values."}

    try:
        r, p_value = pearsonr(clean_df[col1], clean_df[col2])
    except Exception as exc:
        return {"error": f"Could not compute correlation: {exc}"}

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p_value), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a selected column for a year.

    Args:
        column: The column used for ranking, such as happiness_score.
        year: The year to filter the dataset.
        n: The number of top countries to return. Defaults to 5.

    Returns:
        dict: A dictionary containing the year, ranking column, and a list of
        top countries. Each country result includes the country name and the
        requested column value. Returns an error dictionary if no data is loaded,
        if required columns are missing, or if no data exists for the selected
        year.
    """
    global df

    if df is None:
        return {"error": "No data is loaded. Run load_happiness_data first."}

    if column not in df.columns:
        return {
            "error": f"Column '{column}' was not found.",
            "available_columns": list(df.columns),
        }

    if "year" not in df.columns:
        return {"error": "Column 'year' was not found."}

    if "country" not in df.columns:
        return {"error": "Column 'country' was not found."}

    year_df = df[df["year"] == year]

    if year_df.empty:
        return {"error": f"No data found for year {year}."}

    top_rows = (
        year_df.sort_values(by=column, ascending=False)
        .head(n)[["country", column]]
    )

    results = []

    for _, row in top_rows.iterrows():
        results.append(
            {
                "country": row["country"],
                column: row[column],
            }
        )

    return {
        "year": year,
        "ranked_by": column,
        "top_countries": results,
    }


def build_agent() -> CodeAgent:
    """Build and return the World Happiness CodeAgent."""
    if not api_key:
        raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")

    model = OpenAIServerModel(
        api_key=api_key,
        model_id="gpt-4o-mini",
    )

    system_prompt = """
You are a data analyst assistant for the World Happiness dataset.

Use the available tools for:
- loading the data
- summarizing columns
- computing correlations
- ranking countries

Important:
The load_happiness_data tool returns metadata only: shape, columns, and source.
It also loads the shared dataset for the other tools.

When you need to create custom plots, read the CSV directly with:
pd.read_csv("assignments_01/outputs/merged_happiness.csv")

For plots, use matplotlib with a non-GUI backend and save figures directly to files.
Do not call plt.show().

Be concise and student-friendly in your responses.
"""

    agent = CodeAgent(
        tools=[
            load_happiness_data,
            summarize_column,
            compute_correlation,
            get_top_n_countries,
        ],
        model=model,
        instructions=system_prompt,
        additional_authorized_imports=[
            "pandas",
            "matplotlib",
            "matplotlib.pyplot",
            "scipy.stats",
            "pathlib",
        ],
        max_steps=8,
    )

    return agent


def run_guided_queries(agent: CodeAgent) -> None:
    """Run the five required guided assignment queries."""
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        (
            "Plot happiness_score over the years as a line chart, with one line per region. "
            "Save the plot to assignments_07/outputs/happiness_by_region.png. "
            "Use pd.read_csv('assignments_01/outputs/merged_happiness.csv') for plotting. "
            "Do not use plt.show(). "
            "After saving the plot, call final_answer('Saved happiness_by_region.png')."
        ),
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)


def run_my_queries(agent: CodeAgent) -> None:
    """Run two custom student questions."""
    my_query_1 = (
        "Which region had the highest average happiness_score across all years? "
        "Use Python code if needed. "
        "Use pd.read_csv('assignments_01/outputs/merged_happiness.csv') if you need the full dataset."
    )

    print("\n--- My Query 1 ---")
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: This should trigger code generation because there is no specific
    # tool that calculates average happiness_score by region across all years.

    my_query_2 = (
        "Create a bar chart of the top 10 countries by happiness_score in 2020 "
        "and save it to assignments_07/outputs/top_10_2020_happiness.png. "
        "Use pd.read_csv('assignments_01/outputs/merged_happiness.csv') for plotting. "
        "Do not use plt.show(). "
        "After saving the plot, call final_answer('Saved top_10_2020_happiness.png')."
    )

    print("\n--- My Query 2 ---")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: This should trigger both tool use and code generation. The agent
    # may use the ranking tool to get top countries, then write matplotlib code
    # to create and save the bar chart.


if __name__ == "__main__":
    agent = build_agent()
    run_guided_queries(agent)
    run_my_queries(agent)


# --- Reflection ---
#
# 1. In Query 3, the agent should use the p-value to explain whether the
#    correlation is statistically significant. The common threshold is 0.05.
#    If p_value < 0.05, the result is usually considered statistically
#    significant. If p_value >= 0.05, it is usually not considered statistically
#    significant. The agent should also explain the Pearson r value as the
#    strength and direction of the relationship.
#
# 2. One response that surprised me was the plotting response. I expected the
#    agent to only use the tools I created, but the CodeAgent can also write
#    Python code when the tools are not enough. This is useful because no tool
#    was created for a multi-line chart by region, but the agent can still use
#    pandas and matplotlib to create it.
#
# 3. One additional tool that would make this agent more useful is
#    compare_regions(region1, region2, column). This tool would compare two
#    regions using a selected metric, such as happiness_score or gdp_per_capita.
#    It would help answer questions like, "How does North America compare with
#    Sub-Saharan Africa in happiness_score over time?"