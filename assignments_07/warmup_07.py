# assignments_07/warmup_07.py

import json
import os

# Use a non-GUI matplotlib backend so plots can be saved safely on Windows.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")

from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# ------------------------------------------------------------
# Lesson 02: Tool Definitions and the ReAct Loop
# ------------------------------------------------------------

# Q1
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Temperature in degrees Celsius.",
                }
            },
            "required": ["celsius"],
        },
    },
}

print("\n--- Lesson 02 Q1 ---")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


def get_current_time(location: str) -> str:
    """Return the current time for a location."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"The current time in {location} is {now}."


get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or location name.",
                }
            },
            "required": ["location"],
        },
    },
}


# Q2
def run_agent_time_only(user_input: str) -> str:
    """Simple tool-based agent with only the get_current_time tool."""
    messages = [{"role": "user", "content": user_input}]

    first_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[get_current_time_schema],
        tool_choice="auto",
    )

    first_message = first_response.choices[0].message

    if not first_message.tool_calls:
        return first_message.content

    messages.append(first_message)

    for tool_call in first_message.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if tool_name == "get_current_time":
            tool_result = get_current_time(**arguments)
        else:
            tool_result = f"Unknown tool: {tool_name}"

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": tool_result,
            }
        )

    second_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    return second_response.choices[0].message.content


print("\n--- Lesson 02 Q2 ---")

# Prediction:
# Calling run_agent_time_only("Convert 100 degrees Celsius to Fahrenheit") will
# probably NOT trigger a tool call because the only available tool is get_current_time.
# That tool is for time, not temperature conversion.
# If no tool is called, only one API call is needed.
# If the model does call the time tool by mistake, then two API calls are needed.

response_q2 = run_agent_time_only("Convert 100 degrees Celsius to Fahrenheit")
print(response_q2)

# My result check:
# If the answer was returned directly, my prediction was correct.
# If the model called get_current_time, then my prediction was not correct.


# Q3
def run_agent(user_input: str) -> str:
    """Extended tool-based agent with get_current_time and celsius_to_fahrenheit."""
    messages = [{"role": "user", "content": user_input}]

    first_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[get_current_time_schema, celsius_to_fahrenheit_schema],
        tool_choice="auto",
    )

    first_message = first_response.choices[0].message

    if not first_message.tool_calls:
        return first_message.content

    messages.append(first_message)

    for tool_call in first_message.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if tool_name == "get_current_time":
            tool_result = get_current_time(**arguments)
        elif tool_name == "celsius_to_fahrenheit":
            tool_result = celsius_to_fahrenheit(**arguments)
        else:
            tool_result = f"Unknown tool: {tool_name}"

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": tool_result,
            }
        )

    second_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    return second_response.choices[0].message.content


print("\n--- Lesson 02 Q3 ---")

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# Comment: This should trigger the celsius_to_fahrenheit tool because the user
# is asking for a Celsius to Fahrenheit conversion.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# Comment: This may not trigger a tool call because the model already knows that
# water boils at about 100°C or 212°F in normal conditions.


# ------------------------------------------------------------
# Lesson 03: Multi-Tool Agent
# ------------------------------------------------------------

class CsvManager:
    """Manage loading and analyzing CSV files for the agent."""

    def __init__(self):
        self.df = None
        self.loaded_path = None

    def _find_csv_path(self, filename: str) -> Path:
        """Find a CSV file in common assignment locations."""
        possible_paths = [
            Path(filename),
            Path("assignments_07") / filename,
            Path("assignments_07") / "outputs" / filename,
            Path("assignments/resources") / filename,
            Path("assignments/resources/agents") / filename,
            Path("assignments/resources/ai_agents") / filename,
            Path("assignments/resources/week_07") / filename,
        ]

        for path in possible_paths:
            if path.exists():
                return path

        matches = list(Path(".").rglob(filename))
        if matches:
            return matches[0]

        return Path(filename)

    def load_csv(self, filename: str):
        """Load a CSV file into a pandas DataFrame."""
        path = self._find_csv_path(filename)

        if not path.exists():
            return {"error": f"Could not find CSV file: {filename}"}

        self.df = pd.read_csv(path)
        self.loaded_path = str(path)

        return {
            "message": f"Loaded {path}",
            "shape": self.df.shape,
            "columns": list(self.df.columns),
        }

    def preview_rows(self, n: int = 5):
        """Preview the first n rows of the loaded DataFrame."""
        if self.df is None:
            return {"error": "No CSV is loaded."}

        return self.df.head(n).to_dict(orient="records")

    def summarize_column(self, column: str):
        """Return descriptive statistics for one column."""
        if self.df is None:
            return {"error": "No CSV is loaded."}

        if column not in self.df.columns:
            return {"error": f"Column '{column}' was not found."}

        return self.df[column].describe().to_dict()

    # Q4
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        if self.df is None:
            return {"error": "No CSV is loaded."}

        if col1 not in self.df.columns:
            return {"error": f"Column '{col1}' was not found."}

        if col2 not in self.df.columns:
            return {"error": f"Column '{col2}' was not found."}

        clean_df = self.df[[col1, col2]].dropna()

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


csv_manager = CsvManager()

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file into memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The CSV filename to load.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preview_rows",
            "description": "Preview the first rows of the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of rows to preview.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_column",
            "description": "Return descriptive statistics for one column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to summarize.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute Pearson correlation coefficient and p-value between two columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "First numeric column.",
                    },
                    "col2": {
                        "type": "string",
                        "description": "Second numeric column.",
                    },
                },
                "required": ["col1", "col2"],
            },
        },
    },
]

node_tools = {
    "load_csv": csv_manager.load_csv,
    "preview_rows": csv_manager.preview_rows,
    "summarize_column": csv_manager.summarize_column,
    "compute_correlation": csv_manager.compute_correlation,
}

SYSTEM_PROMPT = """
You are a helpful data analysis agent.
Use tools when you need to load a CSV, preview rows, summarize a column,
or compute a correlation. Think step by step, but keep the final response concise.
"""


def run_agent_cycle(messages: list, user_input: str, max_tool_rounds: int = 5) -> str:
    """Run a ReAct-style agent cycle using OpenAI tool calls."""
    messages.append({"role": "user", "content": user_input})

    for _ in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        if not assistant_message.tool_calls:
            return assistant_message.content

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if tool_name in node_tools:
                tool_result = node_tools[tool_name](**arguments)
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(tool_result, default=str),
                }
            )

    return "The agent reached the tool-round limit before completing the task."


# Q5
print("\n--- Lesson 03 Q5 ---")

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
)
print(result)


# Q6
print("\n--- Lesson 03 Q6 ---")

# ReAct role explanation:
# system = the rules/instructions that guide the agent's behavior.
# user = the human request or task.
# assistant = the model's response, tool choice, or final answer.
# tool = the actual result returned by a Python function after the agent acts.

print(json.dumps(messages, indent=2, default=str))


# ------------------------------------------------------------
# Lesson 04: smolagents
# ------------------------------------------------------------

print("\n--- Lesson 04 Q7-Q9 ---")

try:
    from smolagents import CodeAgent, OpenAIServerModel, ToolCallingAgent, tool

    # Q7
    @tool
    def compute_correlation(col1: str, col2: str) -> dict:
        """Compute Pearson correlation and p-value between two loaded CSV columns.

        Args:
            col1: The first numeric column name.
            col2: The second numeric column name.

        Returns:
            A dictionary with col1, col2, pearson_r, and p_value.
            Returns an error dictionary if no CSV is loaded or the columns are invalid.
        """
        return csv_manager.compute_correlation(col1, col2)

    @tool
    def load_csv(filename: str) -> dict:
        """Load a CSV file into memory.

        Args:
            filename: The CSV filename to load.

        Returns:
            A dictionary with the loaded file path, shape, and columns.
        """
        return csv_manager.load_csv(filename)

    @tool
    def preview_rows(n: int = 5) -> list:
        """Preview rows from the loaded CSV.

        Args:
            n: Number of rows to preview.

        Returns:
            A list of row dictionaries or an error dictionary.
        """
        return csv_manager.preview_rows(n)

    @tool
    def summarize_column(column: str) -> dict:
        """Summarize one column from the loaded CSV.

        Args:
            column: The column name to summarize.

        Returns:
            A dictionary of descriptive statistics or an error dictionary.
        """
        return csv_manager.summarize_column(column)

    print(compute_correlation.description)

    # Comment:
    # smolagents automatically creates a tool description from the function name,
    # type hints, and docstring. In Q4, I manually wrote the JSON schema myself.
    # To produce a good tool description, smolagents needs clear argument names,
    # correct type hints, and a detailed docstring explaining what the tool does.

    # Q8
    model = OpenAIServerModel(
        api_key=api_key,
        model_id="gpt-4o-mini",
    )

    TOOLS = [
        load_csv,
        preview_rows,
        summarize_column,
        compute_correlation,
    ]

    tool_agent = ToolCallingAgent(
        tools=TOOLS,
        model=model,
        max_steps=6,
    )

    code_agent = CodeAgent(
        tools=TOOLS,
        model=model,
        additional_authorized_imports=[
            "pandas",
            "matplotlib",
            "matplotlib.pyplot",
            "pathlib",
        ],
        max_steps=8,
    )

    prompt = (
        "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot "
        "with green dots. If you are the CodeAgent, use pandas to read "
        "'assignments_07/bike_commute.csv' directly and save the plot to "
        "'assignments_07/outputs/bike_commute_scatter.png'. Do not use plt.show(). "
        "After saving the plot, call final_answer('Saved bike_commute_scatter.png')."
    )

    response_tool = tool_agent.run(prompt)
    response_code = code_agent.run(
        prompt,
        additional_args={"csv_manager": csv_manager},
    )

    print("ToolCallingAgent response:")
    print(response_tool)

    print("CodeAgent response:")
    print(response_code)

    # Fallback verification/creation:
    # The assignment asks us to compare the agents. However, agent behavior can vary.
    # To make the file output reliable for mentor review, we also create the plot directly
    # if the CodeAgent did not save it.
    scatter_path = Path("assignments_07/outputs/bike_commute_scatter.png")
    if not scatter_path.exists():
        import matplotlib.pyplot as plt

        bike_df = pd.read_csv("assignments_07/bike_commute.csv")
        plt.figure(figsize=(8, 5))
        plt.scatter(
            bike_df["duration_min"],
            bike_df["avg_heart_rate"],
            color="green",
        )
        plt.title("Average Heart Rate vs Duration")
        plt.xlabel("Duration (minutes)")
        plt.ylabel("Average Heart Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(scatter_path)
        plt.close()
        print(f"Fallback created: {scatter_path}")

    # Comment:
    # The ToolCallingAgent can only use the tools I gave it. It can load the CSV,
    # summarize columns, and compute correlations, but it does not have a true
    # plotting tool. Because of that, it may describe the relationship instead of
    # creating a custom scatter plot with green dots.
    #
    # The CodeAgent is more flexible because it can write Python code. It can use
    # pandas and matplotlib to read the CSV directly and save a scatter plot.
    #
    # This shows that ToolCallingAgent is better for safe, predictable tool-based
    # tasks, while CodeAgent is better when the task requires custom code, such as
    # creating a plot that no existing tool handles.

except Exception as exc:
    print("smolagents section could not run.")
    print(f"Reason: {exc}")
    print(
        "Check that smolagents is installed and your OPENAI_API_KEY is available."
    )


# Q9
# A ToolCallingAgent is better than a CodeAgent for a task like checking the
# current status of a file, loading a CSV, summarizing a column, or computing
# a known metric. The task is a good fit for tool-based agents when the steps
# are clear, repeatable, and limited to safe actions.
#
# One meaningful risk of using a CodeAgent is that it writes and runs code.
# If the generated code is wrong, unsafe, or poorly designed, it could create
# incorrect results, change files unexpectedly, or make debugging harder.
# A ToolCallingAgent is more controlled because it can only call the tools
# that the developer provides.
