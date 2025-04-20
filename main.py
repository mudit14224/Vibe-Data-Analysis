# server.py
import os
import json
import pandas as pd
from mcp.server.fastmcp import FastMCP
from utils import standardize_mixed_type_columns, save_plot_to_claude_graphs, save_df_to_work_dir
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import base64
import io

# Create an MCP server
mcp = FastMCP("csv-xlsx-py", dependencies=["pandas", "matplotlib", "seaborn"])

df = []

@mcp.tool()
def send_work_dir() -> str:
    """Retrieve the current value of the WORK_DIR environment variable."""
    try:
        work_dir = os.environ.get("WORK_DIR")
        if not work_dir:
            return json.dumps({
                "success": False,
                "message": "WORK_DIR environment variable is not set."
            })
        return json.dumps({
            "success": True,
            "message": "WORK_DIR found.",
            "data": work_dir
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        })

@mcp.tool()
def set_work_dir(new_work_dir: str) -> str:
    """Set a new value for the WORK_DIR environment variable."""
    try:
        if not new_work_dir:
            return json.dumps({
                "success": False,
                "message": "New work directory cannot be empty."
            })
        os.environ["WORK_DIR"] = new_work_dir
        return json.dumps({
            "success": True,
            "message": f"WORK_DIR has been set to: {new_work_dir}"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        })


@mcp.tool()
def list_work_dir_files() -> str:
    """List all files and directories in the current WORK_DIR."""
    try:
        work_dir = os.environ.get("WORK_DIR")
        if not work_dir:
            return json.dumps({
                "success": False,
                "message": "WORK_DIR environment variable is not set."
            })

        if not os.path.exists(work_dir):
            return json.dumps({
                "success": False,
                "message": f"WORK_DIR path does not exist: {work_dir}"
            })

        files = os.listdir(work_dir)
        return json.dumps({
            "success": True,
            "message": f"Found {len(files)} items in {work_dir}.",
            "data": files
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        })

@mcp.tool()
def load_csv(filename: str) -> str:
    """Load a CSV file from WORK_DIR into global df and return the first 5 rows."""
    global df

    work_dir = os.environ.get("WORK_DIR")
    if not work_dir:
        return json.dumps({
            "success": False,
            "message": "WORK_DIR environment variable is not set."
        })

    file_path = os.path.join(work_dir, filename)

    if not os.path.exists(file_path):
        return json.dumps({
            "success": False,
            "message": f"CSV file not found at: {file_path}"
        })

    try:
        df = pd.read_csv(file_path)

        preview = df.head().to_dict(orient="records")
        return json.dumps({
            "success": True,
            "message": f"CSV loaded successfully from {file_path}.",
            "data": f"The first 5 rows of the file are: \n{preview}"
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Error loading CSV: {str(e)}"
        })
    
@mcp.tool()
def handle_column_mixed_types() -> str:
    """Finds and handles any columns with mixed datatypes."""
    global df

    if isinstance(df, list):
        return json.dumps({
            "success": False,
            "message": "No dataframe is loaded in the system!"
        })
    
    if df.empty:
        return json.dumps({
            "success": False,
            "message": "The dataframe is loaded in the system is empty!"
        }) 

    try: 
        df, mixed_type_columns = standardize_mixed_type_columns(df)

        return json.dumps({
            "success": True,
            "mixed_type_columns": f"These were the mixed type columns that were found: {mixed_type_columns}",
            "message": f"The mixed type columns were handles successfully!"
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Error loading CSV: {str(e)}"
        })
    

@mcp.tool()
def describe_df() -> str:
    """Return pandas describe() summary for the global dataframe."""
    global df

    if isinstance(df, list):
        return json.dumps({
            "success": False,
            "message": "No dataframe is loaded in the system!"
        })
    
    if df.empty:
        return json.dumps({
            "success": False,
            "message": "The dataframe is loaded in the system is empty!"
        }) 

    try:
        description = df.describe(include='all').to_dict()
        return json.dumps({
            "success": True,
            "message": "DataFrame summary using describe()",
            "data": description
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Error describing dataframe: {str(e)}"
        })
    
@mcp.tool()
def generate_correlation_matrix() -> str:
    """Generate correlation matrix heatmap and return/save both matrix and image."""
    global df

    if isinstance(df, list):
        return json.dumps({
            "success": False,
            "message": "No dataframe is loaded in the system!"
        })

    try:
        # Compute correlation matrix for numeric columns
        corr = df.corr(numeric_only=True)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
        ax.set_title("Correlation Matrix")

        # Save figure
        filename = "correlation_matrix.png"
        work_dir = os.environ.get("WORK_DIR")
        save_path = save_plot_to_claude_graphs(fig, filename, work_dir)

        # Convert matrix to nested dict for JSON
        corr_dict = corr.round(3).to_dict()

        return json.dumps({
            "success": True,
            "message": f"Correlation matrix saved to {save_path}",
            "data": corr_dict,
            "imagePath": save_path
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Error generating correlation matrix: {str(e)}"
        })


@mcp.tool()
def plot_graph(
    graph_type: str,
    x_column: str,
    y_column: str = None,
    output_filename: str = "graph.png"
) -> str:
    """
    Plot a graph using seaborn/matplotlib based on the loaded DataFrame and save it to 'claude_graphs/'.

    Parameters:
    ----------
    graph_type : str
        The type of graph to generate. Supported values:
        - 'line': Line plot
        - 'bar': Bar chart
        - 'scatter': Scatter plot
        - 'hist': Histogram with KDE
        - 'box': Box plot
        - 'violin': Violin plot
        - 'pie': Pie chart (only x_column used, for category frequencies)
        - 'count': Count plot (like bar chart for categorical frequency)
        - 'kde': Kernel Density Estimation plot

    x_column : str
        The column from the DataFrame to be used as the x-axis.

    y_column : str, optional (default = None)
        The column to be used for the y-axis. Some plots like 'pie' and 'count' ignore this.

    output_filename : str, optional (default = 'graph.png')
        The filename for the saved graph image inside the 'claude_graphs/' directory.
    """
    global df

    if isinstance(df, list):
        return json.dumps({
            "success": False,
            "message": "No dataframe is loaded in the system!"
        })

    if x_column not in df.columns or (y_column and y_column not in df.columns):
        return json.dumps({
            "success": False,
            "message": f"One or both columns not found in the dataframe: {x_column}, {y_column}"
        })

    try:
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))

        if graph_type == "line":
            sns.lineplot(data=df, x=x_column, y=y_column, ax=ax)
        elif graph_type == "bar":
            sns.barplot(data=df, x=x_column, y=y_column, ax=ax)
        elif graph_type == "scatter":
            sns.scatterplot(data=df, x=x_column, y=y_column, ax=ax)
        elif graph_type == "hist":
            sns.histplot(data=df, x=x_column, ax=ax, kde=True)
        elif graph_type == "box":
            sns.boxplot(data=df, x=x_column, y=y_column, ax=ax)
        elif graph_type == "violin":
            sns.violinplot(data=df, x=x_column, y=y_column, ax=ax)
        elif graph_type == "pie":
            pie_data = df[x_column].value_counts()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
        elif graph_type == "count":
            sns.countplot(data=df, x=x_column, ax=ax)
        elif graph_type == "kde":
            sns.kdeplot(data=df, x=x_column, ax=ax, fill=True)
        else:
            return json.dumps({
                "success": False,
                "message": f"""Unsupported graph type: {graph_type}. You can use only these types: \n
                line, bar, scatter, hist, box, violin, pie, count, kde. 
                If further graph types are needed to be supported please as admin to add them.
                """
            })

        ax.set_title(f"{graph_type.capitalize()} Plot: {x_column}" + (f" vs {y_column}" if y_column else ""))
        work_dir = os.environ.get("WORK_DIR")
        save_path = save_plot_to_claude_graphs(fig, output_filename, work_dir)

        return json.dumps({
            "success": True,
            "message": f"Graph saved to {save_path}",
            "imagePath": save_path
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Error generating plot: {str(e)}"
        })

@mcp.tool()
def run_custom_graph_code(code: str) -> str:
    """
    Run custom Python code provided by Claude to generate a single graph using global df.

    📌 Claude Instructions:
    -----------------------
    ✅ Use the global `df` (do not load or define it again)
    ✅ Create **only one graph at a time**
    ✅ Generate a matplotlib figure: `fig, ax = plt.subplots()`
    ✅ Define `output_filename` as the name of the graph file (e.g., 'myplot.png')
    ✅ Call: `save_path = save_plot_to_claude_graphs(fig, output_filename, work_dir)`

    ❌ Do NOT manually call plt.savefig()
    ❌ Do NOT skip assigning `save_path`

    Your job is to just create the graph and assign `save_path`. This tool handles execution and confirmation.

    Parameters:
    -----------
    code : str
        Valid Python code string that uses the global `df`, defines:
        - `fig`
        - `output_filename`
        - `save_path = save_plot_to_claude_graphs(...)`

    Returns:
    --------
    JSON string with:
        - success: Whether the execution was successful
        - message: Description of outcome
        - imagePath: Path where the image was saved (if successful)
    """
    global df

    if isinstance(df, list):
        return json.dumps({
            "success": False,
            "message": "No dataframe is loaded in the system!"
        })

    try:
        exec_context = {
            "df": df,
            "os": os,
            "save_plot_to_claude_graphs": save_plot_to_claude_graphs,
            "plt": plt,
            "sns": sns,
            "pd": pd,
            "work_dir": os.environ.get("WORK_DIR")
        }

        exec(code, exec_context)

        save_path = exec_context.get("save_path")
        if not save_path or not os.path.exists(save_path):
            return json.dumps({
                "success": False,
                "message": "Code executed but save_path is missing or the file does not exist. Make sure to call save_plot_to_claude_graphs(...) and assign it to save_path."
            })

        return json.dumps({
            "success": True,
            "message": "Code executed and graph saved successfully.",
            "imagePath": save_path
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Error while executing custom code: {str(e)}"
        })

@mcp.tool()
def handle_null_values(strategy: str, columns: list[str] = None) -> str:
    """
    Handle null values in the loaded df using a given strategy and save the modified file.

    Parameters:
    ----------
    strategy : str
        Strategy to handle nulls:
        - 'remove'     : Drop rows with nulls in the selected columns
        - 'mean'       : Fill with mean
        - 'median'     : Fill with median
        - 'mode'       : Fill with mode
        - 'ffill'      : Forward fill
        - 'bfill'      : Backward fill
        - 'constant:X' : Fill with constant value X

    columns : list[str], optional
        List of columns to apply the strategy on (default: all columns)

    Returns:
    -------
    JSON string with:
        - success: True/False
        - message: Outcome summary
        - outputFileName: Name of the new modified file
    """
    try:
        global df

        if columns is None or len(columns) == 0:
            columns = df.columns.tolist()

        for col in columns:
            if col not in df.columns:
                return json.dumps({
                    "success": False,
                    "message": f"Column '{col}' not found in the file."
                })

            if strategy == "remove":
                df.dropna(subset=[col], inplace=True)
            elif strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mode":
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
            elif strategy == "ffill":
                df[col].fillna(method="ffill", inplace=True)
            elif strategy == "bfill":
                df[col].fillna(method="bfill", inplace=True)
            elif strategy.startswith("constant:"):
                value = strategy.split("constant:")[1]
                df[col].fillna(value, inplace=True)
            else:
                return json.dumps({
                    "success": False,
                    "message": f"Invalid strategy: {strategy}"
                })

        return json.dumps({
            "success": True,
            "message": f"The df has been modified successfully!",
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Error: {str(e)}"
        })
    
@mcp.tool()
def run_custom_df_edit_code(code: str) -> str:
    """
    Run custom Python code from Claude to modify the global dataframe (`df`).

    🧠 Claude Instructions:
    -----------------------
    ✅ Use the global variable `df` — already loaded
    ✅ Modify `df` in-place or reassign `df = ...`
    ❌ Do NOT save the dataframe (saving is handled by a separate tool)
    ✅ Only modify one thing at a time — column drop, renaming, type conversion, etc.

    Parameters:
    -----------
    code : str
        Python code that modifies the global `df`.

    Returns:
    --------
    JSON string with:
        - success: True/False
        - message: Outcome summary
    """
    global df

    if isinstance(df, list):
        return json.dumps({
            "success": False,
            "message": "No dataframe is loaded in the system!"
        })

    try:
        exec_context = {
            "df": df,
            "pd": pd,
            "os": os
        }

        exec(code, exec_context)

        # Reassign df if Claude changed it
        df = exec_context.get("df", df)

        return json.dumps({
            "success": True,
            "message": "Code executed successfully. DataFrame updated."
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Error executing code: {str(e)}"
        })
    
@mcp.tool()
def save_global_df(filename: str) -> str:
    """
    Save the current global DataFrame to a file in the WORK_DIR.

    Parameters:
    -----------
    filename : str
        The file name to save (e.g., 'output.csv' or 'result.xlsx')
    """
    global df

    if isinstance(df, list):
        return json.dumps({
            "success": False,
            "message": "No dataframe is loaded to save."
        })

    try:
        work_dir = os.environ.get("WORK_DIR")
        if not work_dir:
            raise EnvironmentError("WORK_DIR environment variable is not set.")

        save_path = save_df_to_work_dir(df, filename, work_dir)

        return json.dumps({
            "success": True,
            "message": f"DataFrame saved successfully as {filename}.",
            "filePath": save_path
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Error saving DataFrame: {str(e)}"
        })
    
@mcp.tool()
def drop_columns(columns: list[str]) -> str:
    """
    Drop specified columns from the global DataFrame.

    Parameters:
    -----------
    columns : list[str]
        List of column names to drop from the DataFrame.
    """
    global df

    if isinstance(df, list):
        return json.dumps({
            "success": False,
            "message": "No dataframe is loaded in the system!"
        })

    invalid_columns = [col for col in columns if col not in df.columns]
    if invalid_columns:
        return json.dumps({
            "success": False,
            "message": f"Error: Columns not found: {', '.join(invalid_columns)}"
        })

    try:
        df.drop(columns=columns, inplace=True)
        return json.dumps({
            "success": True,
            "message": f"Columns dropped: {', '.join(columns)}"
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Error dropping columns: {str(e)}"
        })

@mcp.tool()
def rename_columns(column_mapping: dict) -> str:
    """
    Rename columns in the global DataFrame.

    Parameters:
    -----------
    column_mapping : dict
        Dictionary mapping old column names to new ones (e.g. {"old1": "new1"}).
    """
    global df

    if isinstance(df, list):
        return json.dumps({
            "success": False,
            "message": "No dataframe is loaded in the system!"
        })

    invalid_columns = [col for col in column_mapping if col not in df.columns]
    if invalid_columns:
        return json.dumps({
            "success": False,
            "message": f"Error: Columns not found: {', '.join(invalid_columns)}"
        })

    try:
        df.rename(columns=column_mapping, inplace=True)
        return json.dumps({
            "success": True,
            "message": f"Columns renamed: {column_mapping}"
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Error renaming columns: {str(e)}"
        })

