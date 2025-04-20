import pandas as pd
import re
import os
import matplotlib.pyplot as plt

# Function to get the columns that have data of different dtypes
def detect_mixed_type_columns(df):
    return [
        col for col in df.columns 
        if len({type(value).__name__ for value in df[col].dropna().unique()}) > 1
    ]

# Function to convert numeric strings to numbers 
def try_convert(value):
    if isinstance(value, str) and value.replace('.', '', 1).isdigit():
        return float(value) if '.' in value else int(value)
    return value

# Function to convert columns to a single dtype
def convert_column(column):
    unique_values = column.dropna().unique()
    types = {type(value).__name__ for value in unique_values}

    # Convert numeric strings to actual numbers
    column = column.map(try_convert)

    # Recalculate types after conversion
    unique_values = column.dropna().unique()
    types = {type(value).__name__ for value in unique_values}

    # Ensure final dtype
    if types <= {'int'}:
        return column.astype('int')
    if types <= {'float', 'int'}:
        return column.astype('float')
    if 'str' in types or types == {'str', 'float'}:
        return column.astype('string').fillna('')

    return column

# Function to change object dtype to the dtype of values present in the column
def enforce_dtype(column, dtype):
    if dtype == 'int':
        return column.astype('int')

    elif dtype == 'float':
        return column.astype('float')

    elif dtype == 'str':
        # Check if all values in the string column are numeric (ZIP codes, etc.)
        if column.str.replace('.', '', 1).str.isdigit().all():
            return column.astype('float')  # Convert numeric strings to integers

        return column.astype('string').fillna('')  # Keep it as string if mixed

    return column


# Function to convert all the columns in a df from mixed to single
def standardize_mixed_type_columns(df):
    # Get all the columns with more than 1 dtype of values
    mixed_type_columns = detect_mixed_type_columns(df)

    # Convert only mixed-type columns
    df[mixed_type_columns] = df[mixed_type_columns].apply(convert_column, axis=0)

    # Ensure single-type columns are correctly enforced
    for col in df.columns:
        unique_types = {type(value).__name__ for value in df[col].dropna().unique()}
        if len(unique_types) == 1:
            dtype = list(unique_types)[0]
            df[col] = enforce_dtype(df[col], dtype)

    return df, mixed_type_columns


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda x: re.sub(r'^\_+', '', re.sub(r'\W+', '', x.strip().replace(" ", "_"))))
    return df

def save_plot_to_claude_graphs(fig, filename: str, work_dir: str) -> str:
    """Save a matplotlib figure to the claude_graphs/ directory."""
    folder = os.path.join(work_dir, "claude_graphs")
    os.makedirs(folder, exist_ok=True)

    save_path = os.path.join(folder, filename)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    return save_path

def save_df_to_work_dir(df: pd.DataFrame, filename: str, work_dir: str) -> str:
    """
    Save a pandas DataFrame to the specified filename inside the WORK_DIR.
    Supports .csv and .xlsx formats.
    """
    folder = os.path.join(work_dir, "claude_modified_files")
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)

    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".csv":
        df.to_csv(save_path, index=False)
    elif ext in [".xlsx", ".xls"]:
        df.to_excel(save_path, index=False)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return save_path



