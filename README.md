# Vibe Preprocessing and Analysis MCP Server for CSV files

A powerful MCP (Model Control Protocol) server for preprocessing and analyzing CSV files. This server provides a suite of tools for data manipulation, visualization, and analysis through an intuitive interface.

## Features

- **Data Loading and Management**
  - Load CSV files from a specified working directory
  - Set and manage working directories
  - List files in the working directory
  - Save processed dataframes to new files

- **Data Preprocessing**
  - Handle mixed data types in columns
  - Manage null values with various strategies:
    - Remove rows with nulls
    - Fill with mean/median/mode
    - Forward/backward fill
    - Fill with constant values
  - Drop and rename columns
  - Run custom dataframe editing code
  - Save processed data to new files

- **Data Analysis**
  - Generate comprehensive data descriptions
  - Create correlation matrices with visualizations
  - Handle mixed data types in columns
  - Run custom analysis code

- **Data Visualization**
  - Create various types of plots:
    - Line plots
    - Bar charts
    - Scatter plots
    - Histograms with KDE
    - Box plots
    - Violin plots
    - Pie charts
    - Count plots
    - Kernel Density Estimation plots
  - Custom graph generation through code
  - Save visualizations to the working directory
  - Run custom visualization code

## Setup Instructions

### Prerequisites
- Python 3.x
- uv (recommended package manager). I recommend using [uv](https://docs.astral.sh/uv/) to manage the server. 

### Installation
1. Add MCP and required dependencies:
```bash
uv add "mcp[cli]"
uv add pandas matplotlib seaborn numpy
```

2. Install the server in Claude Desktop:
```bash
mcp install server.py
```

### Alternative Installation with pip
If you prefer using pip:
```bash
pip install "mcp[cli]" pandas matplotlib seaborn numpy
```

## Usage

1. Start the MCP server:
```bash
uv run mcp
```

2. Test the server using MCP Inspector:
```bash
mcp dev server.py
```

You can install this server in [Claude Desktop](https://claude.ai/download) and interact with it right away by running:
```bash
mcp install server.py
```

Alternatively, you can test it with the MCP Inspector:
```bash
mcp dev server.py
```

## Available Tools

### Data Management
- `send_work_dir()`: Retrieve the current working directory
- `set_work_dir(new_work_dir)`: Set a new working directory
- `list_work_dir_files()`: List files in the current working directory
- `load_csv(filename)`: Load a CSV file into the system
- `save_global_df(filename)`: Save the current dataframe to a file

### Data Preprocessing
- `handle_column_mixed_types()`: Handle columns with mixed data types
- `handle_null_values(strategy, columns)`: Handle null values in the dataset with various strategies
- `drop_columns(columns)`: Remove specified columns
- `rename_columns(column_mapping)`: Rename columns in the dataframe
- `run_custom_df_edit_code(code)`: Execute custom dataframe manipulation code

### Data Analysis
- `describe_df()`: Generate a statistical summary of the dataframe
- `generate_correlation_matrix()`: Create a correlation matrix with visualization

### Data Visualization
- `plot_graph(graph_type, x_column, y_column, output_filename)`: Create various types of plots
  - Supported graph types: line, bar, scatter, hist, box, violin, pie, count, kde
- `run_custom_graph_code(code)`: Execute custom visualization code

## Environment Variables

- `WORK_DIR`: The working directory where files are read from and saved to

## Error Handling

The server includes comprehensive error handling for:
- Missing working directories
- File not found errors
- Data loading and processing errors
- Invalid operations on empty dataframes
- Mixed data type handling
- Custom code execution errors
- Invalid column names
- Invalid graph types
- Null value handling errors

## Contributing

Feel free to submit issues and enhancement requests!
