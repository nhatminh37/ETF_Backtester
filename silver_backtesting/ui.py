"""Optional UI helpers: ipywidgets table, Dash app."""

from __future__ import annotations

import socket
from typing import Any

import pandas as pd

from .config import SilverBacktestState

try:
    import dash
    import ipywidgets as widgets
    from dash import dash_table, dcc, html

    DASH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional UI deps
    DASH_AVAILABLE = False
    dash = None  # type: ignore[assignment]
    widgets = None  # type: ignore[assignment]
    dash_table = None  # type: ignore[assignment]
    dcc = None  # type: ignore[assignment]
    html = None  # type: ignore[assignment]

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - script mode
    def display(obj: Any) -> None:
        print(obj)

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)


def widget_table(df: pd.DataFrame) -> None:
    if not DASH_AVAILABLE or widgets is None:
        raise ImportError("ipywidgets is required for widget_table. Install with: pip install ipywidgets")

    col_select1 = widgets.Dropdown(options=df.columns.tolist(), description="Column 1:")
    value_input1 = widgets.Text(description="Filter 1:")
    col_select2 = widgets.Dropdown(options=df.columns.tolist(), description="Column 2:")
    value_input2 = widgets.Text(description="Filter 2:")
    col_select3 = widgets.Dropdown(options=df.columns.tolist(), description="Column 3:")
    value_input3 = widgets.Text(description="Filter 3:")
    sort_col = widgets.Dropdown(options=df.columns.tolist(), description="Sort by:")
    sort_order = widgets.ToggleButtons(
        options=["Ascending", "Descending"],
        description="Order:",
    )
    output = widgets.Output(layout=widgets.Layout(height="600px", width="100%", overflow="auto"))

    def filter_df(change=None):
        with output:
            output.clear_output()
            filtered = df.copy()

            col1, val1 = col_select1.value, value_input1.value
            if val1 != "":
                filtered = filtered[filtered[col1].astype(str).str.contains(val1, case=False, na=False)]

            col2, val2 = col_select2.value, value_input2.value
            if val2 != "":
                filtered = filtered[filtered[col2].astype(str).str.contains(val2, case=False, na=False)]

            col3, val3 = col_select3.value, value_input3.value
            if val3 != "":
                filtered = filtered[filtered[col3].astype(str).str.contains(val3, case=False, na=False)]

            ascending = sort_order.value == "Ascending"
            filtered = filtered.sort_values(by=sort_col.value, ascending=ascending)
            display(filtered)

    for widget in [
        col_select1, value_input1, col_select2, value_input2,
        col_select3, value_input3, sort_col, sort_order,
    ]:
        widget.observe(filter_df, names="value")

    display(
        widgets.HBox([col_select1, value_input1]),
        widgets.HBox([col_select2, value_input2]),
        widgets.HBox([col_select3, value_input3]),
        widgets.HBox([sort_col, sort_order]),
        output,
    )
    filter_df()


def generate_dash_table(df: pd.DataFrame) -> Any:
    if not DASH_AVAILABLE or dash_table is None:
        raise ImportError("dash is required for generate_dash_table. Install with: pip install dash")

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": col, "id": col} for col in df.columns],
        style_table={
            "overflowX": "auto",
            "border": "1px solid #ddd",
            "borderRadius": "4px",
        },
        style_header={
            "backgroundColor": "#0066cc",
            "color": "white",
            "fontWeight": "bold",
            "padding": "12px",
            "textAlign": "left",
            "border": "1px solid #0052a3",
        },
        style_cell={
            "textAlign": "left",
            "padding": "10px",
            "fontSize": "14px",
            "border": "1px solid #ddd",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_data={"backgroundColor": "white", "color": "#333"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
            {
                "if": {"state": "active"},
                "backgroundColor": "#e3f2fd",
                "border": "1px solid #0066cc",
            },
        ],
        page_size=20,
        sort_action="native",
        filter_action="native",
        sort_mode="multi",
    )


def create_dash_app(
    state: SilverBacktestState,
    df_test: pd.DataFrame | None = None,
    port: int = 8065,
) -> Any:
    """Launch an interactive Dash table viewer."""
    if not DASH_AVAILABLE or dash is None:
        raise ImportError("dash is required for create_dash_app. Install with: pip install dash")

    if df_test is None:
        df_test = state.df_data.loc[
            (state.df_data["Trading date"].isin([20260310]))
            & (state.df_data["ind_nyse_rth"] == True)
        ][
            ["DT", "DT_cst", "PAAS", "SLV"]
            + [x for x in state.df_data.columns if "PAAS-SLV" in x]
        ].copy()
        df_test[["DT", "DT_cst"]] = df_test[["DT", "DT_cst"]].astype(str)

    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("Interactive Data Table"),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Test Data",
                        children=[generate_dash_table(df_test)],
                    ),
                    dcc.Tab(
                        label="Appended Data",
                        children=[generate_dash_table(state.df_appended)],
                    ),
                ]
            ),
        ]
    )
    app.run(host=local_ip, port=port, debug=True)
    return app
