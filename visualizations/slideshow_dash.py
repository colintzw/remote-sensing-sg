import base64
import glob
import io

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from PIL import Image


# Function to read and preprocess geotiff files
def read_img(file_path):
    return Image.open(file_path)


# Get list of geotiff files
geotiff_files = sorted(glob.glob("corrected_punggol_slices/*.tif"))

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container(
    [
        html.H1("Punggol 2016-2024", className="text-center my-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Button(
                            "⬅️", id="prev-button", className="btn btn-primary mr-2"
                        ),
                    ],
                    width=1,
                    className="d-flex align-items-center",
                ),
                dbc.Col(
                    [
                        html.Img(id="image-display", style={"width": "100%"}),
                        html.P(id="image-caption", className="text-center"),
                    ],
                    width=10,
                ),
                dbc.Col(
                    [
                        html.Button(
                            "➡️", id="next-button", className="btn btn-primary ml-2"
                        ),
                    ],
                    width=1,
                    className="d-flex align-items-center",
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Button(
                            "▶️", id="play-button", className="btn btn-success mr-2"
                        ),
                        html.Button(
                            "⏸️", id="pause-button", className="btn btn-warning mr-2"
                        ),
                        dcc.Slider(
                            id="speed-slider",
                            min=0.1,
                            max=5,
                            step=0.1,
                            value=1,
                            marks={i: str(i) for i in range(1, 6)},
                            className="mt-2",
                        ),
                        html.Div(
                            "Speed: 1 second(s)", id="speed-display", className="mt-2"
                        ),
                    ],
                    width={"size": 6, "offset": 3},
                    className="text-center",
                ),
            ]
        ),
        dcc.Interval(
            id="interval-component", interval=1000, n_intervals=0, disabled=True
        ),
        dcc.Store(id="current-index", data=0),
    ]
)


# Callback to update image
@app.callback(
    [
        Output("image-display", "src"),
        Output("image-caption", "children"),
        Output("current-index", "data"),
    ],
    [
        Input("interval-component", "n_intervals"),
        Input("prev-button", "n_clicks"),
        Input("next-button", "n_clicks"),
    ],
    [State("current-index", "data")],
)
def update_image(n_intervals, prev_clicks, next_clicks, current_index):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "No clicks yet"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "prev-button" and current_index > 0:
        current_index -= 1
    elif button_id == "next-button" and current_index < len(geotiff_files) - 1:
        current_index += 1
    elif button_id == "interval-component":
        current_index = (current_index + 1) % len(geotiff_files)

    filename = geotiff_files[current_index]
    date = filename.split("/")[-1].split("_")[-1].replace(".tif", "")
    img = read_img(filename)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode()

    return (
        f"data:image/png;base64,{img_str}",
        f"RGB-composite, {date} ({current_index + 1}/{len(geotiff_files)})",
        current_index,
    )


# Callback to control playback
@app.callback(
    Output("interval-component", "disabled"),
    [Input("play-button", "n_clicks"), Input("pause-button", "n_clicks")],
    [State("interval-component", "disabled")],
)
def control_playback(play_clicks, pause_clicks, is_disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_disabled
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "play-button":
        return False
    elif button_id == "pause-button":
        return True
    return is_disabled


# Callback to update speed
@app.callback(
    [Output("interval-component", "interval"), Output("speed-display", "children")],
    Input("speed-slider", "value"),
)
def update_speed(value):
    return int(value * 1000), f"Speed: {value} second(s)"


if __name__ == "__main__":
    app.run_server(debug=True)
