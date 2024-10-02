import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Load the housing data
filepath = r"C:\Users\jessica.nguyen\Downloads\archive\housing.csv"
df = pd.read_csv(filepath)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("House Price Dashboard"),

    # Dropdown for selecting the x-axis
    html.Label("Select X-axis:"),
    dcc.Dropdown(
        id='x-axis-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns],
        value='size'  # Default value
    ),

    # Dropdown for selecting the y-axis
    html.Label("Select Y-axis:"),
    dcc.Dropdown(
        id='y-axis-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns],
        value='price'  # Default value
    ),

    # Graph to display the scatter plot
    dcc.Graph(id='scatter-plot'),

    # Slider for selecting a range of prices
    dcc.RangeSlider(
        id='price-slider',
        min=df['price'].min(),
        max=df['price'].max(),
        value=[df['price'].min(), df['price'].max()],
        marks={int(price): str(price) for price in range(int(df['price'].min()), int(df['price'].max()), 50000)},
        step=1000
    )
])


# Define the callback to update the graph
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('price-slider', 'value')]
)
def update_graph(selected_x, selected_y, price_range):
    # Filter data based on price range
    filtered_df = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])]

    # Create scatter plot
    fig = px.scatter(filtered_df, x=selected_x, y=selected_y, color='location',
                     title=f'{selected_y} vs {selected_x}', labels={selected_x: selected_x, selected_y: selected_y})

    return fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
