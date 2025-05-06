import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def create_dashboard():
    try:
        # Load performance data
        df = pd.read_csv('model_performance.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create plots
        fig1 = px.line(
            df, 
            x='timestamp', 
            y='r2_score', 
            color='ticker',
            title='Model Performance Over Time'
        )
        
        fig2 = px.box(
            df, 
            x='ticker', 
            y='r2_score',
            title='Performance Distribution by Ticker'
        )
        
        return html.Div([
            html.H1("Stock Prediction Model Dashboard"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig1), width=6),
                dbc.Col(dcc.Graph(figure=fig2), width=6)
            ]),
            html.Div([
                html.H3("Latest Metrics"),
                dbc.Table.from_dataframe(df.tail(10), striped=True, bordered=True)
            ])
        ])
    except FileNotFoundError:
        return html.Div("No performance data available yet")
    except Exception as e:
        return html.Div(f"Error loading dashboard: {str(e)}")

app.layout = create_dashboard

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)