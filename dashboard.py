import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import psycopg2
import pandas as pd
import os
import requests
import nltk
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load trained models
xgb_model = joblib.load('xgb_model_final.pkl')
rf_model = joblib.load('rf_model_final.pkl')

# Download VADER lexicon
nltk.download('vader_lexicon')
# PostgreSQL connection information (use environment variables for security)
host = os.getenv('POSTGRES_HOST', 'ie7945.postgres.database.azure.com')
database = os.getenv('POSTGRES_DB', 'ie7945')
user = os.getenv('POSTGRES_USER', 'ie7945')
password = os.getenv('POSTGRES_PASSWORD', 'AgKpAmRePTUUZ9j')

# NewsData.io API key
news_api_key = "pub_5977249fc3fd2d2560e19f5f9e914744239fb"

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fetch data from PostgreSQL database
def fetch_data():
    try:
        conn = psycopg2.connect(host=host, database=database, user=user, password=password)
        query = '''
            SELECT symbol, price, volume, exchange, timestamp
            FROM crypto_table
            ORDER BY timestamp ASC;
        '''
        data = pd.read_sql(query, conn)
        conn.close()
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()






# Fetch live cryptocurrency news
def fetch_crypto_news():
    try:
        url = f"https://newsdata.io/api/1/news?apikey={news_api_key}&q=cryptocurrency&language=en"
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        articles = news_data.get('results', [])
        return articles[:5]  # Return top 5 articles
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Use VADER sentiment analysis to get a sentiment score
def get_sentiment_score(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

# Resample data based on the selected timeframe
def resample_data(df, timeframe):
    df.set_index('timestamp', inplace=True)
    if timeframe == '1 minute':
        return df.resample('T').last()
    elif timeframe == '5 minutes':
        return df.resample('5T').last()
    elif timeframe == '15 minutes':
        return df.resample('15T').last()
    return df

#Feature engineering
def preprocess_data(df):
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['price_change'] = df['price'].diff().fillna(0)
    df['rolling_mean_5'] = df['price'].rolling(window=5).mean().fillna(0)
    df['rolling_std_5'] = df['price'].rolling(window=5).std().fillna(0)
    features = ['price', 'volume', 'hour', 'minute', 'day_of_week', 'price_change', 'rolling_mean_5', 'rolling_std_5']
    return df[features]

# Standardize data
def standardize_data(df, scaler):
    scaled_features = scaler.transform(df)
    return scaled_features

# Initialize scaler and fit it on a sample dataset
data_sample = fetch_data()
data_sample = resample_data(data_sample, '1 minute')
data_sample.dropna(subset=['price'], inplace=True)
data_sample = preprocess_data(data_sample)
scaler = StandardScaler()
scaler.fit(data_sample)

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

app.title = "Cryptocurrency Dashboard"

app.layout = html.Div([

    # Main Header
    html.H1(
        "Cryptocurrency Data Dashboard",
        style={
            'textAlign': 'center',
            'color': '#ffffff',
            'backgroundColor': '#1f1f1f',
            'padding': '20px',
            'borderRadius': '10px',
            'marginBottom': '20px',
            'fontSize': '32px',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
        }
    ),

    # Main Content Container
    html.Div([
        # Left Section: Graphs
        html.Div([
            # Timeframe Dropdown
            html.Div([
                dcc.Dropdown(
                    id='timeframe-dropdown',
                    options=[
                        {'label': '1 Minute', 'value': '1 minute'},
                        {'label': '5 Minutes', 'value': '5 minutes'},
                        {'label': '15 Minutes', 'value': '15 minutes'},
                    ],
                    value='1 minute',
                    style={
                        'width': '80%',
                        'margin': '0 auto',
                        'padding': '10px',
                        'fontSize': '18px',
                        'border': '1px solid #ccc',
                        'borderRadius': '8px',
                        'backgroundColor': '#ffffff',
                        'color': '#333',
                        'cursor': 'pointer',
                    },
                    className='custom-dropdown',
                ),
            ], style={'marginBottom': '20px', 'textAlign': 'center'}),

            # Graphs
            dcc.Graph(id='price-chart', style={'marginBottom': '20px'}),
            dcc.Graph(id='volume-chart', style={'marginBottom': '20px'}),
            dcc.Graph(id='candlestick-chart',responsive=True, style={'marginBottom': '20px','marginTop':'20px'}),
            dcc.Graph(id='prediction-graph'),
        ], style={
            'width': '60%',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '10px',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            'marginRight': '10px'
        }),

        # Right Section: News and Sentiment
        html.Div([
            # Live Cryptocurrency News
            html.Div([
                html.H3(
                    "Live Cryptocurrency News",
                    style={
                        'textAlign': 'center',
                        'color': '#ffffff',
                        'backgroundColor': '#444444',
                        'padding': '10px',
                        'borderRadius': '5px',
                        'marginBottom': '10px'
                    }
                ),
                html.Ul(
                    id='news-feed',
                    style={
                        'padding': '15px',
                        'fontSize': '16px',
                        'backgroundColor': '#ffffff',
                        'borderRadius': '5px',
                        'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                        'maxHeight': '500px',
                        'overflowY': 'auto',
                        'listStyleType': 'none'
                    }
                ),
            ], style={
                'marginBottom': '20px',
                'padding': '20px',
                'backgroundColor': '#f3f3f3',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
            }),

            # Sentiment Analysis Gauge
            html.Div([
                html.H3(
                    "Sentiment Analysis",
                    style={
                        'textAlign': 'center',
                        'color': '#ffffff',
                        'backgroundColor': '#444444',
                        'padding': '10px',
                        'borderRadius': '5px',
                        'marginBottom': '10px'
                    }
                ),
                dcc.Graph(id='sentiment-gauge'),
            ], style={
                'marginBottom': '20px',
                'padding': '20px',
                'backgroundColor': '#f3f3f3',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
            }),

            # Latest Bitcoin Price
            html.Div([
                html.H3("Latest Bitcoin Price", style={'textAlign': 'center', 'color': 'black'}),
                html.Div(
                    id='live-price',
                    children="Fetching live price...",
                    style={
                        'fontSize': '24px',
                        'fontWeight': 'bold',
                        'color': '#333',
                        'padding': '15px',
                        'textAlign': 'center',
                        'border': '1px solid #ddd',
                        'borderRadius': '10px',
                        'backgroundColor': '#f9f9f9',
                    }
                ),
            ], style={
                'marginBottom': '20px',
                'padding': '20px',
                'backgroundColor': '#f3f3f3',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
            }),

            html.Div([
                html.H3("Search Cryptocurrency News", style={'textAlign': 'center'}),
                dcc.Input(
                    id='search-input',
                    type='text',
                    placeholder="Search...",
                    style={'width': '100%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}
                ),
            ], style={
                'marginBottom': '20px',
                'padding': '20px',
                'backgroundColor': '#f3f3f3',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
            }),

            # User Feedback
            html.Div([
                html.H3(
                    "User Feedback",
                    style={
                        'textAlign': 'center',
                        'color': '#ffffff',
                        'backgroundColor': '#444444',
                        'padding': '10px',
                        'borderRadius': '5px',
                        'marginBottom': '10px'
                    }),
                dcc.Textarea(
                    id='user-feedback',
                    placeholder="Share your feedback...",
                    style={'width': '100%', 'height': '100px', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}
                ),
                html.Button(
                    'Submit Feedback',
                    id='submit-feedback',
                    n_clicks=0,
                    style={
                        'padding': '10px',
                        'borderRadius': '5px',
                        'backgroundColor': '#007BFF',
                        'color': 'white',
                        'fontSize': '16px',
                        'border': 'none',
                        'cursor': 'pointer',
                    }
                ),
            ], style={
                'marginBottom': '20px',
                'padding': '20px',
                'backgroundColor': '#f3f3f3',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
            }),

            html.Div(
                id='accuracy-metrics',
                style={
                    'width': '100%',
                    'padding': '20px',
                    'backgroundColor': '#f3f3f3',
                    'borderRadius': '10px',
                    'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',
                    'marginTop': '20px',
                    'textAlign': 'center'
                }),

        ], style={
            'width': '35%',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '10px',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
        }),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),

    # Interval component for live updates
    dcc.Interval(
        id='interval-component',
        interval=60 * 1000,  # Refresh every 60 seconds
        n_intervals=0  # Starts at 0
    )

])

# Callback to update the price chart
@app.callback(
    Output('price-chart', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('timeframe-dropdown', 'value')  # Added input for timeframe
)
def update_price_chart(n_intervals, timeframe):
    df = fetch_data()
    if df.empty:
        return {
            'data': [],
            'layout': go.Layout(
                title="Price Over Time",
                xaxis_title="Timestamp",
                yaxis_title="Price (USD)",
                template='plotly_dark'
            )
        }

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')  # Ensure timestamp is in datetime format
    df['date'] = df['timestamp'].dt.date  # Separate the date
    df['time'] = df['timestamp'].dt.time  # Separate the time
    df_resampled = resample_data(df, timeframe)  # Resample data based on selected timeframe
    grouped = df_resampled.groupby('exchange')
    traces = []
    for exchange, data in grouped:
        traces.append(go.Scatter(
            x=data['time'],  # Display only time on the x-axis
            y=data['price'],
            mode='lines',
            name=f"{exchange} Price"
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            title="Price Over Time",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            template='plotly_dark',
            xaxis=dict(
                tickformat='%M:%S',  # Format for time display
                showgrid=True,  # Show gridlines
                rangeslider=dict(visible=True),
                nticks=15,
                tickmode='auto'  # Optional: Adds a range slider for zooming
            )
        )
    }

# Callback to update the volume chart
@app.callback(
    Output('volume-chart', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('timeframe-dropdown', 'value')  # Added input for timeframe
)
def update_volume_chart(n_intervals, timeframe):
    df = fetch_data()
    if df.empty:
        return {
            'data': [],
            'layout': go.Layout(
                title="Volume Over Time",
                xaxis_title="Timestamp",
                yaxis_title="Volume",
                template='plotly_dark'
            )
        }

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')  # Ensure timestamp is in datetime format
    df['date'] = df['timestamp'].dt.date  # Separate the date
    df['time'] = df['timestamp'].dt.time  # Separate the time
    df_resampled = resample_data(df, timeframe)  # Resample data based on selected timeframe
    grouped = df_resampled.groupby('exchange')
    traces = []
    for exchange, data in grouped:
        traces.append(go.Scatter(
            x=data['time'],  # Display only time on the x-axis
            y=data['volume'],
            mode='lines',
            name=f"{exchange} Volume"
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            title="Volume Over Time",
            xaxis_title="Time",
            yaxis_title="Volume",
            template='plotly_dark',
            xaxis=dict(
                tickformat='%H:%M:%S',  # Format for time display
                showgrid=True,  # Show gridlines
                rangeslider=dict(visible=True),
                nticks=15,
                tickmode='auto'  # Optional: Adds a range slider for zooming
            )
        )
    }


@app.callback(
    Output('candlestick-chart', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('timeframe-dropdown', 'value')  # Added input for timeframe
)
def update_candlestick_chart(n_intervals, timeframe):
    df = fetch_data()
    if df.empty:
        return {
            'data': [],
            'layout': go.Layout(
                title="Candlestick Chart",
                xaxis_title="Timestamp",
                yaxis_title="Price (USD)",
                template='plotly_dark'
            )
        }

    # Ensure 'timestamp' column is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')  # Sort by timestamp

    # Resample data to 1-minute intervals using the 'timestamp' column
    df_resampled = df.resample('1T', on='timestamp').agg(
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last')
    ).dropna().reset_index()  # Drop rows with NaN values (if any)

    # Limit the number of rows for a more manageable chart (e.g., last 60 minutes)
    df_resampled = df_resampled.tail(60)  # Only show the last 60 minutes

    # Create the candlestick chart
    candlestick = go.Candlestick(
        x=df_resampled['timestamp'],  # Use timestamp as the x-axis
        open=df_resampled['open'],
        high=df_resampled['high'],
        low=df_resampled['low'],
        close=df_resampled['close'],
        name='Price'
    )

    return {
        'data': [candlestick],
        'layout': go.Layout(
            title="1-Minute Candlestick Chart",
            xaxis_title="Time (minutes)",
            yaxis_title="Price (USD)",
            template='plotly_dark',
            xaxis=dict(
                tickformat='%H:%M',  # Format for displaying minutes on the x-axis
                showgrid=True,  # Show gridlines
                rangeslider=dict(visible=True),  # Add a range slider for zooming
                nticks=10,  # Reduce the number of ticks to avoid congestion
                tickmode='auto',  # Automatically adjust the tick placement
                tickangle=45,  # Rotate x-axis labels to prevent overlap
                showline=True,  # Display the x-axis line
                zeroline=False  # Hide the zero line
            ),
            margin=dict(l=50, r=50, t=150, b=100),  # Adjust margins to make space for labels
            height=600,  # Increase the height of the chart
            width=1125,  # Increase the width of the chart
            xaxis_rangeslider_visible=True,  # Show the range slider at the bottom
            autosize=True
        )
    }



@app.callback(
    [Output('prediction-graph', 'figure'),
     Output('accuracy-metrics', 'children')],
    [Input('timeframe-dropdown', 'value')]
)
def update_graph(selected_timeframe):
    # Fetch and preprocess data
    data = fetch_data()
    if data.empty:
        return go.Figure(), "No data available"

    data = resample_data(data, selected_timeframe)

    # Drop rows with missing price values early
    data.dropna(subset=['price'], inplace=True)
    if data.empty:
        return go.Figure(), "Insufficient data after cleaning"

    # Preprocess features
    data_features = preprocess_data(data)

    # Handle NaN values in features (fill or drop as needed)
    data_features.dropna(inplace=True)

    # Align data with cleaned features
    data = data.loc[data_features.index]

    # Standardize features
    standardized_data = standardize_data(data_features, scaler)

    # Make predictions
    xgb_predictions = xgb_model.predict(standardized_data)
    rf_predictions = rf_model.predict(standardized_data)

    # Ensure predictions are aligned with data indices
    data['XGBoost Predictions'] = pd.Series(xgb_predictions, index=data.index)
    data['Random Forest Predictions'] = pd.Series(rf_predictions, index=data.index)

    # Calculate metrics
    mse_xgb = mean_squared_error(data['price'], data['XGBoost Predictions'])
    mae_xgb = mean_absolute_error(data['price'], data['XGBoost Predictions'])
    r2_xgb = r2_score(data['price'], data['XGBoost Predictions'])

    mse_rf = mean_squared_error(data['price'], data['Random Forest Predictions'])
    mae_rf = mean_absolute_error(data['price'], data['Random Forest Predictions'])
    r2_rf = r2_score(data['price'], data['Random Forest Predictions'])

    metrics = html.Div([
        html.H4("Model Accuracy Metrics"),
        html.P(f"XGBoost - MSE: {mse_xgb:.2f}, MAE: {mae_xgb:.2f}, R2: {r2_xgb:.2f}"),
        html.P(f"Random Forest - MSE: {mse_rf:.2f}, MAE: {mae_rf:.2f}, R2: {r2_rf:.2f}")
    ])

    # Create graph
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data.index, y=data['price'], mode='lines+markers', name='Actual Price'))
    figure.add_trace(go.Scatter(x=data.index, y=data['XGBoost Predictions'], mode='lines+markers', name='XGBoost Predictions'))
    figure.add_trace(go.Scatter(x=data.index, y=data['Random Forest Predictions'], mode='lines+markers', name='Random Forest Predictions'))

    figure.update_layout(title="Cryptocurrency Price Predictions",
                         xaxis_title="Timestamp (Date)",
                         yaxis_title="Price",
                         template='plotly_dark',
                         legend_title="Legend",
                         xaxis=dict(showgrid=True, tickformat="%Y-%m-%d"))

    return figure, metrics




# Callback to update live news feed
@app.callback(
    Output('news-feed', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_news_feed(n_intervals):
    articles = fetch_crypto_news()
    if not articles:
        return [html.Li("No news available at the moment.")]
 
    news_items = []
    for article in articles:
        sentiment_score = get_sentiment_score(article['title'])
        sentiment_color = 'green' if sentiment_score > 0.1 else 'red' if sentiment_score < -0.1 else 'orange'
        news_items.append(html.Li(
            f"{article['title']} - Sentiment: {sentiment_score:.2f}",
            style={'color': sentiment_color, 'marginBottom': '10px'}
        ))
 
    return news_items

# Callback to update sentiment gauge
@app.callback(
    Output('sentiment-gauge', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_sentiment_score(n_intervals):
    sentiment_score = 0.5  # Default sentiment score
 
    articles = fetch_crypto_news()
    if articles:
        sentiment_scores = [get_sentiment_score(article['title']) for article in articles]
        sentiment_score = sum(sentiment_scores) / len(sentiment_scores)  # Average sentiment score
 
    return {
        'data': [
            go.Indicator(
                mode="gauge+number",
                value=sentiment_score,
                title={'text': "Sentiment Score"},
                gauge={
                    'shape': 'angular',
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 0.25], 'color': 'red'},
                        {'range': [0.25, 0.75], 'color': 'orange'},
                        {'range': [0.75, 1], 'color': 'green'}
                    ],
                }
            )
        ],
        'layout': {
            'template': 'plotly_dark',
        }
    }
 


def fetch_latest_bitcoin_price():
    try:
        # Database connection details
        conn = psycopg2.connect(
            host="ie7945.postgres.database.azure.com",
            database="ie7945",
            user="ie7945",
            password="AgKpAmRePTUUZ9j"  # Ensure this is securely handled in production
        )
        query = """
        SELECT price, timestamp
        FROM crypto_table
        WHERE symbol = 'BTC'
        ORDER BY timestamp DESC
        LIMIT 1;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        if not df.empty:
            price = df['price'].iloc[0]
            timestamp = df['timestamp'].iloc[0]
            return f"${price:.2f} (as of {timestamp})"
        else:
            return "No data available"
    except Exception as e:
        return f"Error fetching price: {e}"
@app.callback(
    Output('live-price', 'children'),  # Updates the `children` of `live-price`
    Input('interval-component', 'n_intervals')  # Triggers on interval
)
def update_live_price(n_intervals):
    return fetch_latest_bitcoin_price()



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=int(os.environ.get("PORT", 8080)))

