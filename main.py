import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pickle
import matplotlib.pyplot as plt
import io
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
import webbrowser
import random
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud

# Defining Global variable
Project_Name = "Sentiment Analysis With Insights"

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

# Function to open browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050")

# Function to create app UI
# Function to create app UI
def create_app_ui():
    main_layout = html.Div(
        [
            html.H1(Project_Name, id="main_title", className="display-4 text-center mt-5 mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dcc.Textarea(
                                            id="text_review1",
                                            placeholder="Enter review 1",
                                            className="form-control mb-3"
                                        ),
                                        dbc.Button(
                                            id="Button_review1",
                                            children="Find Sentiment",
                                            color="primary"
                                        ),
                                        html.Div(id="result1", className="mt-3")
                                    ]
                                ),
                                className="shadow-sm"
                            )
                        ],
                        md=6
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dcc.Textarea(
                                            id="text_review2",
                                            placeholder="Enter reviews (separated by lines) for plotting",
                                            className="form-control mb-3"
                                        ),
                                        dbc.Button(
                                            id="Button_review2",
                                            children="Plot Reviews",
                                            color="primary"
                                        ),
                                        html.Div(id='plot-container', className="mt-3")
                                    ]
                                ),
                                className="shadow-sm"
                            )
                        ],
                        md=6
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dcc.Textarea(
                                            id="text_review3",
                                            placeholder="Search  top Words",
                                            className="form-control mb-3"
                                        ),
                                        dbc.Button(
                                            id="Button_review3",
                                            children="Plot Words",
                                            color="primary"
                                        ),
                                        html.Div(id='plot-cloud', className="mt-3")
                                    ]
                                ),
                                className="shadow-sm"
                            )
                        ],
                        md=6
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dcc.Textarea(
                                            id="text_review4",
                                            placeholder="Generate Random reviews",
                                            className="form-control mb-3"
                                        ),
                                        dbc.Button(
                                            id="Button_review4",
                                            children="Serach Random Review",
                                            color="primary"
                                        ),
                                        html.Div(id='Random-review', className="mt-3")
                                    ]
                                ),
                                className="shadow-sm"
                            )
                        ],
                        md=6
                    )
                ]
            )
        ],
        className="container"
    )
    return main_layout


# Function to check review
def check_review(reviewtext):
    with open(r"D:\data_Science\pickle_model_pkl", "rb") as file:
        recreated_model = pickle.load(file)

    with open(r"D:\data_Science\features.model_pkl", "rb") as vocab_file:
        recreated_vocab = pickle.load(vocab_file)

    recreated_vocab = TfidfVectorizer(vocabulary=recreated_vocab)
    reviewtext_vectorized = recreated_vocab.fit_transform([reviewtext])
    predict = recreated_model.predict(reviewtext_vectorized)
    return predict

# Function to plot sentiment distribution for review
def plot_reviews(reviews):
    global positive, negative
    positive = 0
    negative = 0
    for review in reviews:
        rev_check(review)

    # Plotting
    labels = ["Positive", "Negative"]
    sizes = [positive, negative]
    colors = ['#ff9999', '#66b3ff']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  
    plt.title('Sentiment Distribution')
    
    # Save plot to a byte buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Encode plot to base64 string
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    # Close plot
    plt.close()
    
    # Display plot in HTML
    return html.Img(src='data:image/png;base64,{}'.format(plot_base64))

# Function to plot word cloud for review
def plot_cloud(text_value):
    # Initialize list to store all tokens
    all_tokens = []

    # Split reviews into individual tokens and remove stop words
    reviews = text_value.split('\n')
    for review in reviews:
        tokens = remove_stop_words(review).split()
        all_tokens.extend(tokens)

    # Count frequency of words
    word_counts = Counter(all_tokens)

    # Get the most common words
    most_common_words = word_counts.most_common(10)

    # Create a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(most_common_words))

    # Convert word cloud to base64
    buffer = io.BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    buffer.seek(0)
    wordcloud_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    # Close buffer
    buffer.close()

    # Display word cloud in HTML
    return html.Img(src='data:image/png;base64,{}'.format(wordcloud_base64))

# Function to remove stop words
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

# Function to generate random reviews
def get_random_reviews(reviews, num_reviews=10):
    return random.sample(reviews, num_reviews)

# Function to count positive and negative reviews
def rev_check(reviewtext):
    global positive, negative
    with open(r"D:\data_Science\pickle_model_pkl", "rb") as file:
        recreated_model = pickle.load(file)

    with open(r"D:\data_Science\features.model_pkl", "rb") as vocab_file:
        recreated_vocab = pickle.load(vocab_file)

    recreated_vocab = TfidfVectorizer(vocabulary=recreated_vocab)
    reviewtext_vectorized = recreated_vocab.fit_transform([reviewtext])
    predict = recreated_model.predict(reviewtext_vectorized)
    if predict == 1:
        positive += 1
    else:
        negative += 1

# Callback to update sentiment result for review 1
@app.callback(
    Output("result1", "children"),
    [Input("Button_review1", "n_clicks")],
    [State("text_review1", "value")],
)
def update_ui1(n_clicks, text_value):
    if n_clicks is None:
        return None
    elif n_clicks > 0:
        response = check_review(text_value)
        if response[0] == 0:
            result1 = "Negative"
        elif response[0] == 1:
            result1 = "Positive"
        else:
            result1 = "Unknown"
        return result1

# Callback to plot sentiment distribution for review 2
@app.callback(
    Output('plot-container', 'children'),
    [Input('Button_review2', 'n_clicks')],
    [State('text_review2', 'value')]
)
def plot_reviews_callback(n_clicks, text_value):
    if n_clicks is None:
        return None
    elif n_clicks > 0:
        reviews = text_value.split('\n')
        return plot_reviews(reviews)

# Callback to plot word cloud for review 3
@app.callback(
    Output('plot-cloud', 'children'),
    [Input('Button_review3', 'n_clicks')],
    [State('text_review3', 'value')]
)
def plot_cloud_callback(n_clicks, text_value):
    if n_clicks is None:
        return None
    elif n_clicks > 0:
        return plot_cloud(text_value)

# Callback to generate random review
@app.callback(
    Output("Random-review", "children"),
    [Input("Button_review4", "n_clicks")],
    [State("text_review4", "value")],
)
def generate_random_review(n_clicks, text_value):
    if n_clicks is None:
        return None
    elif n_clicks > 0:
        reviews = text_value.split('\n')
        random_reviews = get_random_reviews(reviews, num_reviews=10)
        return html.Ul([html.Li(review) for review in random_reviews])

# Define main function
def main():
    open_browser()
    app.title = Project_Name
    app.layout = create_app_ui()
    app.run_server()

# Calling the main function
if __name__ == '__main__':
    main()
