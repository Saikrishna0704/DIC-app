
import pandas as pd
from google_play_scraper import reviews_all
from transformers import pipeline
import plotly.express as px
import streamlit as st
from google_play_scraper import search


def main():
    st.set_page_config(page_title='App Success Predictor', layout='wide', page_icon=":iphone:")
    st.title('Predicting App Success')

    # Welcome message
    st.write("Welcome to our App!")
    st.write("This app helps you explore and predict user ratings for apps available on the Google Play Store.")

    # Sidebar Image
    with st.sidebar:
        image_url = "https://zeevector.com/wp-content/uploads/2021/01/Google-Play-Store-Logo-PNG.png"

        #Application description
        st.sidebar.image(image_url, use_column_width=True)
        st.header('Project Details')
        if st.button('Data Source and Dataset Overview'):
            st.markdown("""
            **Data Source:** Google Play Store Apps dataset on Kaggle.

            **Dataset Overview:**
            The dataset comprises over 10,000 records of mobile applications. Attributes include App,
            Category, Rating, Reviews, Size, Installs, Type, Price, Content Rating, Genres, Last Updated,
            Current Version, and Android Version.

            **Data Licensing and Attribution:**
            The dataset is accessed under the Creative Commons Attribution-ShareAlike 4.0 International
            License, ensuring proper credit to dataset contributors and creators.
            """)

        if st.button('Project Teammates'):
            st.markdown("""
            **Teammates:**
            - **Chandravardhan Reddy Yanamala**
            - **Sai Krishna Tammali**
            - **Venkata gayathri satwik Mylavarapu**
            """)
        # Input field for app ID
    app_id = st.text_input("Enter the Google Play Store App ID:")

    if app_id:
        try:
            st.write("Fetching reviews for app ID:", app_id)
            reviews = reviews_all(app_id, lang='en')
            df = pd.json_normalize(reviews)
            df['content'] = df['content'].astype('str')
            sentiment_analysis = pipeline("sentiment-analysis", 'siebert/sentiment-roberta-large-english')

            # Perform sentiment analysis
            df['result'] = df['content'].apply(lambda x: sentiment_analysis(x))
            st.write("It might take quite a few minutes proportional to the count of reviews for your app id")
            st.write("Please wait!! Getting your results")
            df['sentiment'] = df['result'].apply(lambda x: (x[0]['label']))
            proportions = df['sentiment'].value_counts(normalize=True).reset_index()
            proportions.columns = ['Sentiment', 'Proportion']
            fig = px.bar(proportions, x='Proportion', y='Sentiment', orientation='h',
            color='Sentiment', labels={'Proportion': 'Proportion', 'Sentiment': 'Sentiment'},
            title='Proportion of Positive and Negative Sentiments')
            st.plotly_chart(fig)

        except Exception as e:
            st.error("Error fetching data. Please check the entered App ID or try again later.")



if __name__ == "__main__":
    main()
