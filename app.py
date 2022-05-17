#import matplotlib
#import matplotlib.pyplot as plt
#from matplotlib.font_manager import FontProperties
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import nltk
import streamlit as st
import preprocessor,helper
import json
import tweepy
from tweepy import OAuthHandler
import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import openpyxl
import time
import tqdm
import seaborn as sns
from PIL import Image
import numpy as np
from annotated_text import annotated_text
def main():
    st.sidebar.title("Choose one of the following")
    selected_box = st.sidebar.selectbox(
        'Tasks',
        ('Introduction', 'WhatsApp Chat analysis','Twitter sentiment Analysis','About','Exit')
    )
    if selected_box == 'Introduction':
        welcome()
    if selected_box == 'WhatsApp Chat analysis':
        a()
    if selected_box == 'Twitter sentiment Analysis':
        tweeter()
    if selected_box == 'About':
        about()
    if selected_box == 'Exit':
        terminate()

def welcome():

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("")
    with col2:
         #st.markdown("<marquee style='FILTER: glow(Strength=900); font-family:comic sans ms; font-size:300pt; text-color:red; direction=left ;behavior=alternate ;scrollamount=5'>CURAJ</marquee>", unsafe_allow_html=True)
         st.markdown("<marquee direction ='side' height='50' width='300' bgcolor='orange'><p style='color:red ;font-size:30px;'> Central University of Rajasthan</p></marquee>",unsafe_allow_html=True)
    with col3:
        st.write("")



    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("")

    with col2:
        image = Image.open('Logo_of_Central_University_of_Rajasthan.gif')
        st.image(image,width=290,caption='Department of Computer Science           Central University of Rajasthan, Ajmer ')
        #st.write('Department of Computer Science School of Mathematics, Statistics & Computational Sciences\nCentral University of Rajasthan, Ajmer ')
       # st.markdown("<h1 style='text-align: center; color: red;font-size: 22px;'> '</h1>", unsafe_allow_html=True)
    with col3:
        st.write("")

    st.markdown("<h1 style='text-align: center; color: white;font-size:60px;'>SOCIAL MEDIA ANALYSER</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h1 style='text-align: center; color: yellow;font-size:20px;'>Supervised By :"
                    "       Dr mamta rani ma'am</h1>", unsafe_allow_html=True)
    with col2:
        st.write("")
    with col3:
        st.markdown("<h1 style='text-align: center; color:yellow;font-size:20px'> Made By :"
                    "Gourav Rajpurohit</h1>", unsafe_allow_html=True)


    #####Fist  pageeee
    #introduction


    st.text("")
    st.text("")
    if st.button('1. introduction'):
        html_temp = """
            	<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:6px;text-align:center">MAJOR PROJECT</p></div>
            	"""
        st.markdown(html_temp, unsafe_allow_html=True)
        st.markdown("Hy My name is GAURAV RAJPUROHIT and I am making this project in fulfillment of the"  "\n" "requirements for the award of the degree of MASTER OF SCIENCE in COMPUTER SCIENCE ""\n""Under the Guidance of Dr Mamta Rani Ma'am")


        st.text("")
        st.text("")
 #      col1 , col2 , col3 = st.columns(3)
  #      with col1:
   #         st.write("")

    #    with col2:
     #       image = Image.open('methodology.png')
      #      st.image(image,width=290,caption='')

       # with col3:
        #    st.write("") */

    m = st.markdown(""" <style> div.stButton > button:first-child { background-color: rgb(255,153,153);} </style>""",
                    unsafe_allow_html=True)
    if st.button("2. About university"):
       st.markdown("The Central University of Rajasthan (CURAJ) has been established by an Act of Parliament (Act No. 25 of 2009) as a new Central University, and is fully funded by the Government of India. The President of India, His Excellency Shri Ram Nath Kovind, is the Visitor of the CURAJ. Prof. Anand Bhalerao is the Vice Chancellor of the University. CURAJ is located in Ajmer district of Rajasthan. In order to meet the challenges of the knowledge era and to keep pace with the knowledge explosion in Higher Education, the Central University of Rajasthan is committed to inculcating and sustaining quality in all the dimensions of Higher Education viz. teaching, learning, research, extension and governance while catering to the regional and global needs. CURAJ offers 50+ Masters and PhD programmes in its 29 departments. The University operates from its permanent campus of 550 (approx.) acres and it has adequate hostels housing nearly 2000 students in the campus.")
       video = open("curaj_video.mp4", "rb")
       st.video(video)

    #col1,col2,col3=st.columns(3)
    #with col1:
     #   st.write("")
    #with col2:


    #with col3:
        #st.write("")







#function of whatsapp chat analyser

def a():

    image=Image.open('WhatsApp-Chat-Sentiment-Analysis-using-Python (1).webp')
    st.image(image)
    st.sidebar.title("Whatsapp Chat Analyzer")

    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if st.sidebar.button('Help'):
        st.title("How To use ")
        col1, col2=st.columns(2)
        with col1:
            video=open("WhatsApp Video 2022-05-03 at 6.18.50 PM.mp4","rb")
            st.video(video)
        with col2:
            st.header("follow the step ")
            st.markdown("1.Open your chat with a person or a group ")
            st.markdown("2.Click on the three dots above")
            st.markdown("3.Click on more")
            st.markdown("4.Click on Expoert chat ")
            st.markdown("You will see an option to attach media while exporting your chat. For simplicity, it is best not to attach media. Finally, Save it to your device.Then click on Browse file button and upload the Text file and click on show analysis for overall or for any individual")


    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)

        # fetch unique users
        user_list = df['user'].unique().tolist()
        user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

        if st.sidebar.button("Show Analysis"):

            # Stats Area
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
            st.title("Top Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.header("Total Messages")
                st.title(num_messages)
            with col2:
                st.header("Total Words")
                st.title(words)
            with col3:
                st.header("Media Shared")
                st.title(num_media_messages)
            with col4:
                st.header("Links Shared")
                st.title(num_links)

            # monthly timeline
            st.title("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # daily timeline
            st.title("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # activity map
            st.title('Activity Map')
            col1, col2 = st.columns(2)

            with col1:
                st.header("Most busy day")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='purple')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Most busy month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            st.title("Weekly Activity Map")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

            # finding the busiest users in the group(Group level)
            if selected_user == 'Overall':
                st.title('Most Busy Users')
                x, new_df = helper.most_busy_users(df)
                fig, ax = plt.subplots()

                col1, col2 = st.columns(2)

                with col1:
                    ax.bar(x.index, x.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)

            # WordCloud
            st.title("Wordcloud")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            # most common words
            most_common_df = helper.most_common_words(selected_user, df)

            fig, ax = plt.subplots()

            ax.barh(most_common_df[0], most_common_df[1])
            plt.xticks(rotation='vertical')

            st.title('Most commmon words')
            st.pyplot(fig)

            # emoji analysis
            emoji_df = helper.emoji_helper(selected_user, df)
            st.title("Emoji Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)

            sentiment = helper.sentiment(selected_user, df)

            st.title("Sentiment")
            st.markdown(sentiment)
def end():
    st.title("Thanks for using this application ")
    st.balloons()

def tweeter():


    # To Hide Warnings
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Viz Pkgs
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')
    import seaborn as sns

    # sns.set_style('darkgrid')

    STYLE = """
    <style>
    img {
        max-width: 100%;
    }
    </style> """

    def main():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("")

        with col2:
            from PIL import Image
            image = Image.open('twitter.jpeg')
            st.image(image)

        with col3:
            st.write("")

       # """ Common ML Dataset Explorer """
        # st.title("Live twitter Sentiment analysis")
        # st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

        html_temp = """
    	<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Live twitter Sentiment analysis</p></div>
    	"""
        st.markdown(html_temp, unsafe_allow_html=True)
        st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

        ################# Twitter API Connection #######################

        consumer_key = 'oEyPq6SICQYBLbqOeEEsefRjI'
        consumer_secret = 'meNUMNIpcsrJYeMCHM7WlxSxX0qtfgTGbYx4uAUU2JPsmegFoO'
        access_token = '764141391731630080-GRZQOYuQ2JO4suPHI4OFfK2Hk1mcK7v'
        access_token_secret = 'vqnEiq0YQk3twaxNORZAVTAdMWliWA7J4CyNmkkl0pImm'

        # Use the above credentials to authenticate the API.

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        ################################################################

        df = pd.DataFrame(columns=["Date", "User", "IsVerified", "Tweet", "Likes", "RT", 'User_location'])

        # Write a Function to extract tweets:
        def get_tweets(Topic, Count):
            i = 0
            # my_bar = st.progress(100) # To track progress of Extracted tweets
            for tweet in tweepy.Cursor(api.search_tweets, q=Topic, count=100, lang="en", exclude='retweets').items():
                # time.sleep(0.1)
                # my_bar.progress(i)
                df.loc[i, "Date"] = tweet.created_at
                df.loc[i, "User"] = tweet.user.name
                df.loc[i, "IsVerified"] = tweet.user.verified
                df.loc[i, "Tweet"] = tweet.text
                df.loc[i, "Likes"] = tweet.favorite_count
                df.loc[i, "RT"] = tweet.retweet_count
                df.loc[i, "User_location"] = tweet.user.location
                df.to_csv("TweetDataset.csv",index=False)
                # df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
                i = i + 1
                if i > Count:
                    break
                else:
                    pass


        # Function to Clean the Tweet.
        def clean_tweet(tweet):
            return ' '.join(
                re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet.lower()).split())

        # Funciton to analyze Sentiment
        def analyze_sentiment(tweet):
            analysis = TextBlob(tweet)
            if analysis.sentiment.polarity > 0:
                return 'Positive'
            elif analysis.sentiment.polarity == 0:
                return 'Neutral'
            else:
                return 'Negative'

        # Function to Pre-process data for Worlcloud
        def prepCloud(Topic_text, Topic):
            Topic = str(Topic).lower()
            Topic = ' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
            Topic = re.split("\s+", str(Topic))
            stopwords = set(STOPWORDS)
            stopwords.update(Topic)  ### Add our topic in Stopwords, so it doesnt appear in wordClous
            ###
            text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
            return text_new

        #
        from PIL import Image
        #image = Image.open('Logo1.jpg')
        #st.image(image, caption='Twitter for Analytics', use_column_width=True)

        # Collect Input from user :
        Topic = str()
        Topic = str(st.text_input("Enter the topic you are interested in (Press Enter once done)"))

        if len(Topic) > 0:

            # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
            with st.spinner("Please wait, Tweets are being extracted"):
                get_tweets(Topic, Count=499)
            st.success('Tweets have been Extracted !!!!')

            # Call function to get Clean tweets
            df['clean_tweet'] = df['Tweet'].apply(lambda x: clean_tweet(x))

            # Call function to get the Sentiments
            df["Sentiment"] = df["Tweet"].apply(lambda x: analyze_sentiment(x))

            # Write Summary of the Tweets
            st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic, len(df.Tweet)))
            st.write("Total Positive Tweets are : {}".format(len(df[df["Sentiment"] == "Positive"])))
            st.write("Total Negative Tweets are : {}".format(len(df[df["Sentiment"] == "Negative"])))
            st.write("Total Neutral Tweets are : {}".format(len(df[df["Sentiment"] == "Neutral"])))

            # See the Extracted Data :
            if st.button("See the Extracted Data"):
                # st.markdown(html_temp, unsafe_allow_html=True)
                st.success("Below is the Extracted Data :")
                df = pd.read_csv("TweetDataset.csv")

                @st.cache
                def convert_df(df):
                    return df.to_csv().encode('utf-8')

                csv = convert_df(df)
                st.dataframe(df)
                col1,col2,col3=st.columns(3)
                with col1:
                    st.write("")
                with col2:
                    st.download_button( "Press to Download",csv,"file.csv","text/csv",key='download-csv')
                    st.write("")
                    st.write("")
                    st.write("")
                with col3:
                    st.write("")
                    st.write("")
                    st.write("")

                #if st.button("map"):
                 #   f=pd.read_csv('TweetDataset.csv')
                  #  pf = pd.DataFrame(f.loc[:, "User_location"], columns=['lat', 'lon'])
                   # st.map(pf)

            # get the countPlot
            if st.button("Get Count Plot for Different Sentiments"):
                st.success("Generating A Count Plot")
                st.subheader(" Count Plot for Different Sentiments")
                st.write(sns.countplot(df["Sentiment"]))
                st.pyplot()

            # Piechart
            if st.button("Get Pie Chart for Different Sentiments"):
                st.success("Generating A Pie Chart")
                a = len(df[df["Sentiment"] == "Positive"])
                b = len(df[df["Sentiment"] == "Negative"])
                c = len(df[df["Sentiment"] == "Neutral"])
                d = np.array([a, b, c])
                explode = (0.1, 0.0, 0.1)
                st.write(
                    plt.pie(d, shadow=True, explode=explode, labels=["Positive", "Negative", "Neutral"],
                            autopct='%1.2f%%'))
                st.pyplot()

            # get the countPlot Based on Verified and unverified Users
            if st.button("Get Count Plot Based on Verified and unverified Users"):
                st.success("Generating A Count Plot (Verified and unverified Users)")
                st.subheader(" Count Plot for Different Sentiments for Verified and unverified Users")
                st.write(sns.countplot(df["Sentiment"], hue=df.IsVerified))
                st.pyplot()

            ## Points to add 1. Make Backgroud Clear for Wordcloud 2. Remove keywords from Wordcloud

            # Create a Worlcloud
            if st.button("Get WordCloud for all things said about {}".format(Topic)):
                st.success("Generating A WordCloud for all things said about {}".format(Topic))
                text = " ".join(review for review in df.clean_tweet)
                stopwords = set(STOPWORDS)
                text_newALL = prepCloud(text, Topic)
                wordcloud = WordCloud(stopwords=stopwords, max_words=800, max_font_size=70).generate(text_newALL)
                st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                st.pyplot()

            # Wordcloud for Positive tweets only
            if st.button("Get WordCloud for all Positive Tweets about {}".format(Topic)):
                st.success("Generating A WordCloud for all Positive Tweets about {}".format(Topic))
                text_positive = " ".join(review for review in df[df["Sentiment"] == "Positive"].clean_tweet)
                stopwords = set(STOPWORDS)
                text_new_positive = prepCloud(text_positive, Topic)
                # text_positive=" ".join([word for word in text_positive.split() if word not in stopwords])
                wordcloud = WordCloud(stopwords=stopwords, max_words=800, max_font_size=70).generate(text_new_positive)
                st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                st.pyplot()

            # Wordcloud for Negative tweets only
            if st.button("Get WordCloud for all Negative Tweets about {}".format(Topic)):
                st.success("Generating A WordCloud for all Positive Tweets about {}".format(Topic))
                text_negative = " ".join(review for review in df[df["Sentiment"] == "Negative"].clean_tweet)
                stopwords = set(STOPWORDS)
                text_new_negative = prepCloud(text_negative, Topic)
                # text_negative=" ".join([word for word in text_negative.split() if word not in stopwords])
                wordcloud = WordCloud(stopwords=stopwords, max_words=800, max_font_size=70).generate(text_new_negative)
                st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                st.pyplot()

        st.sidebar.header("About App")
        st.sidebar.info("A Twitter Sentiment analysis Project which will scrap twitter for the topic selected by the user. The extracted tweets will then be used to determine the Sentiments of those tweets. \
                        The different Visualizations will help us get a feel of the overall mood of the people on Twitter regarding the topic we select.")



        # st.sidebar.subheader("Scatter-plot setup")
        # box1 = st.sidebar.selectbox(label= "X axis", options = numeric_columns)
        # box2 = st.sidebar.selectbox(label="Y axis", options=numeric_columns)
        # sns.jointplot(x=box1, y= box2, data=df, kind = "reg", color= "red")
        # st.pyplot()

        if st.button("Exit"):
            st.balloons()
            end()

    if __name__ == '__main__':
        main()

def about():
    col1, col2, col3 = st.columns(3)

    with col1:
        image = Image.open('WhatsApp-Chat-Sentiment-Analysis-using-Python (1).webp')
        st.image(image)

    with col2:
        st.write("")
    with col3:
        image = Image.open('twitter.jpeg')
        st.image(image)
    html_temp = """
       	<div style="background-color:tomato;"><p style="color:white;font-size:20px;padding:9px">In this application I am using  two social media platform the first one is whatsapp and second one is Twitter for analyse of data. This Project  is focused on data processing and analysis. The first step in putting a machine learning algorithm into action is to figure out what kind of learning experience the model should be based on. I opted to work on Twitter and Whatsapp because, in comparison to traditional online articles and web blogs, we believe twitter provides a better representation of popular sentiment. The reason for this is because Twitter has a far higher amount of relevant data than traditional blogging sites. Furthermore, the answer on Twitter is both faster and more general (due to the fact that the number of users who tweet is significantly higher than the number of users who post daily web blogs). It is a rapidly growing site with over 200 million registered users, 100 million of whom are active users, and half of them log on daily, resulting in almost 250 million tweets every day. Due to this large amount of usage we hope to achieve a reflection of public sentiment by analyzing the sentiments expressed in the tweets which can be used in my app by this I can add one more statical data i.e. Sentiment analyse of Tweet that is classifying tweets according to the sentiment expressed in them: positive, negative or neutral. Sentiment analyse is important Many applications require analysing public mood, including corporations attempting to determine the market response to their products, political election forecasting, and macroeconomic phenomena such as stock exchange forecasting.And the reason behind working on whatsapp is When it comes to machine learning, data pre-processing is crucial. We needed a lot of data to make the model more efficient, so we focused on WhatsApp, one of the largest data generators owned by Facebook . Every day, WhatsApp claims to send approximately 55 billion messages. The average WhatsApp user spends 195 minutes each week on the app and belongs to a variety of groups. So we can analyse many things on whatsapp.I deploy this project work as a fully functional web app in which we can upload our data and we can analyse our data with different different types of graphs.The ability to extract information and insights from any form of data is considered  a valuable skill.Data can be in various forms like structured tabular data, text data, etc. The ability to extract information and insights from any form of data is considered  a valuable skill.Data can be in various forms like structured tabular data, text data, etc.WhatsApp is a type of social media platform that allows it users to send messages, pictures, videos etc to each other.Sending these messages generate text data which can be extracted easily and with knowledge of a programming language like python, can be transformed and analyzed.</p></div>
       	"""
    st.markdown(html_temp, unsafe_allow_html=True)
#    st.write('In this application I am using  two social media platform the first one is whatsapp and second one is Twitter for analyse of data. This Project  is focused on data processing and analysis. The first step in putting a machine learning algorithm into action is to figure out what kind of learning experience the model should be based on. I opted to work on Twitter and Whatsapp because, in comparison to traditional online articles and web blogs, we believe twitter provides a better representation of popular sentiment. The reason for this is because Twitter has a far higher amount of relevant data than traditional blogging sites. Furthermore, the answer on Twitter is both faster and more general (due to the fact that the number of users who tweet is significantly higher than the number of users who post daily web blogs). It is a rapidly growing site with over 200 million registered users, 100 million of whom are active users, and half of them log on daily, resulting in almost 250 million tweets every day. Due to this large amount of usage we hope to achieve a reflection of public sentiment by analyzing the sentiments expressed in the tweets which can be used in my app by this I can add one more statical data i.e. Sentiment analyse of Tweet that is classifying tweets according to the sentiment expressed in them: positive, negative or neutral. Sentiment analyse is important Many applications require analysing public mood, including corporations attempting to determine the market response to their products, political election forecasting, and macroeconomic phenomena such as stock exchange forecasting.And the reason behind working on whatsapp is When it comes to machine learning, data pre-processing is crucial. We needed a lot of data to make the model more efficient, so we focused on WhatsApp, one of the largest data generators owned by Facebook . Every day, WhatsApp claims to send approximately 55 billion messages. The average WhatsApp user spends 195 minutes each week on the app and belongs to a variety of groups. So we can analyse many things on whatsapp.I deploy this project work as a fully functional web app in which we can upload our data and we can analyse our data with different different types of graphs.')
def terminate():
    html_temp = """
                	<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:6px;text-align:center">Thanks for using the application</p></div>
                	"""
    st.markdown(html_temp, unsafe_allow_html=True)

    #st.title("Thanks for using the application")
    st.balloons()
    #import time
    #import SessionState

main()
#hide_streamlit_style = """
 #           <style>
  #          #MainMenu {visibility: hidden;}
   #         footer {visibility: hidden;}
    #        </style>
 #           """
#st.markdown(hide_streamlit_style, unsafe_allow_html=True)
#######################################






