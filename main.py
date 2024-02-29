import re
import string
import threading
import urllib
from pathlib import Path
import time
import os

import nltk
import numpy as np
import pandas as pd
import praw
import pytesseract
import torch
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
nltk.download('all')
from threading import Thread
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from PIL import Image
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

tokenizer = AutoTokenizer.from_pretrained("StephanAkkerman/FinTwitBERT-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("StephanAkkerman/FinTwitBERT-sentiment")

# Read-only instance for subreddit
reddit_read_only = praw.Reddit(client_id="baW0zy5tuaCY9-BglOd1mg",         # your client id
                               client_secret="_W7HlrsUFwCHv9k7GAI0kfRju774LQ",      # your client secret
                               user_agent="heatY_12")        # your user agent

subreddit = reddit_read_only.subreddit("wallstreetbets")

print("Display Name: ", subreddit.display_name)
print("Description: ", subreddit.description)
print()

#initialize sentiment analysis tool from NLTK
sia = SentimentIntensityAnalyzer()

#Map to store posts
text_posts = {}
image_posts = {}

#Number of posts to pull
scraped_posts = 10


#Filter keywords

buy_keywords = [
    "bullish",
    "positive outlook",
    "strong fundamentals",
    "growth potential",
    "make money",
    "good investment",
    "undervalued",
    "attractive valuation",
    "positive momentum",
    "favorable market conditions",
    "strong earnings",
    "positive catalysts"
]

sell_keywords = [
    "bearish",
    "negative outlook",
    "weak fundamentals",
    "overvalued",
    "lose",
    "drop value",
    "loss",
    "negative",
    "unattractive valuation",
    "negative momentum",
    "unfavorable market conditions",
    "weak earnings",
    "negative catalysts",
    "deteriorating financials"
]

image_keywords = [
    ".jpg",
    ".png",
    ".jpeg",
    ".webp"
]

#Helper function to preprocess the text
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'^\d+','',text)
        text = re.sub(r'[^\w\s]','',text)
        tokens = nltk.word_tokenize(text)
        return tokens
    except:
        print("Error preprocessing text")

#Helper function to remove stopwords
def remove_stopwords(tokens):
    try:
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return filtered_tokens
    except: 
        print("Error removing stopwords")

#Helper function to lemmatize words
def lemmatize_words(tokens):
    try:
        lemmatizer = nltk.WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens
    except:
        print("Error lemmatizing words")

#Helper function to clean text
def clean_text(text):
    try:
        print("Cleaning text")
        tokens = preprocess_text(text)
        filtered_tokens = remove_stopwords(tokens)
        lemmatize_tokens = lemmatize_words(filtered_tokens)
        clean_text = ' '.join(lemmatize_tokens)
        print("Text cleaned, returning cleaned text")
        return clean_text
    except:
        print("Failure in clean text")

#Helper function to check if a word from keywords exists in the post
def contains_word(keywordList: list, text: str):
    return any(keyword in text for keyword in keywordList)

#Helper function to process text from an image with the url
def img_to_txt(url: str):
    try:
        urllib.request.urlretrieve(url, 'image')
        # Open the image file
        image = Image.open('image')
        # Perform OCR using PyTesseract
        text = pytesseract.image_to_string(image)
        # Print the extracted text
        return text
    except:
        print("FAILURE WITHIN IMAGETOTEXT HELPER FUNCTION!")

for post in subreddit.hot(limit=scraped_posts):

    #NOT NEEDED YET
    #if post.title in image_posts.keys():
        #print("title already exists")

    #FILTER BY FLAIR
    #if post.link_flair_text.strip() == "Discussion":

    if len(post.selftext) == 0 and hasattr(post, 'url') and post.url and contains_word(image_keywords, post.url):
        extracted_text = img_to_txt(str(post.url))
        if contains_word(buy_keywords, extracted_text) or contains_word(sell_keywords, extracted_text):
            try:
                image_posts[post.title] = "THIS IS FROM A PROCESSED IMAGE: " + post.url + "\n" + clean_text(extracted_text)
            except:
                print("Could not process image from url: " + post.url)

    elif len(post.selftext) == 0:
        continue

    else:
        if contains_word(buy_keywords, post.selftext) or contains_word(sell_keywords, post.selftext):
            try:
                text_posts[post.title] = clean_text(post.selftext)
            except:
                print("Could not add text post: " + post)

sentiment = 0.000

print("-----------------------------------------------------")
print("LOOPING THROUGH TEXT POSTS")
print("-----------------------------------------------------")

for index, value in text_posts.items():
    print("Title: ", index)
    print("-----------------------------------------------------")
    print("Content: ", value)
    score = sia.polarity_scores(value)
    print(score)
    sentiment += score["compound"]
    print()

print("-----------------------------------------------------")
print("LOOPING THROUGH IMAGE POSTS")
print("-----------------------------------------------------")

for index, value in image_posts.items():
    print("Title: ", index)
    print("-----------------------------------------------------")
    print("Content: ", value)
    score = sia.polarity_scores(value)
    print(score)
    sentiment += score["compound"]
    print()

print("-----------------------------------------------------")
print("COLLECTED DATA SUMMARY")
print()
print("Total posts: " + str(len(text_posts) + len(image_posts)) + "/" + str(scraped_posts))
print()
print("Amount of text posts: " + str(len(text_posts)))
print()
print("Amount of image posts: " + str(len(image_posts)))
print()
print("Final sentiment score: " + str(sentiment))
print()

print("-----------------------------------------------------")
print("Switching to GPU")

gc.collect()

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Entering bot loop")

start = time.time()

# Define the prompt
prompt = "What does 2+2 equal?"

# Tokenize the input
input_ids = tokenizer.encode(prompt, return_tensors="pt")
input_ids = input_ids.to(device)

attention_mask = torch.ones_like(input_ids)
attention_mask = attention_mask.to(device)

print("Prompt acquired, asking the bot")

# Generate a response
output = model.generate(input_ids, attention_mask=attention_mask, max_length=1, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)
output = output.to("cpu")
print("Decoding output")

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

end = time.time()
print(end - start)


"""
#EXPORT TO CSV, may be useful later
filepath = Path('processed_data/processed_data.csv')
filepath.parent.mkdir(parents=True,exist_ok=True)
df = pd.Series(text_posts)
df.to_csv(filepath)
"""