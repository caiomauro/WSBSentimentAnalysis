import urllib

import pandas as pd
import praw
import pytesseract
from PIL import Image

# Read-only instance for subreddit
reddit_read_only = praw.Reddit(client_id="baW0zy5tuaCY9-BglOd1mg",         # your client id
                               client_secret="_W7HlrsUFwCHv9k7GAI0kfRju774LQ",      # your client secret
                               user_agent="heatY_12")        # your user agent

subreddit = reddit_read_only.subreddit("wallstreetbets")

print("Display Name: ", subreddit.display_name)
print("Description: ", subreddit.description)
print()

#Map to store posts
hashMap = {}

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

#Helper function to check if a word from keywords exists in the post
def containsWord(keywordList: list, text: str):
    for i in keywordList:
        if i in text:
            return True
    return False

#Helper function to process text from an image with the url
def imageToText(url: str):

    urllib.request.urlretrieve(url, 'image')
    # Open the image file
    image = Image.open('image')

    # Perform OCR using PyTesseract
    text = pytesseract.image_to_string(image)

    # Print the extracted text
    return text

imageToText('https://preview.redd.it/djc1rhwuyxkc1.jpg?width=2021&format=pjpg&auto=webp&s=700352bfc1ed7c171dcac37db9144e77c799302f')

for post in subreddit.hot(limit=10):

    if post.title in hashMap.keys():
        print("title already exists")

    #if post.link_flair_text.strip() == "Discussion":

    if len(post.selftext) == 0 and hasattr(post, 'url') and post.url and containsWord(image_keywords, post.url):
        hashMap[post.title] = "THIS IS FROM A PROCESSED IMAGE: " + post.url + "\n" + imageToText(str(post.url))
    elif len(post.selftext) == 0:
        continue
    else:
        if containsWord(buy_keywords, post.selftext) or containsWord(sell_keywords, post.selftext):
            hashMap[post.title] = post.selftext


for index, value in hashMap.items():
    print("Title: ", index)
    print("-----------------------------------------------------")
    print("Content: ", value)
    print()