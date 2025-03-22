# Ai Sentiment Analysis

# About the project
In the world of artificial intelligence, people have different opinions about it. In this project, we will analyze the sentiment of comments about AI using two popular videos about GPT and Gemini.

# Youtube Resources
Video from YouTube channel [Fireship](https://www.youtube.com/@Fireship)
- [Video about GPT](https://youtu.be/FW2XOIxaNqg?si=jaz2trhQh1VZCgQD)
- [Video about Gemini](https://youtu.be/k9xbh9LUYn0?si=Xri8VEv4VPAFk2FX)

# Getting comments from videos
The first step is to get the YouTube Data API v3 from [Google Cloud Console](https://console.cloud.google.com/).

Next installing a library for working with YouTube:
``` python
pip install google-api-python-client
```
commentThreads().list() — API method for getting a list of comments for a given video (videoId=video link).
