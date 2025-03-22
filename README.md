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

The comments may contain emojis, Cyrillic and other non-ASCII characters, so we will remove them.
``` python
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)
```
Next create a dataframe with our received comments, apply the remove_emojis function and remove spaces
``` python
df = pd.DataFrame(comments, columns=['comments'])
df['comments'] = df['comments'].apply(remove_emojis)
df = df[df['comments'].str.strip() != '']
df.head()
```
|       | Comment                                                                 |
|-------|-------------------------------------------------------------------------|
| 0     | The greatest innovation OpenAI can make is to ...                        |
| 1     | Yes, I also hear like this when it just release                          |
| 2     | I'm using 4.5 for interpreting case law, it's ...                        |
| 3     | 0:23 best movie ever lmao                                                |
| 4     | Q tristeza                                                             |
