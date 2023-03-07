import yaml
import tweepy
import pandas as pd
import pickle

from typing import List

#########################   TWITTER FIELDS     #########################
# Available tweet fields to choose from (as of October 13 2022): 
# attachments, 
# author_id, 
# context_annotations, 
# conversation_id, 
# created_at,
# edit_controls,
# entities, 
# geo, 
# id, 
# in_reply_to_user_id, 
# lang, 
# public_metrics, [Important for retweet and like count]
# possibly_sensitive, 
# referenced_tweets, 
# reply_settings, 
# source, 
# text, 
# withheld
# NOTE: You can only get non-public metrics for Tweets that belong to the account you’re authenticated as. You can use the account token and secret with OAuth 1.0A - this will not work with an app-only bearer token.
TWEET_FIELDS = ['attachments', 'author_id', 'context_annotations', 'created_at', 'entities', 'geo', 'id', 'lang', 'possibly_sensitive', 'public_metrics', 'referenced_tweets', 'text']

# In case of user fields, public_metrics include followers count, following count, tweet count and listed count
USER_FIELDS = ['created_at', 'description', 'id', 'location', 'name', 'pinned_tweet_id', 'protected', 'public_metrics', 'url', 'username', 'verified']
# In case of media fields, public_metrics include view count
# duration_ms available only if type is video
MEDIA_FIELDS = ['media_key', 'type', 'duration_ms', 'public_metrics']

EXPANSIONS = ['author_id', 'attachments.media_keys']

class TwitterClientWrapper:
    def __init__(self, bearer_token, wait_on_rate_limit=False) -> None:

        # Get the bearer token and authenticate with Twitter Client

        # Authenticate to Twitter
        self.client = tweepy.Client(bearer_token, wait_on_rate_limit=wait_on_rate_limit)


    def retrieve_tweets_by_ids(self, ids):
        return self.client.get_tweets(
            ids, tweet_fields=TWEET_FIELDS, expansions=EXPANSIONS, user_fields=USER_FIELDS, media_fields=MEDIA_FIELDS)

    def retrieve_tweet(self, id):
        return self.client.get_tweet(
            id, tweet_fields=TWEET_FIELDS, expansions=EXPANSIONS, user_fields=USER_FIELDS, media_fields=MEDIA_FIELDS)

    
#########################   HELPER FUNCTIONS     #########################
def format_users_df(user_data: List[tweepy.user.User]):
    '''Format the user using his data. If we specify it and the fields, Twitter API can include user data
    inside the "includes" field when retrieving tweets. This returns a Tweepy User object, out of which
    we can retrieve the user fields we specified when querying, by accessing the "data" field.
    '''
    users_data_df = pd.json_normalize(user_data)
    users_data_df.columns = users_data_df.columns.str.removeprefix("public_metrics.")
    return users_data_df

# TODO: Flatten everything and remove prefixes
def format_tweets_df(tweets_data):
    # Remove prefix after normalization of json value columns
    tweets_data_df = pd.json_normalize(tweets_data)
    tweets_data_df.columns = tweets_data_df.columns.str.removeprefix("public_metrics.").str.removeprefix("entities.")

    tweets_data_df.rename(columns={"attachments.media_keys": "has_media"}, inplace=True)
    tweets_data_df['has_media'] = ~tweets_data_df['has_media'].isna()

    # Get the hashtags of a tweet if any
    tweets_data_df['hashtags'] = tweets_data_df['hashtags'].map(lambda hashtags: [hashtag['tag'] for hashtag in hashtags], na_action='ignore')
    '''
    tweets_data_df.rename(columns={"hashtags": "has_hashtags"}, inplace=True)
    tweets_data_df['has_hashtags'] = ~tweets_data_df['has_hashtags'].isna()
    '''

    return tweets_data_df

def format_context_annotations(tweet):
    '''Retrieve the context annotations of a tweet if any, and keep only the ids of the topic domains and entities.
    
    Returns: the tweet formatted, dict of domains retrieved, dict of entities retrieved
    '''
    tweet_copy = tweet.copy()
    context_annotations = tweet_copy.get('context_annotations', [])

    # Create a dict of all topic domains in this tweet
    tweet_topic_domains = dict([(topic['domain']['id'], topic['domain']) for topic in context_annotations])
    # Create a dict of all topic entities in this tweet
    tweet_topic_entities = dict([(topic['entity']['id'], topic['entity']) for topic in context_annotations])
    # Columns contain only the ids of the above topic domains and entities
    tweet_copy['topic_domains'] = list(tweet_topic_domains.keys()) if len(tweet_topic_domains.keys()) > 0 else pd.NA
    tweet_copy['topic_entities'] = list(tweet_topic_entities.keys()) if len(tweet_topic_entities.keys()) > 0 else pd.NA

    # Remove the context annotations column to save space
    tweet_copy.pop('context_annotations', None)
    return tweet_copy, tweet_topic_domains, tweet_topic_entities

def load_topic_domains(path):
    try:
        with open(path, 'rb') as handle:
            topic_domains = pickle.load(handle)
    except FileNotFoundError:
        topic_domains = {}
    return topic_domains

def load_topic_entities(path):
    try:
        with open(path, 'rb') as handle:
            topic_entities = pickle.load(handle)
    except FileNotFoundError:
        topic_entities = {}
    return topic_entities

