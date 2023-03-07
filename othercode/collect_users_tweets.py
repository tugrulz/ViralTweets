#!/usr/bin/env python3
from tweepy import Paginator, TooManyRequests
import os
import pandas as pd
import pickle
from tqdm import tqdm
import yaml

import boto3

from helper.twitter_client_wrapper import (
    TWEET_FIELDS,
    format_tweets_df, format_context_annotations,
    load_topic_domains, load_topic_entities, TwitterClientWrapper
)

USER_IDS_PATH = "users_ids.csv"

def run(twitter_client, directory, users_ids, tweets_per_user=20000, push_to_remote=True):
    topic_domains = load_topic_domains(f'{directory}topic_domains.pickle')
    topic_entities = load_topic_entities(f'{directory}topic_entities.pickle')

    # List where we accumulate the tweets retrieved so far
    viral_users_tweets = []
    # Number of users processed so far
    users_processed = 0
    filename = f"tweets/{users_ids.id[0]}-to-"

    try:
        for user_id in tqdm(users_ids.id):
            for tweet in Paginator(twitter_client.get_users_tweets, id=user_id, tweet_fields=TWEET_FIELDS, exclude="retweets").flatten(limit=tweets_per_user):
                processed_tweet, tweet_topic_domains, tweet_topic_entities = format_context_annotations(tweet.data)
                viral_users_tweets.append(processed_tweet)
                topic_domains.update(tweet_topic_domains)
                topic_entities.update(tweet_topic_entities)
            users_processed += 1
    except TooManyRequests:
        # Reached API limit
        print("Hit Rate Limit")
    finally:
        # Dump all to parquet and keep track at which user we stopped.
        if len(viral_users_tweets) > 0:
            # Append end user id for this iteration to end of filename
            filename += f"{user_id}.parquet.gzip"
            filepath = directory + filename
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            format_tweets_df(viral_users_tweets).to_parquet(filepath, compression="gzip", index=False)

            # Save the topics encountered so far as pickle file
            with open(f'{directory}topic_domains.pickle', 'wb') as handle:
                pickle.dump(topic_domains, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'{directory}topic_entities.pickle', 'wb') as handle:
                pickle.dump(topic_entities, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Update the users ids to remove the ones already processed
            users_ids[users_processed:].to_csv(f"{directory}{USER_IDS_PATH}", index=False)

            if (push_to_remote):
                s3 = boto3.resource("s3")
                bucket_name = "semester-project-twitter-storage"
                # Upload to S3
                s3.Bucket(bucket_name).upload_file(filepath, filename)
        else:
            print("Finished processing users")

    return

def main():
    # TODO: Change depending on whether you're executing this script locally or on a remote server (possibly with s3 access)
    LOCAL = False
    TWEETS_PER_USER = 4000
    
    if LOCAL:
        DIRECTORY = ""
        with open("api_key.yaml", 'rt') as file:
            secret = yaml.safe_load(file)
        BEARER_TOKEN = secret['Bearer Token']
        PUSH_TO_REMOTE = False
    else:
        DIRECTORY="/home/ubuntu/tweet/"
        BEARER_TOKEN = os.environ["BearerToken"]
        PUSH_TO_REMOTE = True
    
    # Authenticate to Twitter
    client_wrapper = TwitterClientWrapper(BEARER_TOKEN, wait_on_rate_limit=False)
    client = client_wrapper.client

    users_ids = pd.read_csv(f"{DIRECTORY}{USER_IDS_PATH}", dtype={"id": str})

    if len(users_ids) != 0:
        run(client, DIRECTORY, users_ids=users_ids, tweets_per_user=TWEETS_PER_USER, push_to_remote=PUSH_TO_REMOTE)

if __name__ == "__main__":
    main()