#!/usr/bin/env python3
from tweepy import TooManyRequests
import os
import pandas as pd
import pickle
import yaml
import boto3

from helper.twitter_client_wrapper import (
    format_tweets_df, format_users_df, format_context_annotations,
    load_topic_domains, load_topic_entities, TwitterClientWrapper
)

COVID_IDS_PATH = "covid_ids.parquet.gzip"
STEP_SIZE = 100

def run(twitter_client, directory, covid_tweets_ids, gather_retweets=True, push_to_remote=True):
    topic_domains = load_topic_domains(f'{directory}topic_domains.pickle')
    topic_entities = load_topic_entities(f'{directory}topic_entities.pickle')

    # List where we accumulate the tweets retrieved so far
    collected_tweets = []
    # List where we accumulate the users retrieved so far
    collected_users = []
    if gather_retweets:
        # We're gathering retweet ids
        covid_filepath = "covid"
    else:
        # We're gathering retweets themselves
        covid_filepath = "covid_retweets"
    tweet_filepath_temp = f"{covid_filepath}/tweets/"
    user_filepath_temp = f"{covid_filepath}/users/"
    retweet_filepath_temp = f"{covid_filepath}/retweets/"

    # Take the ceil to process any remaining tweet ids
    steps = int(len(covid_tweets_ids)/STEP_SIZE) + 1

    try:
        for i in range(steps):
            tweets = twitter_client.retrieve_tweets_by_ids(ids=covid_tweets_ids[i*STEP_SIZE:(i+1)*STEP_SIZE])
            included_users = tweets.includes.get('users', [])
            collected_users += included_users
            for tweet in tweets.data:
                processed_tweet, tweet_topic_domains, tweet_topic_entities = format_context_annotations(tweet.data)
                collected_tweets.append(processed_tweet)
                topic_domains.update(tweet_topic_domains)
                topic_entities.update(tweet_topic_entities)
    except TooManyRequests:
        # Reached API limit
        print(f"Hit Rate Limit, processed {i * STEP_SIZE}")
        print(f'tweets left: {len(covid_tweets_ids) - (i * STEP_SIZE)}')
    finally:
        # Dump all to parquet and keep track at which user we stopped.
        if len(collected_tweets) > 0:
            # Append end tweet id for this iteration to end of filename
            first_processed_tweet_id = collected_tweets[0]['id']
            last_processed_tweet_id = collected_tweets[-1]['id']
            tweet_filename = f"{first_processed_tweet_id}-to-{last_processed_tweet_id}.parquet.gzip"
            tweet_filepath = directory + tweet_filepath_temp + tweet_filename
            os.makedirs(os.path.dirname(tweet_filepath), exist_ok=True)
            format_tweets_df(collected_tweets).to_parquet(tweet_filepath, compression="gzip", index=False)

            user_filepath = directory + user_filepath_temp + tweet_filename
            os.makedirs(os.path.dirname(user_filepath), exist_ok=True)
            format_users_df([user.data for user in collected_users]).to_parquet(user_filepath, compression="gzip", index=False)

            if gather_retweets:
                # Check if tweet has referenced tweets
                retweeted = [tweet for tweet in collected_tweets if tweet.get('referenced_tweets')]
                # Retrieve all referenced tweets ids in the tweet
                referenced_tweets_ids = set([referenced_tweet['id'] for tweet in retweeted for referenced_tweet in tweet['referenced_tweets'] if referenced_tweet['type'] == 'retweeted'])
                retweet_filepath = directory + retweet_filepath_temp + tweet_filename
                os.makedirs(os.path.dirname(retweet_filepath), exist_ok=True)
                pd.DataFrame(referenced_tweets_ids, columns=['id']).to_parquet(retweet_filepath, compression="gzip", index=False)

            # Save the topics encountered so far as pickle file
            with open(f'{directory}topic_domains.pickle', 'wb') as handle:
                pickle.dump(topic_domains, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'{directory}topic_entities.pickle', 'wb') as handle:
                pickle.dump(topic_entities, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Update the tweets ids to remove the ones already processed
            if len(covid_tweets_ids) < 100:
                pd.DataFrame([], columns=['id']).to_parquet(f"{directory}{COVID_IDS_PATH}", index=False)
            else:
                pd.DataFrame(covid_tweets_ids[(i*STEP_SIZE):], columns=['id']).to_parquet(f"{directory}{COVID_IDS_PATH}", index=False)

            if (push_to_remote):
                s3 = boto3.resource("s3")
                bucket_name = "semester-project-twitter-storage"
                # Upload to S3
                bucket = s3.Bucket(bucket_name)
                bucket.upload_file(tweet_filepath, f"{tweet_filepath_temp}{tweet_filename}")
                bucket.upload_file(user_filepath, f"{user_filepath_temp}{tweet_filename}")
                if gather_retweets:
                    bucket.upload_file(retweet_filepath, f"{retweet_filepath_temp}{tweet_filename}")
        else:
            print("Finished processing users")

        return

def main():
    # TODO: Change depending on whether you're executing this script locally or on a remote server (possibly with s3 access)
    LOCAL = False
    
    if LOCAL:
        DIRECTORY = ""
        with open("api_key.yaml", 'rt') as file:
            secret = yaml.safe_load(file)
        BEARER_TOKEN = secret['Bearer Token']
        PUSH_TO_REMOTE = False
    else:
        DIRECTORY="/home/ubuntu/covid_tweets/"
        BEARER_TOKEN = os.environ["BearerToken"]
        PUSH_TO_REMOTE = True
    
    # Authenticate to Twitter
    client_wrapper = TwitterClientWrapper(BEARER_TOKEN, wait_on_rate_limit=False)

    covid_ids = list(pd.read_parquet(f"{DIRECTORY}{COVID_IDS_PATH}").id)

    if len(covid_ids) != 0:
        run(client_wrapper, DIRECTORY, covid_tweets_ids=covid_ids, gather_retweets=False, push_to_remote=PUSH_TO_REMOTE)

if __name__ == "__main__":
    main()