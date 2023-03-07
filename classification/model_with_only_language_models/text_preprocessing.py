import html

def clear_reply_mentions(tweet):
    '''Remove user mentions found in a reply to a tweet.

    Example: @user1 @user2 okay @user3 -> okay @user3
    '''
    # We don't need to use any sophisticated tokenization here like nltk
    tokens = tweet.split(" ")
    for index in range(len(tokens)):
        if not tokens[index].startswith("@"):
            return " ".join(tokens[index:])
    return ""

from emoji import demojize, is_emoji
from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer()

def normalizeToken(token, emojis_found=[], replace_user_mentions=True, replace_urls=True, demojize_emojis=True):
    lowercased_token = token.lower()
    if token.startswith("@") and replace_user_mentions:
        return "@USER"
    elif (lowercased_token.startswith("http") or lowercased_token.startswith("www")) and replace_urls:
        return "HTTPURL"
    elif len(token) == 1 and is_emoji(token):
        emojis_found.append(token)
        if demojize_emojis:
            return demojize(token)
        else:
            return token
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet, tokenizer=tweet_tokenizer, replace_user_mentions=True, replace_urls=True, demojize_emojis=True, bert_tweet_specific_processing=True):
    emojis_found = []
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token, emojis_found=emojis_found, 
                                         replace_user_mentions=replace_user_mentions, 
                                         replace_urls=replace_urls,
                                         demojize_emojis=demojize_emojis) for token in tokens])

    if bert_tweet_specific_processing:
        normTweet = (
            normTweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
        )
        normTweet = (
            normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
        )
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )

    return " ".join(normTweet.split()), emojis_found


def clean_tweet(tweet, clear_html_chars=True, replace_user_mentions=True, replace_urls=True,
                demojize_emojis=True, bert_tweet_specific_processing=True):
    '''Helper function to clean tweets. Highly customizable to fit different needs.

    Params:
        tweet: the tweet to clean
        clear_html_chars: If true, will unescape any special html entities found in the tweet
        replace_user_mentions: If true, will replace any user mention with the token @USER
        replace_urls: If true, will replace any urls with the token HTTPURL
        demojize_emojis: If true, will demojize emojis
        bert_tweet_specific_clean: if true, will do some additional preprocessing for the BertTweet model

    Returns:
        The cleaned tweet
    '''
    # First step: clear mentions at the beginning of tweets (inserted automatically by Twitter when replying to a tweet).
    # These do not count in the character count of a tweet and may make the tweet length go way overboard.
    cleaned_tweet = clear_reply_mentions(tweet)
    
    # Second step: Remove any new lines 
    cleaned_tweet = cleaned_tweet.replace('\r', '').replace('\n', '')

    # Third step: if True, escape any html entities
    if clear_html_chars:
        cleaned_tweet = html.unescape(cleaned_tweet)

    # Normalize Tweet with remaining preprocessing (emojis, urls, mentions, etc..)
    normalized_tweet, emojis = normalizeTweet(cleaned_tweet,
                                              replace_user_mentions=replace_user_mentions,
                                              replace_urls=replace_urls,
                                              demojize_emojis=demojize_emojis,
                                              bert_tweet_specific_processing=bert_tweet_specific_processing)
    
    # TODO: process emoticons? e.g. :)

    return normalized_tweet