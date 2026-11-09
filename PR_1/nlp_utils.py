import re

def clean_text(text):
    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


positive_words = {
    "good", "great", "excellent", "happy", "love",
    "amazing", "best", "wonderful", "friendly",
    "helpful", "thanks", "awesome"
}

negative_words = {
    "bad", "worst", "poor", "hate", "angry",
    "terrible", "delayed", "rude", "late",
    "awful", "disappointed", "problem"
}


def heuristic_sentiment(text):

    words = clean_text(text).split()

    pos = sum(word in positive_words for word in words)
    neg = sum(word in negative_words for word in words)

    if pos > neg:
        return "positive"

    elif neg > pos:
        return "negative"

    else:
        return "neutral"