import pandas as pd

from tqdm import tqdm
import argparse
import os

from flair.models import TextClassifier
from flair.data import Sentence

sia = TextClassifier.load('en-sentiment')

def flair_prediction(x):
    sentence = Sentence(x)
    sia.predict(sentence)
    if len(sentence) == 0:
        return None
    label = sentence.labels[0].value
    score = sentence.score

    if label == "POSITIVE":
        return score
    else:
        return -1 * score

def sample_df(df, prop, col):
    df = df.dropna(subset=[col])
    return df.groupby('op_gender', group_keys=False).apply(lambda x: x.sample(frac=prop))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_dir")
    parser.add_argument("--post_or_resp")
    args = parser.parse_args()

    tqdm.pandas()
    print("Reading in file...")
    df = pd.read_csv(args.input_file)
    text_type = args.post_or_resp
    print("Sampling dataset...")
    #sampled_df = sample_df(df, .05, text_type)

    
    print("Sentiment Analysis...")
    sentiment = df[text_type].progress_apply(flair_prediction)

    #sampled_df['Sentiment'] = sentiment

    new_name = os.path.basename(args.input_file)
    new_name = new_name.split(".")[0] + "_sampled.csv"
    df.to_csv(os.path.join("ScriptOutputs/", new_name), index=False)

main()

