import argparse
import pandas as pd
from joblib import dump
from scripts.automation_preprocessing import TextPreprocessor

def label_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'

def preprocess_text_data(input_path, output_path_data, output_path_pipeline):
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['review_text', 'rating'])
    df = df.drop_duplicates(subset='review_text')

    text_preprocessor = TextPreprocessor()
    df['processed_text'] = text_preprocessor.fit_transform(df['review_text'])
    df['sentiment'] = df['rating'].apply(label_sentiment)

    df.to_csv(output_path_data, index=False)
    print(f"Data processed saved to: {output_path_data}")

    dump(text_preprocessor, output_path_pipeline)
    print(f"TextPreprocessor saved to: {output_path_pipeline}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess review text data for sentiment analysis")
    parser.add_argument("--data_path", type=str, default="data/reviews_clinique.csv")
    parser.add_argument("--output_path_data", type=str, default="output/processed_reviews_clinique.csv")
    parser.add_argument("--output_path_pipeline", type=str, default="output/text_preprocessor.joblib")
    args = parser.parse_args()

    preprocess_text_data(
        input_path=args.data_path,
        output_path_data=args.output_path_data,
        output_path_pipeline=args.output_path_pipeline
    )
