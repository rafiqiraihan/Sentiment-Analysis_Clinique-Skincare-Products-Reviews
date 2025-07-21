import pandas as pd
from joblib import dump
from automation_preprocessing import TextPreprocessor

def preprocess_text_data(input_path, output_path_data, output_path_pipeline):
    
    df = pd.read_csv(input_path)

    text_preprocessor = TextPreprocessor()
    df['processed_text'] = text_preprocessor.fit_transform(df['review_text'])

    df.to_csv(output_path_data, index=False)
    print(f"Data teks telah diproses dan disimpan ke:{output_path_data}")

    dump(text_preprocessor, output_path_pipeline)
    print(f"TextPrerocessor disimpan ke:{output_path_pipeline}")

if __name__ == "__main__":
    preprocess_text_data(
        input_path='data/reviews_clinique.csv',
        output_path_data='processed_reviews_clinique.csv',
        output_path_pipeline='output/text_preprocessor.joblib'
    )