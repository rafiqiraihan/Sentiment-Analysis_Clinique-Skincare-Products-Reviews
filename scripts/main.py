import pandas as pd
from joblib import dump
from automation_preprocessing import TextPreprocessor

def preprocess_text_data(input_path, output_path_data, output_path_pipeline):
    df = pd.read_csv(input_path)

    df.columns = df.columns.str.strip()

    if 'review_text' not in df.columns:
        raise KeyError(f"Kolom 'review_text' tidak ditemukan dalam file: {input_path}\nKolom tersedia: {df.columns.tolist()}")

    text_preprocessor = TextPreprocessor()
    df['processed_text'] = text_preprocessor.fit_transform(df['review_text'])

    df.to_csv(output_path_data, index=False)
    print(f"Data teks telah diproses dan disimpan ke: {output_path_data}")

    dump(text_preprocessor, output_path_pipeline)
    print(f"TextPreprocessor disimpan ke: {output_path_pipeline}")

if __name__ == "__main__":
    preprocess_text_data(
        input_path='data/reviews_clinique.csv',
        output_path_data='processed_reviews_clinique.csv',
        output_path_pipeline='output/text_preprocessor.joblib'
    )
