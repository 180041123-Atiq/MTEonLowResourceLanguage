import pandas as pd

def gen_data_csv(xlsx_file_path, csv_file_path):
    # Load data
    df_da_xlsx = pd.read_excel(xlsx_file_path)
    df_da_csv = pd.read_csv(csv_file_path)

    # print(df_da_csv.index[df_da_csv['DA Score'] == '?'].tolist())
    # exit() 
    
    # Average the DA Scores
    avg_score = (df_da_xlsx['DA Score'].astype(float) + df_da_csv['DA Score'].astype(float)) / 2

    # Z-score normalization
    z_scores = (avg_score - avg_score.mean()) / avg_score.std()
    z_scores_clipped = z_scores.clip(-3, 3)
    scaled_z_scores = 100 * (z_scores_clipped + 3) / 6

    # Create final dataframe
    df_final = pd.DataFrame({
        'lp': 'bn-en',
        'src': df_da_xlsx['Sylhet Language'],
        'mt': df_da_xlsx['Machine Translated English'],
        'ref': df_da_xlsx['English Translation'],
        'score': scaled_z_scores,
        'raw': avg_score,
        'annotators': 1,
        'domain': 'text book',
        'year': 2025
    })

    # Train/Val/Test split
    train_df = df_final.iloc[:1200]
    val_df = df_final.iloc[1200:1274]
    test_df = df_final.iloc[1274:1499]  # assuming 1499 total samples

    # Save splits
    dest_path = '/content/drive/MyDrive/MTEonLowResourceLanguage/ONUBAD/extended/OnlySylhetDialectToEnglish/'
    train_df.to_csv(dest_path+'train.csv', index=False)
    val_df.to_csv(dest_path+'val.csv', index=False)
    test_df.to_csv(dest_path+'test.csv', index=False)

if __name__ == '__main__':
    xlsx_file_path = '/content/drive/MyDrive/MTEonLowResourceLanguage/ONUBAD/extended/OnlySylhetDialectToEnglish/samples_miraj.xlsx'
    csv_file_path = '/content/drive/MyDrive/MTEonLowResourceLanguage/ONUBAD/extended/OnlySylhetDialectToEnglish/samples_tasfia.csv'

    gen_data_csv(xlsx_file_path, csv_file_path)