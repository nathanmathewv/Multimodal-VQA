# load csv's and merge them into one
import pandas as pd
import os
import glob

def merge_csv_files(input_folder, output_file):

    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    csv_files = [file for file in csv_files if os.path.basename(file).startswith('image_data_with_vqa')]
    print(f"Found {len(csv_files)} CSV files in {input_folder}")
    print(f"CSV files: {csv_files}")


    dataframes = []


    for file in csv_files:
        print(f"Reading {file}")
        df = pd.read_csv(file)
        print(f"Shape : {df.shape}")
        print(f"Columns : {df.columns.tolist()}")
        print(f"Length of image_id: {len(df['image_id'])}")
        print(f"Length of vqa_response: {len(df['vqa_response'].notna())}")
        dataframes.append(df)

    #for each row in datafram, if vqa_response is not null, then add that row to the new dataframe
    # if that row is already there in the new dataframe (by checking image_id), then skip that row
    merged_df = pd.DataFrame(columns=dataframes[0].columns.tolist())
    for df in dataframes:
        for index, row in df.iterrows():
            if pd.notna(row['vqa_response']) and row['image_id'] not in merged_df['image_id'].values:
                merged_df = pd.concat([merged_df, pd.DataFrame([row])], ignore_index=True)

    print("Length of the image vqa dataset",len(merged_df[merged_df['image_id'].notna()]))
    print("Number of populated responses",len(merged_df[merged_df['vqa_response'].notna()]))


    merged_df.to_csv(output_file, index=False)
    print(f'Merged {len(csv_files)} CSV files into {output_file}')

if __name__ == "__main__":
    input_folder = '/Users/nathanmathew/Desktop/College/Sem 6/AIM 825 Visual Recognition/MiniProject-2/Multimodal-VQA/Dataset/metadata'
    output_file = 'merged_image_data_vqa.csv' 


    merge_csv_files(input_folder, output_file)