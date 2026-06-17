import pandas as pd
import os

input_file = 'data\CSVFiles\dataset2.csv'  # Replace with your input CSV file path
df = pd.read_csv(input_file)

def create_file_csv(row):
    subject = row['subject']
    filename = os.path.basename(row['video_path']).replace('.mp4', '.txt')
    return f'Physiological\\{subject}\\{filename}'

df['file_CSV'] = df.apply(create_file_csv, axis=1)

df = df[['subject', 'video_path', 'file_CSV', 'x1', 'y1', 'x2', 'y2']]

output_file = 'data\CSVFiles\DataSet2.csv'  
df.to_csv(output_file, index=False)

print(f"Modified CSV saved as {output_file}")