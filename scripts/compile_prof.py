import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='Files to compile')
    parser.add_argument('--output', '-o', help='Output file', default='prof/compiled.csv')
    
    args = parser.parse_args()
    files = args.files
    output = args.output
    print("inputs:", files)
    print("output:", output)
    
    df_out = pd.read_csv(files[0], index_col="Metric Name")[['Section Name', 'Metric Unit']]
    rows = []
    for file in files:
        df = pd.read_csv(file, index_col="Metric Name")

        # Change the column name to the file name
        file_name = file.split('/')[-1].split('.')[0]
        rows.append(file_name)
        df[file_name] = df['Metric Value']
        df_out = pd.merge(df_out, df[[file_name]], left_index=True, right_index=True)
    
    df_out.reset_index(inplace=True)
    df_out.reindex(columns=['Section Name', 'Metric Name', 'Metric Unit'] + rows)
    
    df_out.to_csv(output, index=False)

if __name__ == '__main__':
    main()