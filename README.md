Coles notes

    import pandas as pd

    url = 'https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv?raw=true'
    df = pd.read_csv(url,index_col=0)
    print(df.head(5))

    https://stackoverflow.com/questions/55240330/how-to-read-csv-file-from-github-using-pandas

    Need to figure out how to upload files to our github and then pull down with pandas code
