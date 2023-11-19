Sourcing Notes:

where we got the zip codes
https://www.unitedstateszipcodes.org/zip-code-database/

here is the county level home values
https://cdn.nar.realtor/sites/default/files/documents/2023-q1-county-median-prices-and-monthly-mortgage-payment-by-price-07-10-2023.pdf?_gl=1*1wb2var*_gcl_au*MTE3MzQ2MjU3Ni4xNzAwMzU2ODM2

Coles notes

    import pandas as pd

    url = 'https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv?raw=true'
    df = pd.read_csv(url,index_col=0)
    print(df.head(5))

    https://stackoverflow.com/questions/55240330/how-to-read-csv-file-from-github-using-pandas

    Need to figure out how to upload files to our github and then pull down with pandas code
