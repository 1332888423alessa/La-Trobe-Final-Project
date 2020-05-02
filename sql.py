import pyodbc
import pandas as pd
# Some other example server values are
# server = 'localhost\sqlexpress' # for a named instance
# server = 'myserver,port' # to specify an alternate port
server = 'ltuteamx.database.windows.net'
database = 'teamx_db (ltuteamx/teamx_db)'
username = 'teamx' #'myusername'
password = 'latrobe123.'#'mypassword'
                                     #'Trusted_Connection=yes'
#cnxn = pyodbc.connect('DRIVER={ SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)

cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                                    'SERVER=ltuteamx.database.windows.net;'
                                    'DATABASE=teamx_db (ltuteamx/teamx_db);'
                                    'UID=teamx;''PWD=latrobe123.')
cursor = cnxn.cursor()
csv_data = pd.read_csv("Train_Final.csv") # Read the csv
for rows in csv_data: # Iterate through csv
    cursor.execute("INSERT INTO ltuteamx.dbo.MyTable(Col1,Col2,Col3,Col4) VALUES (?,?,?,?)",rows)
cnxn.commit()