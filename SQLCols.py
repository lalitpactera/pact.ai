import pandas as pd
import pandasql as ps
class SQLCols:
    def executeSQL(name, query):
        df = pd.read_csv(name)
        return ps.sqldf(query, locals())