import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
class CategoricalColumns:
    def getCategoryLabel(Xcol):
        Xcolunique = Xcol.unique()
        i1 = 0
        for num in Xcolunique:
            Xcol = Xcol.replace(num, str(i1))
            i1 = i1 + 1
        return(Xcol)

    def convertAll(pandas_object):
        for col in pandas_object.columns:
            if is_numeric_dtype(pandas_object[col]):
                pandas_object[col] = pandas_object[col]
            else:
                pandas_object[col] = pd.to_numeric(CategoricalColumns.getCategoryLabel(pandas_object[col])) 