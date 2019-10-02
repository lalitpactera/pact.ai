import pandas as pd
class RemoveDuplicates:
    def duplicateAll(pandas_object):
        pandas_object.drop_duplicates(inplace=True)
    def duplicateRows(pandas_object, row_name):
        pandas_object.drop_duplicates(row_name, inplace=True)