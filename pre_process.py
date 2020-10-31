'''
Fucntions to help with the preprocessing of the dhs data
'''

def num_of_vars(df, column):
    '''
    Function that prints the number of unique variables there are, and what the unique variables are
    ---
    Input: <DataFrame>
           <string> the column that you want to know about
    ---
    Output: print statements described above
    '''
    print(f'Num of {column} variables:', df[column].nunique())
    print(df[column].value_counts(), '\n')
    
    
def make_binary(df, column, default, default_num = 0):
    '''
    Function that changes any number of inputs into binary results
    ---
    input: <DataFrame>
           <string> column name
           <string> the default value
           <int> either 0 or 1, the value you want for the default value
    ---
    output: df
    '''
    df[column + '_dup'] = df[column].astype(str)
    
    for idx, old_var in enumerate(df[column + '_dup']):
        if str(old_var) == default:
            df.loc[idx, column + '_dup'] = default_num
        else:
            df.loc[idx, column + '_dup'] = 1 - default_num 
            #if default_num = 1, this will be 0, otherwise it stays 1
        
    return df


def change_vars(mapping, df, column):
    '''
    Function that replaces the old variables with numerical values
    ---
    Input: <dictionary> a mapping of the old vars to the new ones
           <DataFrame>
           <string> column name of column to be changed
    ---
    Output: df
    '''
    
    #make a duplicate column to change
    df[column + '_dup'] = df[column].astype(str)
    
    #replace the old variable with the numerical value in the new column
    for idx, old_var in enumerate(df[column + '_dup']):
        try:        
            df.loc[idx, column + '_dup'] = mapping[old_var]

        except:
            df.drop(idx, inplace = True)

    #reset the index at the end in case any rows got dropped
    df.reset_index(inplace = True, drop = True)
           
    return df


