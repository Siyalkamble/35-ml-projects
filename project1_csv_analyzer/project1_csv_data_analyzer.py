import pandas as pd
from tabulate import tabulate

def read_and_show(data_path):

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("File not found")
        return

    # display rows and columns
    print("=" * 60)
    print("Total Rows : ",df.shape[0])
    print("=" * 60)
    print("Total Columns : ",df.shape[1])

    numeric_col = df.select_dtypes(include = 'number').columns

    if numeric_col.empty: # i can use len(num_col) == 0
        print("File dose not Contains Numeric Columns")
        return

    result = [] # coz if we need we can create a new file
    # You ALWAYS need column names to loop: we did it in num_cols
    for col in numeric_col: 
        result.append({
            "Column" : col,
            'Mean' : round(df[col].mean(),2),
            'Median' : round(df[col].median(),2),
            'Mode': df[col].mode()[0] if not df[col].mode().empty else 'N/A',
            'Sum' : round(df[col].sum(),2),
            'Count' : df[col].count() # this always give you int so not need to round
            
        })
        
    print("=" * 60)
    print("\nStatistical Summary")
    print(tabulate(result,headers = 'keys',tablefmt='grid'))
    print("=" * 60)

    new_data = pd.DataFrame(result)
    new_data.to_csv("Statistical Data.csv",index = False)

if __name__ == "__main__":
    read_and_show("data/data.csv")