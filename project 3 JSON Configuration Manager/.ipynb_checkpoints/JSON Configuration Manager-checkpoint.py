import json

def json_manager(json_file,key,value):

    try:
        with open(json_file,'r',encoding = 'utf-8') as f:
            data = json.load(f)
            
    except FileNotFoundError:
        print(f"{json_file} not found")
        data = {}
        
    except json.JSONDecodeError:
        print(f"{json_file} invalid JSON")         
        data = {}     
        
    if isinstance(data,dict):
       if len(data) == 0:
           print("The Dictionary was not Dictionary, Resetting")
           
    print(f"Before {data}")

    data[key] = value

    print(f"After {data}")

    with open(json_file,'w',encoding = 'utf-8') as f:
        json.dump(data,f)

    return data

if __name__ = "__main__":
    json_manager("sample.json",'name','Siyal_the_Great')