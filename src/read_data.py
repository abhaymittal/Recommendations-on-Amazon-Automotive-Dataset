import pandas as pd

def read_data(file_path):
    frame=pd.read_json(file_path,lines=True)
    return frame
    
        
            
