import pandas as pd
df = pd.read_excel("data/raw/2025 Enola Income Statement - 12 Month Rolling - 7617.xlsx", header=None)




for i in range(20):
    print(i, df.iloc[i].tolist())