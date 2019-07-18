import os
import pandas as pd


folder = []
for root, dirs, files in os.walk("./Birds/dataset/nabirds/images/", topdown=False):

   for name in dirs:
      folder.append(os.path.join(root, name))
      
data = {"path":folder}
df = pd.DataFrame(data)
df.to_csv("bird_corpus.csv",index=False)
