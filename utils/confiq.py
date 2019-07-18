
# Constant configuaration for the model


import argparse
import torch


parser = argparse.ArgumentParser(description="Training Model")


#parser.add_argument("user", nargs='?',default="Admin")

parser.add_argument("--no_classes", default=200, type=int, help ="Enter the total the number of classes")
parser.add_argument("--dropout", default=0.3, type=str)
parser.add_argument("--surename", nargs="+", type=str)
args = parser.parse_args()
print ("My name is ", args.name, end=' ')
if args.surename:
   print (args.surename)


#args = parser.parse_args()


