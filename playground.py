import pandas as pd

path = "Participant_export_Basic_Programming_in_Python.csv"

mails = pd.read_csv(path, sep=";")["E-mail"]

str_mails = ""
for mail in mails:
    str_mails += mail + " "

print(str_mails)