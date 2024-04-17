from datetime import datetime

import pandas as pd
import json


def json_file_to_csv(json_file_path, csv_file_path):
	with open(json_file_path, 'r') as file:
		json_data = json.load(file)

	df = pd.DataFrame(json_data)
	df.to_csv(csv_file_path, index=False)
	return df


# json_file_to_csv("qa_dataset.json","qa_dataset.csv")
def Debug(message):
	now = datetime.now()
	print(f"[{now.strftime("%H:%M:%S")}]: {message}")
