import os
import pandas as pd

def final_evaluation():
	directory = "results"

	files = [f for f in os.listdir(directory) if f.startswith("BASIC_Eval") and f.endswith(".csv")]

	combined_df = pd.DataFrame()

	for file in files:
		file_path = os.path.join(directory, file)
		temp_data = pd.read_csv(file_path)

		temp_data = temp_data[['cost', 'length', 'time taken', 'accuracy']].copy()

		model_name = file.replace("BASIC_Eval_", "").replace(".csv", "")
		temp_data['Model'] = model_name

		temp_data.rename(columns={'time taken': 'speed'}, inplace=True)

		combined_df = pd.concat([combined_df, temp_data], ignore_index=True)

	average_df = combined_df.groupby('Model').agg({
		'speed': 'mean',
		'accuracy': 'mean',
		'cost': 'mean',
		'length': 'mean'
	}).reset_index()

	average_df['speed'] = average_df['speed'].round(3)
	average_df['accuracy'] = average_df['accuracy'].round(2) * 100
	average_df['length'] = average_df['length'].round(2)
	# maybe change to cost per 100k prompts?

	if 'appropriateness' in combined_df.columns:
		average_df['appropriateness'] = combined_df.groupby('Model')['appropriateness'].mean().round(2).values
	else:
		average_df['appropriateness'] = pd.NA

	average_csv_path = os.path.join(directory, 'Final_BASIC_Rankings.csv')
	average_df.to_csv(average_csv_path, index=False)

	print(f"{average_csv_path} updated")
