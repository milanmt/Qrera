#! /usr/bin/env python3

from datetime import datetime, timedelta
import pandas as pd 

def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		t = datetime.now()
		func_val = func(*args,**kwargs)
		time_taken = datetime.now() -t
		print (str(func),' took: ', time_taken)
		return func_val
	return wrapper


@timing_wrapper
def handle_duplicates(df):
	df = df.sort_values(by=['TS'])
	old_duplicates = df[df.duplicated(subset=['TS'], keep=False)]
	df['TS'] = df['TS'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
	power = []
	ts = []
	power.append(df.iloc[0,0])
	ts.append(df.iloc[0,1])
	duplicate_count = 0
	for i in range(1,df.shape[0]):
		# print (i)
		if df.iloc[i,1] != df.iloc[i-1,1]:
			if duplicate_count != 0:
				for j in range(duplicate_count+1):
					ts[-1-j] = ts[-1]-timedelta(seconds=j)

				duplicate_count = 0
		else: 
			duplicate_count += 1
		
		power.append(df.iloc[i,0])
		ts.append(df.iloc[i,1])
	
	new_df = pd.DataFrame(data= list(zip(power,ts)), columns=['POWER', 'TS'])
	new_df['TS'] = new_df['TS'].apply(lambda x: datetime.isoformat(x, sep=' '))

	### Recursive handling of missing values
	new_duplictaes = new_df[new_df.duplicated(subset=['TS'], keep=False)]
	print (new_duplictaes )
	print (new_duplictaes.shape, old_duplicates.shape)

	if not new_duplictaes.empty:
		if new_duplictaes.shape != old_duplicates.shape:
			new_df = handle_duplicates(new_df)

	return new_df 


if __name__ == '__main__':
	path = '/media/milan/DATA/Qrera/PYN/B4E62D388561/2018/11/2018_11_12.csv.gz'

	df = pd.read_csv(path)
	print (df.shape)

	la = df[df.duplicated(subset=['TS'], keep=False)]
	print (la)
	print(la.shape)
	
	df = handle_duplicates(df)
	print (df.shape, 'after handling')

	df = df.drop_duplicates(subset=['TS'], keep='first')
	print (df.shape, 'after second drop')

