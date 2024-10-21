
# Preproc collections and cuisnes in binary vectors 

from sklearn.preprocessing import MultiLabelBinarizer

# Replace NaN values with empty lists
restaurants['Collections'] = restaurants['Collections'].apply(lambda x: x if isinstance(x, list) else [])


mlb = MultiLabelBinarizer()
one_hot_encoded = mlb.fit_transform(restaurants['Collections'])
# Add the one-hot encoded lists as a new column in the DataFrame
restaurants['encoded_Collections'] = one_hot_encoded.tolist()

one_hot_encoded = mlb.fit_transform(restaurants['Cuisines'])
restaurants['encoded_Cuisines'] = one_hot_encoded.tolist()