"""
# Preproc collections and cuisnes in binary vectors 

from sklearn.preprocessing import MultiLabelBinarizer

# Replace NaN values with empty lists
restaurants['Collections'] = restaurants['Collections'].apply(lambda x: x if isinstance(x, list) else [])


mlb = MultiLabelBinarizer()
one_hot_encoded = mlb.fit_transform(restaurants['Collections'])
# Add the one-hot encoded lists as a new column in the DataFrame
restaurants['encoded_Collections'] = one_hot_encoded.tolist()

one_hot_encoded = mlb.fit_transform(restaurants['Cuisines'])
restaurants['encoded_Cuisines'] = one_hot_encoded.tolist()"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import Graphs

def plot_histogram(ax, df, column):
    ax.hist(df[column], color='sandybrown')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_title(f"{column} Distribution")

def plot_histograms(df, cols):
    n_cols = 3
    n_rows = math.ceil(len(cols) / n_cols)
    fig, axs = plt.subplots(figsize=(16, 12), ncols=n_cols, nrows=n_rows)
    axs = axs.flatten()
    for i in range(len(cols)):
        plot_histogram(axs[i], df, cols[i]) 
    for ax in axs:
        if not ax.has_data():  
            fig.delaxes(ax) 
    plt.tight_layout()
    plt.show()

def plot_top10_bar(exploded_col):
    collection_counts = exploded_col.value_counts()
    top_categories = collection_counts.head(10) 

    plt.figure(figsize=(8,5))
    top_categories.plot(kind='bar', color='sandybrown')
    plt.title(f'Top 10 Restaurant {exploded_col.name}', fontsize=14)
    plt.xlabel(exploded_col.name, fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_heatmap(df):
    sns.heatmap(df.corr(numeric_only=True), 
            vmin=-1, 
            vmax=1, 
            cmap=sns.diverging_palette(220, 20, as_cmap=True), 
            annot=True,
            fmt='.2f',
            mask=np.triu(np.ones_like(df.corr(numeric_only=True), dtype=bool), k=1),
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8.5}).set_title(f'Correlation Heatmap')
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.show()

#####################################################################################
def word_freq_calculator(td_matrix, word_list, df_output=True):
    word_counts = np.sum(td_matrix, axis=0).tolist()
    if df_output == False:
        word_counts_dict = dict(zip(word_list, word_counts))
        return word_counts_dict
    else:
        word_counts_df = pd.DataFrame({"words":word_list, "frequency":word_counts})
        word_counts_df = word_counts_df.sort_values(by=["frequency"], ascending=False)
        return word_counts_df
    
def plot_term_frequency(df, nr_terms, df_name, show=True):
    
    # Create the Seaborn bar plot
    plt.figure(figsize=(10, 8))
    sns_plot = sns.barplot(x='frequency', y='words', data=df.head(nr_terms))  # Plotting top 20 terms for better visualization
    plt.title('Top 20 Term Frequencies of {}'.format(df_name))
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    if show==True:
        plt.show()

    fig = sns_plot.get_figure()
    plt.close()

    return fig

def generate_bow_wordcloud(reviews_df, review_column, max_features=10000, ngram_range=(1, 3)):
    """
    Generates a word cloud based on the most frequent terms in a reviews DataFrame.
    
    Parameters:
    - reviews_df (DataFrame): DataFrame containing the reviews.
    - review_column (str): Column name where reviews are stored.
    - max_features (int): Maximum number of features (most frequent terms) for the BOW model.
    - ngram_range (tuple): N-gram range for the vectorizer.
    
    Returns:
    - WordCloud image.
    """
    # Initialize CountVectorizer with the specified max features and n-gram range
    trigram_bow_vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range, token_pattern=r"(?u)\b\w+\b")
    
    # Fit and transform the reviews to get the term-document matrix
    reviews_bow_td_matrix = trigram_bow_vectorizer.fit_transform(reviews_df[review_column])
    
    # Convert to list and get feature names
    reviews_df["initial_bow_vector"] = reviews_bow_td_matrix.toarray().tolist()
    bow_word_list = trigram_bow_vectorizer.get_feature_names_out()
    
    # Calculate word frequencies
    reviews_raw_vocabulary = Graphs.word_freq_calculator(reviews_bow_td_matrix, bow_word_list, df_output=False)
    
    # Generate word cloud
    wc = WordCloud(background_color="white",max_words=120, width = 220,height = 220, color_func=lambda *args, **kwargs: (0,0,0))
    wc.generate_from_frequencies(reviews_raw_vocabulary)
    
    # Return the word cloud image
    return wc.to_image()