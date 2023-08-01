import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tkinter as tk
from tkinter import ttk
from pandastable import Table


# Load datasets
books_df = pd.read_csv('Books.csv', nrows=30000, usecols=['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher'])
users_df = pd.read_csv('Users.csv', nrows=5000)
ratings_df = pd.read_csv('Ratings.csv', nrows=50000)

# Preprocess data
books_df['Book-Title'] = books_df['Book-Title'].fillna('')
books_df['Book-Author'] = books_df['Book-Author'].fillna('')
books_df['Publisher'] = books_df['Publisher'].fillna('')

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['Book-Title'])
print(tfidf_matrix)
# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim)
# Create a mapping between the book title and its index in the dataset
title_to_idx = pd.Series(books_df.index, index=books_df['Book-Title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = title_to_idx[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1][0] if isinstance(x[1], np.ndarray) else x[1], reverse=True)
    print(sim_scores)
    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]

    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return books_df.iloc[book_indices]



def on_click():
    title = entry.get()
    recommendations = get_recommendations(title)
    table_frame = tk.Frame(root)
    table_frame.grid(column=0, row=3, columnspan=2, sticky='nsew')
    table = Table(table_frame, dataframe=recommendations, showtoolbar=False, showstatusbar=False)
    table.autoResizeColumns()
    table.show()

# Create the main window
root = tk.Tk()
root.title("Book Recommender")
root.configure(bg="#E1E5EA")
root.wm_iconbitmap('logo.ico')





# Set initial window size
root.geometry("1000x500")

# Create a custom style for widgets
style = ttk.Style()
style.configure("TLabel", background="#E1E5EA", font=("Helvetica", 14))
style.configure("TButton", font=("Helvetica", 14))
style.configure("TEntry", font=("Helvetica", 14))

# Create and place widgets
label = ttk.Label(root, text="Enter book title:")
label.grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)

entry = ttk.Entry(root, width=50)
entry.grid(column=1, row=0, padx=10, pady=10)

button = ttk.Button(root, text="Get Recommendations", command=on_click)
button.grid(column=1, row=1, padx=10, pady=10)

# Configure grid weights for resizing
root.grid_rowconfigure(3, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Start the Tkinter main loop
root.mainloop()

# Test the recommendation system
# title = ""
# recommendations = get_recommendations(title)
# print(recommendations)