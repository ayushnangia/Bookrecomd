import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tkinter as tk
from tkinter import ttk
import pandastable
from pandastable import Table
import ctypes
import requests
import os
import uuid
import glob
from PIL import Image, ImageTk
from io import BytesIO

# Load datasets
books_df = pd.read_csv('Books.csv', nrows=30000, usecols=['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher'])
users_df = pd.read_csv('Users.csv', nrows=5000)
ratings_df = pd.read_csv('Ratings.csv', nrows=50000)

# Preprocess data
books_df['Book-Title'] = books_df['Book-Title'].fillna('')
books_df['Book-Author'] = books_df['Book-Author'].fillna('')
books_df['Publisher'] = books_df['Publisher'].fillna('')
books_df['Image-URL-M'] = pd.read_csv('Books.csv', usecols=['Image-URL-M'])['Image-URL-M']

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['Book-Title'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a mapping between the book title and its index in the dataset
title_to_idx = pd.Series(books_df.index, index=books_df['Book-Title']).drop_duplicates()
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tkinter as tk
from tkinter import ttk
import pandastable
from pandastable import Table
import ctypes
import requests
import os
import uuid
import glob
from PIL import Image, ImageTk
from io import BytesIO

# Load datasets
books_df = pd.read_csv('Books.csv', nrows=30000, usecols=['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher'])
users_df = pd.read_csv('Users.csv', nrows=5000)
ratings_df = pd.read_csv('Ratings.csv', nrows=50000)

# Preprocess data
books_df['Book-Title'] = books_df['Book-Title'].fillna('')
books_df['Book-Author'] = books_df['Book-Author'].fillna('')
books_df['Publisher'] = books_df['Publisher'].fillna('')
books_df['Image-URL-M'] = pd.read_csv('Books.csv', usecols=['Image-URL-M'])['Image-URL-M']

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['Book-Title'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a mapping between the book title and its index in the dataset
title_to_idx = pd.Series(books_df.index, index=books_df['Book-Title']).drop_duplicates()
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the book that matches the title
    idx = title_to_idx[title]

    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1][0] if isinstance(x[1], np.ndarray) else x[1], reverse=True)

    # Get the scores and image URLs of the 5 most similar books
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    recommendations = books_df.iloc[book_indices][['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-M']].reset_index(drop=True)

    # Add a new column with the images as PIL ImageTk objects
    images = []
    for url in recommendations['Image-URL-M']:
        if not pd.isna(url):
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).resize((75, 100))
                images.append(ImageTk.PhotoImage(img))
            else:
                images.append(None)
        else:
            images.append(None)
    recommendations.insert(0, 'Image', images)

    return recommendations


def show_image(index):
    # Get the filename of the image for the selected row
    filename = recommendations_table.model.df.iloc[index]['Image']

    # Create a new window to display the image
    image_window = tk.Toplevel(root)
    image_window.title("Cover Image")
    image_window.configure(bg="#E1E5EA")

    # Load the image and display it in a Label widget
    img = Image.open(filename)
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(image_window, image=photo)
    label.image = photo
    label.pack()

def on_click():
    title = entry.get()
    recommendations = get_recommendations(title)

    # Create a frame to hold the table and buttons
    global table_frame
    table_frame = tk.Frame(root)
    table_frame.grid(column=0, row=3, columnspan=2, sticky='nsew')

    # Create the recommendations table
    global recommendations_table
    recommendations_table = Table(table_frame, dataframe=recommendations, showtoolbar=False, showstatusbar=False)
    recommendations_table.autoResizeColumns()
    recommendations_table.show()

    # Set the width of the first column
    recommendations_table.columnconfigure(0, minsize=100)

    # Download the cover images and update the table
    download_cover_images(recommendations)
    update_table_with_images(recommendations)

def download_cover_images(recommendations):
    # Set the user agent for the requests module
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    # Create the images folder if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')

    # Download the images for the recommended books
    for index, row in recommendations.iterrows():
        url = row['Image-URL-M']
        filename = None
        if not pd.isna(url):
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                # Convert the mode to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((75, 100))
                filename = f"images/{uuid.uuid4()}.jpg"  # set a unique filename for the image
                img.save(filename)
            else:
                filename = 'images/default.jpg'  # use the default image
        else:
            filename = 'images/default.jpg'  # use the default image
        recommendations.at[index, 'Image'] = filename

def update_table_with_images(recommendations):
    # Create a new column with the images as PhotoImage objects
    photo_images = []
    image_buttons = []
    for filename in recommendations['Image']:
        if filename is not None:
            img = Image.open(filename).resize((75, 100))
            photo_images.append(ImageTk.PhotoImage(img))
            button = tk.Button(root, text="View Cover", command=lambda filename=filename: show_image_by_filename(filename))
            image_buttons.append(button)
        else:
            photo_images.append(None)
            image_buttons.append(None)
    recommendations.insert(0, 'Image Preview', photo_images)
    recommendations.insert(1, 'Image Button', image_buttons)

    # Remove the Image-URL-M and Image columns
    recommendations = recommendations.drop(['Image-URL-M', 'Image'], axis=1)

    # Update the recommendations table with the new data and images
    recommendations_table.updateModel(dataframe=recommendations)
    for i, img in enumerate(photo_images):
        if img is not None:
            recommendations_table.model.dfTk[i, 0] = img
            recommendations_table.model.dfTk[i, 1] = image_buttons[i]

def show_image_by_filename(filename):
    # Create a new window to show the image
    image_window = tk.Toplevel(root)
    image_window.title("Cover Image")
    image_window.configure(bg="#E1E5EA")

    # Load the image and display
    img = Image.open(filename)
    photo_image = ImageTk.PhotoImage(img)
    label = tk.Label(image_window, image=photo_image)
    label.pack()

    # Center the window on the screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    image_window_width = 400
    image_window_height = 500
    x = (screen_width // 2) - (image_window_width // 2)
    y = (screen_height // 2) - (image_window_height // 2)
    image_window.geometry(f"{image_window_width}x{image_window_height}+{x}+{y}")

    # Make the window stay on top of the main window
    image_window.attributes("-topmost", True)
    image_window.protocol("WM_DELETE_WINDOW", lambda: image_window.destroy())


def clear_images():
    # Delete all image files in the images folder
    for file in glob.glob("images/*.jpg"):
        os.remove(file)

    # Clear the Image and Image Button columns in the recommendations table
    recommendations_table.model.df['Image'] = None
    recommendations_table.model.df['Image Button'] = None

def main():
    # Load datasets
    books_df = pd.read_csv('Books.csv', nrows=30000, usecols=['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher'])
    users_df = pd.read_csv('Users.csv', nrows=5000)
    ratings_df = pd.read_csv('Ratings.csv', nrows=50000)

    # Preprocess data
    books_df['Book-Title'] = books_df['Book-Title'].fillna('')
    books_df['Book-Author'] = books_df['Book-Author'].fillna('')
    books_df['Publisher'] = books_df['Publisher'].fillna('')
    books_df['Image-URL-M'] = pd.read_csv('Books.csv', usecols=['Image-URL-M'])['Image-URL-M']

    # Compute TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books_df['Book-Title'])

    # Compute cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Create a mapping between the book title and its index in the dataset
    title_to_idx = pd.Series(books_df.index, index=books_df['Book-Title']).drop_duplicates()

    # Create the main window
    global root
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

    label = ttk.Label(root, text="Enter book title:")
    label.grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)

    global entry
    entry = ttk.Entry(root, width=50)
    entry.grid(column=1, row=0, padx=10, pady=10)

    button = ttk.Button(root, text="Get Recommendations", command=on_click)
    button.grid(column=1, row=1, padx=10, pady=10)

    root.grid_rowconfigure(3, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    root.mainloop()

if __name__ == '__main__':
    main()
