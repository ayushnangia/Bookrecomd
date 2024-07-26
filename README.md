# Bookrecomd


## Overview

This project is a Book Recommendation System with a graphical user interface built using Tkinter. It uses collaborative filtering and TF-IDF (Term Frequency-Inverse Document Frequency) to suggest books based on user input.

## Features

- User-friendly GUI for easy interaction
- Book recommendations based on title input
- Displays book details including title, author, publication year, and publisher
- Shows book cover images for visual reference
- Ability to view larger cover images

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- tkinter
- pandastable
- Pillow
- requests

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ayushnangia/Bookrecomd.git
   cd Bookrecomd
   ```

2. Install the required packages:
   ```
   pip install pandas numpy pillow requests pandastable sklearn
   ```

3. Download the dataset files (Books.csv, Users.csv, Ratings.csv) and place them in the project directory.

## Usage

1. Run the main script:
   ```
   python project.py
   ```

2. Enter a book title in the input field.

3. Click the "Get Recommendations" button to see similar book recommendations.

4. Click on "View Cover" to see a larger version of the book cover.

## How it works

1. The system loads and preprocesses data from CSV files containing book information, user data, and ratings.

2. It uses TF-IDF vectorization on book titles to create a matrix representation of the books.

3. Cosine similarity is computed between book vectors to find similar books.

4. When a user inputs a book title, the system finds the most similar books based on cosine similarity scores.

5. The recommendations are displayed in a table format with book details and cover images.

## Project Structure

- `project.py`: The main script that runs the GUI and recommendation system.
- `Books.csv`: Dataset containing book information.
- `Users.csv`: Dataset containing user information.
- `Ratings.csv`: Dataset containing user ratings for books.
- `images/`: Directory to store downloaded book cover images.
- `logo.ico`: Icon file for the application window.

