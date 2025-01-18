FashionFinder: Fashion Recommendation System

Project Description

FashionFinder is a personalized fashion recommendation system designed to suggest similar fashion products based on a user's uploaded image. Leveraging a pre-trained VGG16 model for feature extraction, the system compares the uploaded image to a curated dataset of fashion products to identify the most similar items, offering a seamless and intuitive way to explore fashion choices.

Dataset

The dataset utilized in this project is the "Fashion Product Images (Small)" dataset from Kaggle. This dataset includes comprehensive information about various fashion products, including their images and style details.

Downloading the Dataset

You can download the dataset from Kaggle using the following steps:

Visit the Fashion Product Images (Small) page.

Click the Download button to retrieve the dataset files.

Dataset Structure

The dataset includes a CSV file (styles.csv) with the following columns:

id: Unique identifier for each product

gender: Gender category of the product (e.g., Men, Women)

masterCategory: Main category of the product (e.g., Apparel)

subCategory: Sub-category of the product (e.g., Topwear)

articleType: Specific type of article (e.g., T-Shirts)

baseColour: Primary color of the product

season: Season associated with the product

year: Year the product was released

usage: Usage category (e.g., Casual)

productDisplayName: Display name of the product

The images corresponding to these products are named using the id field followed by .jpg.

File Structure

app.py: Main application script implementing the recommendation system.

styles.csv: Dataset file containing details about fashion products.

requirements.txt: File containing the list of dependencies for the project.

fashion_small/images/: Directory containing product images.

fashion-finder.ipynb: Jupyter notebook for additional analysis and development.

Requirements

To run this project, ensure the following Python packages are installed:

numpy
pandas
tensorflow
keras
matplotlib
opencv-python-headless
streamlit
PIL
joblib

These dependencies are listed in the requirements.txt file. Install the
pip install -r requirements.txt

Running the Project

Step 1: Prepare the Dataset

Place styles.csv in the root directory of the project.

Ensure the images are stored in a directory named fashion_small/images/.

Step 2: Run the Streamlit Application

Execute the following command to launch the application:

python -m streamlit run app.py

Step 3: Upload an Image

Open the Streamlit application in your web browser. Upload a fashion product image in .jpg, .jpeg, or .png format.

Step 4: View Recommendations

The system will process the uploaded image and display the top 5 most similar fashion items.
