import numpy as np
import pandas as pd
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics.pairwise import linear_kernel  # Import for similarity computation
from keras.applications.densenet import DenseNet121
from tensorflow.keras.applications import VGG16 # type: ignore
from keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.densenet import preprocess_input # type: ignore
from keras.layers import GlobalMaxPooling2D # type: ignore
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
from PIL import Image
import joblib  # For saving and loading data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import os
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
import joblib




def preprocess_image(uploaded_file):
    """Preprocess the uploaded image and save it for further processing."""
    try:
        img = Image.open(uploaded_file)

        # Convert palette or RGBA images to RGB
        if img.mode in ("P", "RGBA"):
            img = img.convert("RGB")

        # Resize the image to the required dimensions
        img = img.resize((224, 224))

        # Save the processed image
        img_path = "uploaded_image.jpg"
        img.save(img_path, format="JPEG")
        return img_path

    except Exception as e:
        raise RuntimeError(f"Failed to preprocess the image: {e}")


class FashionRecommendations:
    """Production class for fashion recommendations based on similarity."""

    def __init__(self, img_path,df_embeddings,styles_path):
        self.img_path = img_path
        self.df_embeddings = df_embeddings
        self.styles_path = styles_path

    def get_styles_df(self):
        """Load a DataFrame containing style details and image paths."""
        styles_df = pd.read_csv(self.styles_path, nrows=6000, on_bad_lines='skip') 
        styles_df['image'] = styles_df['id'].astype(str) + ".jpg"  # Generate image column
        return styles_df

    def load_model(self):
        """Load a pre-trained VGG16 model for feature extraction."""
        vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(100, 100, 3))
        vgg16.trainable = False
        return keras.Sequential([vgg16, GlobalMaxPooling2D()])

    def predict(self, model, img_path):
        """Preprocess the image and generate predictions."""
        img = image.load_img(img_path, target_size=(100, 100))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return model.predict(img)

    def get_similarity(self):
        """Compute the similarity of the uploaded image with the dataset."""
        model = self.load_model()
        sample_image = self.predict(model, self.img_path)
        sample_similarity = linear_kernel(sample_image, self.df_embeddings)
        return sample_similarity

    def normalize_sim(self):
        """Normalize the similarity scores."""
        similarity = self.get_similarity()
        x_min = similarity.min(axis=1)
        x_max = similarity.max(axis=1)
        return (similarity - x_min) / (x_max - x_min)[:, np.newaxis]

    def get_recommendations(self):
        """Retrieve the top 5 recommended images."""
        similarity = self.normalize_sim()
        df = self.get_styles_df()
        sim_scores = sorted(enumerate(similarity[0]), key=lambda x: x[1], reverse=True)[:5]
        cloth_indices = [i[0] for i in sim_scores]
        return df['image'].iloc[cloth_indices]

    def print_recommendations(self, output_dir='output_images'):
        """Save the recommended images as output."""
        recommendations = self.get_recommendations()
        os.makedirs(output_dir, exist_ok=True)
        for idx, image_name in enumerate(recommendations):
            cloth_img_path = os.path.join("fashion_small/images", image_name)
            if os.path.exists(cloth_img_path):
                cloth_img = mpimg.imread(cloth_img_path)
                plt.imshow(cloth_img)
                plt.axis("off")
                plt.title("Recommended Image")
                plt.savefig(os.path.join(output_dir, f'recommended_image_{idx}.png'))
                plt.close()


def main():
     st.set_page_config(page_title="FashionFinder: Fashion Recommendation System ", page_icon="👗", layout="wide")
     st.title("👗👜👠 FashionFinder: Fashion Recommendation System ")
     st.markdown("""
            Welcome to the FashionFinder : Upload an image of your outfit, 
            and we will suggest trendy styles that match your look.
            """)
     uploaded_file = st.file_uploader(
        label="Choose an image of your outfit (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="We support JPG, JPEG, and PNG formats. Ensure the image is clear for the best recommendations!"
    )

     if uploaded_file is not None:
        try:
           
            image = Image.open(uploaded_file)
            # Display the resized image
            st.image(image, caption="Uploaded Outfit ", width=200) 
            st.markdown("---")
            st.write("Now, let's find the best fashion recommendations for you!")

            # Process the image and run the recommendation system
            img_path = preprocess_image(uploaded_file)
            df_embeddings = joblib.load('df_embeddings.joblib')
            styles_path = 'C:/Users/jaich/Downloads/Fashion-Recommendation-System-main/Fashion-Recommendation-System-main/fashion_small/styles.csv'
            recommendation_system = FashionRecommendations(
                img_path=img_path,
                df_embeddings=df_embeddings,
                styles_path=styles_path
            )
            recommendation_system.print_recommendations()
            st.markdown("### Recommended Outfits for You:")
            cols = st.columns(5)
            for idx in range(5):
                output_img_path = f'output_images/recommended_image_{idx}.png'
                if os.path.exists(output_img_path):
                    with cols[idx % 5]:
                        st.image(output_img_path, caption=f"Recommendation {idx + 1}", use_container_width=True)
        except FileNotFoundError as e:
            st.error(f"File not found: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
