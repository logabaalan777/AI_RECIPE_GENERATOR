import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from streamlit_lottie import st_lottie
import json
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import textwrap

# Function to load a Lottie animation from a local JSON file
def load_lottie_file(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

# Load the data from the pickle file
pickle_file = 'recipe_recommender.pkl'
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

tfidf_vectorizer = data['tfidf_vectorizer']
tfidf_matrix = data['tfidf_matrix']
rp = data['rp']
cv = data['cv']
vectors = data['vectors']
similarity = data['similarity']

# Function to recommend recipes based on user input
def recommend_user_input(user_input, top_n=10):
    user_input_vector = tfidf_vectorizer.transform([' '.join(user_input)])
    similarities = cosine_similarity(user_input_vector, tfidf_matrix)
    top_indices = similarities.argsort()[0, ::-1][:top_n]
    recommended_recipes = rp.iloc[top_indices]
    return recommended_recipes

# Function to create an image with recipe details
def create_recipe_image(recipe, filepath):
    width = 1200  # Canvas width
    background_color = "white"
    font_path = "arial.ttf"  # Update this to your preferred font

    # Calculate the required height based on the number of instructions
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        font = ImageFont.load_default()

    instructions = recipe['TranslatedInstructions'].replace('. ', '.\n').split('\n')
    num_lines = sum([len(textwrap.wrap(instruction, width=70)) for instruction in instructions])  # Wrap instructions
    height = 700 + num_lines * 30  # Adjust height based on number of instructions

    # Create a blank image with white background
    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # Load font
    try:
        font_bold = ImageFont.truetype(font_path, 30)
    except IOError:
        font_bold = ImageFont.load_default()

    # Draw the recipe title
    draw.text((20, 20), recipe['TranslatedRecipeName'], fill="black", font=font_bold)

    # Draw the recipe image
    try:
        response = requests.get(recipe['image-url'])
        recipe_image = Image.open(BytesIO(response.content)).resize((1160, 400))
        image.paste(recipe_image, (20, 60))
        y_text = 480
    except:
        draw.text((20, 60), "Image not available", fill="black", font=font)
        y_text = 100

    # Draw the instructions
    draw.text((100, y_text), "Instructions:", fill="black", font=font_bold)
    y_text += 60
    for instruction in instructions:
        if instruction.strip():
            wrapped_text = textwrap.fill(instruction.strip(), width=100)
            y_text += 5
            draw.text((50, y_text), f"--> ", fill="black", font=font)
            for line in wrapped_text.split('\n'):
                draw.text((80, y_text), f" {line}", fill="black", font=font)
                y_text += 30
    # Save the image
    image.save(filepath)

# Load a Lottie animation from a local file
lottie_animation = load_lottie_file("cooking-animation.json")  # Update with your file path

# Streamlit app title
st.title("Recipe Recommender")

# Custom CSS for styling
st.markdown("""
    <style>
    .recipe-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .recipe-card h3 {
        margin-top: 0;
        color: #333;
    }
    .recipe-card img {
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .recipe-card .instructions {
        margin-top: 10px;
    }
    .instructions ul {
        padding-left: 20px;
    }
    .instructions ul li {
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

    # Create tabs
tab1, = st.tabs(["Ingredients-based Recommendation"])

with tab1:
    st.header("Get Recipe Recommendations Based on Ingredients")
    ingredients = st.text_area("Enter your preferred ingredients (separated by commas):").lower().split(',')

    # Display Lottie animation on the side
    with st.sidebar:
        st_lottie(lottie_animation, height=300)

    if st.button("Recommend Recipes"):
        if ingredients:
            recommended_recipes = recommend_user_input(ingredients)
            st.subheader("Recommended Recipes:")
            for idx, (_, recipe) in enumerate(recommended_recipes.iterrows(), start=1):
                st.markdown(f"""
                <div class="recipe-card">
                    <h3>{idx}. {recipe['TranslatedRecipeName']}</h3>
                    <img src="{recipe['image-url']}" width="500" />
                    <div class="instructions">
                        <ul>
                """, unsafe_allow_html=True)
                st.markdown("#### Instructions:")
                instructions = recipe['TranslatedInstructions'].replace('. ', '.\n').split('\n')
                for instruction in instructions:
                    if instruction.strip(): 
                        st.markdown(f"<li>{instruction.strip()}</li>", unsafe_allow_html=True)
                
                # Path to save the image
                recipe_image_path = f"recipe_{idx}.png"
                
                # Create the image with the recipe details
                create_recipe_image(recipe, recipe_image_path)
                
                # Download button for the recipe image
                with open(recipe_image_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Recipe",
                        data=file,
                        file_name=recipe_image_path,
                        mime="image/png"
                    )
                
                st.markdown("""
                        </ul>
                    </div>
                </div>
                <hr>
                """, unsafe_allow_html=True)
        else:
            st.write("Please enter some ingredients.")

