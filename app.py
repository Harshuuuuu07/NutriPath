import pandas as pd
import ast
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ========================
# Load and preprocess the data
# ========================
df = pd.read_csv('nutrition.csv')

def safe_parse_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            return val if isinstance(val, list) else [str(val)]
        except:
            return [x.strip()]
    else:
        return []

df['Tags'] = df['Tags'].apply(safe_parse_list)
df['Allergens'] = df['Allergens'].apply(safe_parse_list)

mlb_tags = MultiLabelBinarizer()
tags_encoded = pd.DataFrame(mlb_tags.fit_transform(df['Tags']), columns=mlb_tags.classes_)

mlb_allergens = MultiLabelBinarizer()
allergens_encoded = pd.DataFrame(mlb_allergens.fit_transform(df['Allergens']), columns=mlb_allergens.classes_)

le_cuisine = LabelEncoder()
df['Cuisine'] = le_cuisine.fit_transform(df['Cuisine'])

le_meal = LabelEncoder()
df['Meal Type'] = le_meal.fit_transform(df['Meal Type'])

le_diet = LabelEncoder()
df['Diet'] = le_diet.fit_transform(df['Diet'])

X = pd.concat([
    df[['Cuisine', 'Meal Type', 'Calories (kcal)', 'Protein (g)', 'Fat (g)', 'Carbohydrates (g)']],
    tags_encoded,
    allergens_encoded
], axis=1)

y = df['Diet']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ========================
# Define prediction function
# ========================
def recommend_diet(input_dish):
    input_dish['Cuisine'] = le_cuisine.transform([input_dish['Cuisine']])[0]
    input_dish['Meal Type'] = le_meal.transform([input_dish['Meal Type']])[0]

    input_tags = set(input_dish['Tags'])
    known_tags = set(mlb_tags.classes_)
    filtered_tags = list(input_tags & known_tags)
    tags = pd.DataFrame(mlb_tags.transform([filtered_tags]), columns=mlb_tags.classes_)

    input_allergens = set(input_dish['Allergens'])
    known_allergens = set(mlb_allergens.classes_)
    filtered_allergens = list(input_allergens & known_allergens)
    allergens = pd.DataFrame(mlb_allergens.transform([filtered_allergens]), columns=mlb_allergens.classes_)

    input_features = pd.DataFrame([{
        'Cuisine': input_dish['Cuisine'],
        'Meal Type': input_dish['Meal Type'],
        'Calories (kcal)': input_dish['Calories (kcal)'],
        'Protein (g)': input_dish['Protein (g)'],
        'Fat (g)': input_dish['Fat (g)'],
        'Carbohydrates (g)': input_dish['Carbohydrates (g)']
    }])

    final_input = pd.concat([input_features, tags, allergens], axis=1).reindex(columns=X.columns, fill_value=0)

    pred = model.predict(final_input)[0]
    return le_diet.inverse_transform([pred])[0]

# ========================
# Streamlit UI (Improved)
# ========================
st.set_page_config(page_title="NutriPath AI Diet Recommender", page_icon="🥗")
st.title("🥗 NutriPath: AI-Based Diet and Wellness Guide")

st.markdown("""
Welcome to **NutriPath**! This AI tool helps you get the **best diet type** for your custom dish input.  
Simply fill in the nutritional details, cuisine style, and any dietary tags or allergens to get your recommendation.
""")

with st.form("diet_form"):
    st.subheader("🍽️ Dish Details")

    col1, col2 = st.columns(2)
    with col1:
        cuisine = st.selectbox("🌏 Select Cuisine", le_cuisine.classes_, help="Choose the cuisine this dish belongs to")
    with col2:
        meal_type = st.selectbox("🍴 Meal Type", le_meal.classes_, help="Select whether it's Breakfast, Lunch, Dinner, etc.")

    st.subheader("🧪 Nutritional Information")

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        calories = st.number_input("🔥 Calories (kcal)", min_value=0, help="Total energy provided by the dish")
    with col4:
        protein = st.number_input("💪 Protein (g)", min_value=0.0, help="Protein content in grams")
    with col5:
        fat = st.number_input("🧈 Fat (g)", min_value=0.0, help="Fat content in grams")
    with col6:
        carbs = st.number_input("🍞 Carbohydrates (g)", min_value=0.0, help="Carbohydrate content in grams")

    st.subheader("🏷️ Tags & ⚠️ Allergens")

    selected_tags = st.multiselect("Tags", list(mlb_tags.classes_), help="Health-related or descriptive tags like High-Protein, Spicy, etc.")
    selected_allergens = st.multiselect("Allergens", list(mlb_allergens.classes_), help="Mention allergens this dish may contain (e.g., Milk, Gluten)")

    submit = st.form_submit_button("🔍 Recommend Diet")

if submit:
    user_input = {
        'Cuisine': cuisine,
        'Meal Type': meal_type,
        'Calories (kcal)': calories,
        'Protein (g)': protein,
        'Fat (g)': fat,
        'Carbohydrates (g)': carbs,
        'Tags': selected_tags,
        'Allergens': selected_allergens
    }

    result = recommend_diet(user_input)

    st.success(f"✅ Recommended Diet Type: **{result}**")
    st.markdown("This diet type is predicted based on your nutritional inputs, dish profile, and selected tags/allergens.")
