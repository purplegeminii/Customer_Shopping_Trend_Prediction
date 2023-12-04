import streamlit as st  # importing streamlit
import numpy as np  # importing numpy
import pickle  # importing pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow import keras
from keras.models import load_model

st.title('Customer Purchase')
#
# col1, col2 = st.columns(2)
#
# col1.image(str('fifa-soccer.png'))
image_column, content_column = st.columns([1, 4])

# Add the image to the first column

# Add content to the second column


age = int(st.slider("Age", 0, 105, 50))
purchase_amount = int(st.slider("Purchase Amount", 0, 100, 50))
review_rating = float(st.slider("Review Rating", 0.0, 5.0, 1.0, 0.1))
previous_purchase = int(st.slider("Previous Purchase", 0, 100, 5))


categories = ['Clothing', 'Footwear', 'Outwear','Accessories']
category = str(st.selectbox('Select a category', categories))


# locations from dataset
locations = ['Kentucky', 'Maine', 'Massachusetts', 'Rhode Island', 'Oregon', 'Wyoming', 'Montana', 'Louisiana',
             'West Virginia', 'Missouri', 'Arkansas', 'Hawaii', 'Delaware', 'New Hampshire', 'New York', 'Alabama',
             'Mississippi', 'North Carolina', 'California', 'Oklahoma', 'Florida', 'Texas', 'Nevada', 'Kansas',
             'Colorado', 'North Dakota', 'Illinois', 'Indiana', 'Arizona', 'Alaska', 'Tennessee', 'Ohio', 'New Jersey',
             'Maryland', 'Vermont', 'New Mexico', 'South Carolina', 'Idaho', 'Pennsylvania', 'Connecticut', 'Utah',
             'Virginia', 'Georgia', 'Nebraska', 'Iowa', 'South Dakota', 'Minnesota', 'Washington', 'Wisconsin',
             'Michigan']
# Create a dropdown menu
location = str(st.selectbox('Select a location', locations))

# colors from dataset
colors = ['Gray', 'Maroon', 'Turquoise', 'White', 'Charcoal', 'Silver', 'Pink', 'Purple', 'Olive', 'Gold', 'Violet',
          'Teal', 'Lavender', 'Black', 'Green', 'Peach', 'Red', 'Cyan', 'Brown', 'Beige', 'Orange', 'Indigo', 'Yellow',
          'Magenta', 'Blue']
color = str(st.selectbox('Select a color', colors))

# seasons
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
# season = str(st.selectbox('Select a season', seasons, index=0))
season = st.radio("Select a season", seasons)

payment_methods = ['Venmo', 'Cash', 'Credit Card', 'PayPal', 'Bank Transfer', 'Debit Card']
payment_method = str(st.selectbox('Select a payment method', payment_methods))

frequency_of_purchase_list = ['Fortnightly', 'Weekly', 'Annually', 'Quarterly', 'Bi-Weekly', 'Monthly',
                              'Every 3 Months']
frequency_of_purchase = str(st.selectbox('Select a purchase frequency', frequency_of_purchase_list))

#
if st.button('SUBMIT'):
    user_values = [age, purchase_amount, review_rating, previous_purchase, category, location, color, season,
                   payment_method, frequency_of_purchase]
    column_names = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases', 'Category', 'Location', 'Color',
                    'Season', 'Payment Method', 'Frequency of Purchases']

    # user_values = {column_names[i]: user_values[i] for i in range(len(column_names))}
    user_values = np.array(user_values)
    user_values = pd.DataFrame(user_values.reshape(1, -1), columns=column_names)
    user_values = pd.DataFrame(user_values, columns=column_names)

    # label encoding
    with open('encoders_dict.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    for col in label_encoders.keys():
        print(col)
        if col in user_values.columns:
            encoder = label_encoders[col]
            user_values[col] = encoder.fit_transform(user_values[col])
        else:
            print(1)

    scaler = pickle.load(open('x_scaler.pkl', 'rb'))
    # model = pickle.load(open('grid_model9.keras', 'rb'))
    model = load_model('grid_model8.h5')
    scaled_user_inputs = pd.DataFrame(scaler.transform(user_values))
    y_pred = model.predict(scaled_user_inputs)
    predicted_class = tensorflow.argmax(y_pred, axis=1)
    decoded_pred = label_encoders['Item Purchased'].inverse_transform(predicted_class)
    st.write(user_values)
    st.write(decoded_pred)
