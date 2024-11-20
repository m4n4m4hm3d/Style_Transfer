import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image
import io

st.title("Style Transfer")

page = st.radio("Choose a Page", ["Home", "Style Transfer"])


if page == "Home":
    st.header("Welcome to the Style Transfer")
    
    with st.form(key="Login"):
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=4, max_value=150)
        gender = st.selectbox("Gender (There are only 2 Genders)", ["Male", "Female"])
        submit = st.form_submit_button("Submit")
    
    if submit:
        st.session_state.name = name
        st.session_state.age = age
        st.session_state.gender = gender
        st.success(f"Welcome, {name}! Please proceed to the Style Transfer page.")

elif page == "Style Transfer":
    if "name" in st.session_state:
        st.header(f"Hello, {st.session_state.name}! Let's do it.")

        image1 = st.file_uploader("Upload the content image", type=["png", "jpg", "jpeg", "avif"])
        image2 = st.file_uploader("Upload the style image", type=["png", "jpg", "jpeg", "avif"])

        if image1 is not None or image2 is not None:
            col1, col2 = st.columns(2)
            
            if image1 is not None:
                with col1:
                    st.image(image1, caption="Content Image", width=300)

            if image2 is not None:
                with col2:
                    st.image(image2, caption="Style Image", width=300)

        model = tf.saved_model.load('saved_model') #https://www.kaggle.com/models/google/arbitrary-image-stylization-v1/tensorFlow1/256/2?tfhub-redirect=true

        def tensor_to_image(tensor):
            tensor = np.array(tensor * 255, dtype=np.uint8)
            if np.ndim(tensor) > 3:
                tensor = tensor[0]
            return PIL.Image.fromarray(tensor)

        def load_image(uploaded_file):
            max_res = 551
            image = tf.image.decode_image(uploaded_file.read(), channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            shape = tf.cast(tf.shape(image)[:-1], tf.float32)
            long_side = max(shape)
            scaling_factor = max_res / long_side
            new_shape = tf.cast(shape * scaling_factor, tf.int32)
            image = tf.image.resize(image, new_shape)
            image = image[tf.newaxis, :]
            return image

        if image1 is not None and image2 is not None:
            button = st.button("Generate")
            if button:
                placeholder = st.empty()
                placeholder.write("Generating stylized image...")
                
                content_image = load_image(image1)
                style_image = load_image(image2)
                SI = model(tf.constant(content_image), tf.constant(style_image))[0]
                stylized_image = tensor_to_image(SI)
                
                placeholder.empty()

                st.write("### Stylized Image")
                col_mid = st.columns([1, 2, 1])[1]
                
                with col_mid:
                    st.image(stylized_image, caption="Stylized Image", width=400)
                    
                img_byte_arr = io.BytesIO()
                stylized_image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                    
                st.download_button(
                    label="Download",
                    data=img_byte_arr,
                    file_name="stylized_image.png",
                    mime="image/png"
                )
    else:
        st.warning("Please go to the Home page first to enter your details.")


