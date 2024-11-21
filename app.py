import os
import requests
import base64
import streamlit as st
from PIL import Image
from io import BytesIO
from streamlit_cropper import st_cropper  
from utils.model_wrappers.multimodal_models import SambastudioMultimodal 
from dotenv import load_dotenv


load_dotenv()


lvlm = SambastudioMultimodal(
    model="Llama-3.2-11B-Vision-Instruct",
    temperature=0.01,
    max_tokens_to_generate=1024,
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1/chat/completions",
)


st.title("Enhanced Image Upload and Query Processing")
st.write("""
    - Upload multiple images or provide a URL.
    - Optionally crop or edit the images before processing.
    - Ask a query related to the image and get the answer.
""")


os.makedirs("uploaded_images", exist_ok=True)


uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


image_url = st.text_input("Or provide an image URL")

images = []
cropped_images = [] 


if uploaded_files:
    st.write("Uploaded Images:")
    for uploaded_file in uploaded_files:
       
        local_path = os.path.join("uploaded_images", uploaded_file.name)
        with open(local_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

       
        image = Image.open(uploaded_file)
        images.append(image)
        st.image(image, caption=uploaded_file.name, use_column_width=True)


if image_url:
    try:
        response = requests.get(image_url)
        
        
        if response.status_code == 200:
            
            content_type = response.headers.get("Content-Type")
            if "image" in content_type:
                image = Image.open(BytesIO(response.content))
                images.append(image)
                st.image(image, caption="Image from URL", use_column_width=True)
            else:
                st.error("The URL does not link to a valid image file.")
        else:
            st.error("Failed to fetch the image. Please check the URL.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


if images:
    st.write("### Crop Images")
    for idx, image in enumerate(images):
        st.write(f"Crop Image {idx + 1}")
        cropped_img = st_cropper(image, realtime_update=True, box_color='blue', aspect_ratio=None)
        cropped_images.append(cropped_img)

       
        st.image(cropped_img, caption=f"Cropped Image {idx + 1}", use_column_width=True)


if 'history' not in st.session_state:
    st.session_state['history'] = []


user_query = st.text_input("Ask a query related to the image:")


if user_query:
    st.write(f"Processing your query: {user_query}")
    
   
    if cropped_images:
        
        image_to_process = cropped_images[0]
    else:
        
        image_to_process = images[0]

   
    buffered = BytesIO()
    image_to_process.save(buffered, format="PNG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode()  

    
    try:
        response = lvlm.invoke(
            images=image_b64,  
            prompt=user_query  
        )

        
        st.session_state['history'].append({
            "query": user_query,
            "response": response
        })

        
        st.write("### Previous Queries and Responses")
        for idx, history_item in enumerate(st.session_state['history']):
            st.write(f"**Query {idx + 1}:** {history_item['query']}")
            st.write(f"**Response {idx + 1}:** {history_item['response']}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
