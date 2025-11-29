# Skin-Cancer Early check 

This app loads a TensorFlow skin-lesion classifier model **directly from Google Drive** and provides:
- Image upload
- Prediction
- Grad-CAM visualization

## Deployment
Upload this folder to a GitHub repository, then deploy using Streamlit Cloud:

https://share.streamlit.io

## Model Setup
Place your `.keras` model file in Google Drive and make it public.

Extract its FILE_ID and update `app.py`:

```python
FILE_ID = "YOUR_FILE_ID_HERE"
