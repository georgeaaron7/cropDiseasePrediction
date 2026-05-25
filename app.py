import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import pandas as pd

# cnn architecture
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

# the 15 classes from the dataset
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

NUM_CLASSES = len(CLASS_NAMES)

# loading the models
@st.cache_resource 
def load_cnn():
    model = CustomCNN(NUM_CLASSES)
    model.load_state_dict(torch.load('cnn_plant_disease.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource 
def load_mobilenet():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load('mobilenetv2_plant_disease.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

cnn_model = load_cnn()
mobilenet_model = load_mobilenet()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(model, tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0] * 100 # percentages
        
    # gte top 3 predictions for the bar chart
    top_prob, top_catid = torch.topk(probabilities, 3)
    
    results = {}
    for i in range(top_prob.size(0)):
        results[CLASS_NAMES[top_catid[i]]] = top_prob[i].item()
        
    best_class = CLASS_NAMES[top_catid[0]]
    best_conf = top_prob[0].item()
    
    return best_class, best_conf, results

# streamlit ui stuff
st.set_page_config(layout="wide") 
st.title("🌿 AI Crop Disease Detection Engine")
st.write("Compare the custom CNN against Transfer Learning (MobileNetV2).")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.subheader("1. Image Preprocessing")
    st.image(image, caption="Original Upload", width=300)
    input_tensor = transform(image).unsqueeze(0)
    st.code(f"Under the hood: Image converted to PyTorch Tensor of shape {input_tensor.shape}\n(Batch Size, Channels, Height, Width)", language="python")
    
    st.markdown("---")
    st.subheader("2. Model Inference & Comparison")

    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🧠 Custom CNN")
        cnn_class, cnn_conf, cnn_results = predict(cnn_model, input_tensor)
        
        if cnn_conf > 60.0:
            st.success(f"**Diagnosis:** {cnn_class.replace('_', ' ')}")
        else:
            st.warning(f"**Diagnosis:** {cnn_class.replace('_', ' ')}")
            st.error("Low confidence. Consult an expert.")
            
        st.metric(label="Confidence", value=f"{cnn_conf:.2f}%")
        
        st.write("**Top 3 Probabilities:**")
        chart_data = pd.DataFrame.from_dict(cnn_results, orient='index', columns=['Probability'])
        st.bar_chart(chart_data)

    # transfer learning
    with col2:
        st.header("⚡ MobileNetV2 (Transfer Learning)")
        mn_class, mn_conf, mn_results = predict(mobilenet_model, input_tensor)
        
        if mn_conf > 60.0:
            st.success(f"**Diagnosis:** {mn_class.replace('_', ' ')}")
        else:
            st.warning(f"**Diagnosis:** {mn_class.replace('_', ' ')}")
            st.error("Low confidence. Consult an expert.")
            
        st.metric(label="Confidence", value=f"{mn_conf:.2f}%")
        
        st.write("**Top 3 Probabilities:**")
        chart_data2 = pd.DataFrame.from_dict(mn_results, orient='index', columns=['Probability'])
        st.bar_chart(chart_data2)