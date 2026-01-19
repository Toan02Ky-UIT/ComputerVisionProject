import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import pandas as pd
import time


st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 3em;
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #ff3333;
            border-color: #ff3333;
        }
        div[data-testid="stImage"] img {
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


CLASS_NAMES = ["Auto Rickshaws", "Bikes", "Cars", "Motorcycles", "Planes", "Ships", "Trains"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "resnet50.pt"

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/traffic-jam.png", width=80)
    st.title("⚙️ Cấu hình")
    
    st.write("Điều chỉnh thông số mô hình:")
    CONF_THRESHOLD = st.slider("Ngưỡng độ tin cậy (Confidence)", 0.0, 1.0, 0.70, 0.05)
    
    st.info(f"""
    **Model:** ResNet50  
    **Device:** `{DEVICE.upper()}`  
    **Classes:** {NUM_CLASSES} loại phương tiện
    """)
    
    st.markdown("---")
    st.caption("Developed with Streamlit & PyTorch")


@st.cache_resource
def load_model(model_path: str):
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.toast("⚠️ Không tìm thấy file model. Đang dùng model rỗng để demo giao diện!", icon="⚠️")
        pass 
        
    model.to(DEVICE)
    model.eval()
    return model

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess_image(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

@torch.inference_mode()
def predict(model, image_tensor):
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top_prob, top_idx = torch.max(probs, dim=0)
    
    if top_prob.item() < CONF_THRESHOLD:
        pred_label = "Không biết"
        is_confident = False
    else:
        pred_label = CLASS_NAMES[top_idx.item()]
        is_confident = True
        
    return pred_label, top_prob.item(), probs.cpu().tolist(), is_confident


st.title("🚗 Nhận Diện Phương Tiện Giao Thông")
st.markdown("---")

model = load_model(MODEL_PATH)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("1. Upload Ảnh")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh gốc", use_container_width=True)
    else:
        st.info("Vui lòng tải ảnh lên từ máy tính.")
        st.markdown(
            """
            <div style="text-align:center; padding: 20px; border: 2px dashed #ccc; border-radius: 10px; color: #ccc;">
                Khu vực hiển thị ảnh
            </div>
            """, unsafe_allow_html=True
        )

with col2:
    st.subheader("2. Kết Quả Phân Tích")
    
    if uploaded_file and model:
        if st.button("🚀 Phân tích ngay"):
            with st.spinner("Đang xử lý hình ảnh..."):
                time.sleep(0.5) 
                
                img_tensor = preprocess_image(image)
                pred_label, conf, probs_full, is_confident = predict(model, img_tensor)
            
            result_container = st.container()
            
            with result_container:
                m1, m2 = st.columns(2)
                
                lbl_color = "normal" if is_confident else "off"
                
                with m1:
                    st.metric(label="Dự đoán", value=pred_label)
                with m2:
                    st.metric(label="Độ tin cậy", value=f"{conf:.2%}", delta_color=lbl_color)

                st.write("Độ tự tin:")
                bar_color = "#28a745" if is_confident else "#ffc107"
                st.progress(conf)
                
                if not is_confident:
                    st.warning(f"Độ tin cậy thấp hơn ngưỡng {CONF_THRESHOLD}. Có thể ảnh không rõ hoặc không thuộc các lớp đã học.")

                st.markdown("#### 📊 Chi tiết Độ tự tin:")
                
                chart_data = pd.DataFrame({
                    "Phương tiện": CLASS_NAMES,
                    "Độ tự tin": probs_full
                }).set_index("Phương tiện")
                
                st.bar_chart(chart_data, color="#ff4b4b", horizontal=True)

                with st.expander("Xem bảng số liệu chi tiết"):
                    st.dataframe(
                        chart_data.sort_values("Độ tự tin", ascending=False).style.format("{:.4f}"),
                        use_container_width=True
                    )

    elif not model:
        st.error("❌ Model chưa được load. Vui lòng kiểm tra file .pt")
    else:
        st.write("Kết quả sẽ hiển thị tại đây sau khi bạn upload ảnh.")