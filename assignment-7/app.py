import streamlit as st
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


# App UI
st.set_page_config(page_title="Drug Interaction Predictor", page_icon="💊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .header {
        font-size: 36px !important;
        font-weight: bold;
        color: #2b5876;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 20px !important;
        color: #4e4376;
        margin-bottom: 10px;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        background-color: #f0f2f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .severe {
        color: #e63946;
        font-weight: bold;
    }
    .moderate {
        color: #f4a261;
        font-weight: bold;
    }
    .safe {
        color: #2a9d8f;
        font-weight: bold;
    }
    .drug-select {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load the saved model
@st.cache_resource
def load_model():
    class DrugInteractionNN(torch.nn.Module):
        def __init__(self, num_drugs, embedding_dim=64, hidden_dim=128):
            super(DrugInteractionNN, self).__init__()
            self.drug_embedding = torch.nn.Embedding(num_drugs, embedding_dim)
            self.fc1 = torch.nn.Linear(embedding_dim * 2, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = torch.nn.Linear(hidden_dim // 2, 1)
            self.dropout = torch.nn.Dropout(0.3)
            self.sigmoid = torch.nn.Sigmoid()
            
        def forward(self, x):
            drug1 = x[:, 0]
            drug2 = x[:, 1]
            drug1_embed = self.drug_embedding(drug1)
            drug2_embed = self.drug_embedding(drug2)
            combined = torch.cat([drug1_embed, drug2_embed], dim=1)
            x = torch.relu(self.fc1(combined))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return self.sigmoid(x)

    # You'll need to replace this with your actual number of drugs
    model = DrugInteractionNN(num_drugs=1701)  # Temporary placeholder
    model.load_state_dict(torch.load('model/drug_interaction_model.pth'))
    model.eval()
    return model

# Load drug encoder (you'll need to save this during training)
@st.cache_resource
def load_encoder():
    # Create a dummy encoder - replace with your actual saved encoder
    drugs = ["Digoxin", "Verteporfin", "Paclitaxel", "Aminolevulinic acid", 
             "Amphotericin B", "Docetaxel", "Cyclophosphamide", "Carboplatin"]
    encoder = LabelEncoder()
    encoder.fit(drugs)
    return encoder

model = load_model()
encoder = load_encoder()



# Header
st.markdown('<div class="header">💊 Drug Interaction Predictor</div>', unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=80)
    st.markdown("### About")
    st.info("""
    This tool predicts potential interactions between medications.
    Select two drugs below to assess their interaction risk.
    """)
    
    st.markdown("### Sample Drugs")
    st.code(", ".join(encoder.classes_[:8]))

# Main content
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="subheader">Select Medications</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="drug-select">', unsafe_allow_html=True)
        drug1 = st.selectbox("First Drug", encoder.classes_, index=0, key="drug1")
        drug2 = st.selectbox("Second Drug", encoder.classes_, index=1, key="drug2")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Check Interaction", use_container_width=True):
        # Encode drugs
        try:
            drug1_enc = encoder.transform([drug1])[0]
            drug2_enc = encoder.transform([drug2])[0]
            
            # Make prediction
            with torch.no_grad():
                input_tensor = torch.tensor([[drug1_enc, drug2_enc]], dtype=torch.long)
                prediction = model(input_tensor).item()
            
            # Display results
            with col2:
                st.markdown('<div class="subheader">Interaction Analysis</div>', unsafe_allow_html=True)
                
                with st.container():
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    
                    # Risk level
                    if prediction > 0.7:
                        risk_level = '<span class="severe">HIGH RISK</span>'
                        recommendation = "🚨 Avoid combination. Consult healthcare provider immediately."
                    elif prediction > 0.4:
                        risk_level = '<span class="moderate">MODERATE RISK</span>'
                        recommendation = "⚠️ Use with caution. Monitoring may be required."
                    else:
                        risk_level = '<span class="safe">LOW RISK</span>'
                        recommendation = "✅ Generally safe. Normal monitoring recommended."
                    
                    st.markdown(f"""
                    **Drug Pair**: {drug1} + {drug2}  
                    **Interaction Probability**: {prediction:.1%}  
                    **Risk Level**: {risk_level}  
                    **Recommendation**: {recommendation}
                    """, unsafe_allow_html=True)
                    
                    # Visual gauge
                    st.progress(float(prediction))
                    st.caption(f"Interaction risk score: {prediction:.3f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional info expander
                    with st.expander("📊 Detailed Analysis"):
                        st.markdown(f"""
                        - **Mechanism**: Predicted pharmacodynamic interaction
                        - **Severity Score**: {prediction:.3f} (0-1 scale)
                        - **Confidence**: {(min(prediction, 1-prediction)*200):.1f}%
                        """)
                        
                        # Placeholder for hypothetical interaction effects
                        effects = {
                            "Digoxin + Amphotericin B": "May increase risk of cardiac arrhythmias",
                            "Verteporfin + Paclitaxel": "May increase photosensitivity reactions",
                            "Aminolevulinic acid + Cyclophosphamide": "May reduce therapeutic effects"
                        }
                        
                        combo = f"{drug1} + {drug2}"
                        if combo in effects:
                            st.warning(f"⚠️ Potential Effect: {effects[combo]}")
                        else:
                            st.info("ℹ️ No specific interaction effects documented in knowledge base")
        
        except Exception as e:
            st.error(f"Error processing drugs: {str(e)}")

# Footer
st.markdown("---")
st.caption("""
⚠️ This tool is for informational purposes only. Always consult a healthcare professional 
before making changes to medication regimens.
""")

# Sample data table (optional)
if st.checkbox("Show sample interaction data"):
    sample_data = pd.DataFrame({
        "Drug A": ["Digoxin", "Verteporfin", "Paclitaxel"],
        "Drug B": ["Amphotericin B", "Aminolevulinic acid", "Docetaxel"],
        "Risk Score": [0.82, 0.45, 0.63],
        "Severity": ["High", "Moderate", "High"]
    })
    st.dataframe(sample_data, hide_index=True)