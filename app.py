import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from utils.preprocessing import create_features, prepare_input_features, encode_education, create_marital_status_dummies

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Retail Buyer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODELS AND DATA
# ============================================
@st.cache_resource
def load_models():
    """Load all saved models and configuration files"""
    try:
        model = joblib.load('models/logistic_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_info = joblib.load('models/feature_names.pkl')
        
        with open('data/cluster_info.json', 'r') as f:
            cluster_info = json.load(f)
        
        return model, scaler, feature_info, cluster_info
    except FileNotFoundError as e:
        st.error(f"""
        ⚠️ **Missing model files!** 
        
        Please run the export cell in your notebook first:
        1. Open RetailBuyerSegmentation.ipynb
        2. Run the model export cell at the end
        3. Verify that models/ and data/ folders are created
        
        Error: {str(e)}
        """)
        st.stop()

model, scaler, feature_info, cluster_info = load_models()

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    
    /* AGGRESSIVE METRIC LABEL FIX */
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.75rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word !important;
        line-height: 1.1 !important;
        height: auto !important;
        max-height: none !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        white-space: normal !important;
        overflow: visible !important;
        height: auto !important;
        min-height: 2.8rem !important;
    }
    
    /* Adjust metric value */
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        padding-top: 0.3rem !important;
    }
    
    /* Give metrics more breathing room */
    div[data-testid="metric-container"] {
        padding: 0.8rem 0.3rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# APP TITLE AND DESCRIPTION
# ============================================
st.markdown('<p class="main-header">🛒 Retail Buyer Segmentation System</p>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p>AI-powered customer segmentation with <strong>97.8% accuracy</strong>. 
    Predict customer segments based on purchasing behavior and demographics.</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - INPUT METHOD SELECTION
# ============================================
st.sidebar.header("📊 Configuration")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["🧑 Single Customer (Manual)", "📁 Batch Upload (CSV)"],
    help="Enter data manually or upload a CSV file with multiple customers"
)

# ============================================
# SINGLE CUSTOMER INPUT
# ============================================
if input_method == "🧑 Single Customer (Manual)":
    st.header("📊 Enter Customer Information")
    
    # Create three columns for input fields
    col1, col2, col3 = st.columns(3)
    
    # ========================================
    # COLUMN 1: Demographics
    # ========================================
    with col1:
        st.subheader("👤 Demographics")
        birth_year = st.number_input(
            "Birth Year", 
            min_value=1940, 
            max_value=2005, 
            value=1980,
            help="Customer's year of birth"
        )
        education_level = st.selectbox(
            "Education Level",
            ["Graduation", "PhD", "Master", "2n Cycle", "Basic"],
            help="Highest education level completed"
        )
        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Together", "Single", "Divorced", "Widow", "Other"],
            help="Current marital status"
        )
        annual_income = st.number_input(
            "Annual Income ($)", 
            min_value=0, 
            max_value=200000, 
            value=50000, 
            step=1000,
            help="Annual household income in dollars"
        )
        num_children = st.number_input(
            "Number of Children", 
            min_value=0, 
            max_value=5, 
            value=0,
            help="Number of children in household"
        )
        num_teenagers = st.number_input(
            "Number of Teenagers", 
            min_value=0, 
            max_value=5, 
            value=0,
            help="Number of teenagers in household"
        )
    
    # ========================================
    # COLUMN 2: Shopping Behavior
    # ========================================
    with col2:
        st.subheader("🛍️ Shopping Behavior")
        spend_wine = st.number_input(
            "Wine Spending ($)", 
            min_value=0, 
            max_value=2000, 
            value=300,
            help="Amount spent on wine products"
        )
        spend_fruits = st.number_input(
            "Fruits Spending ($)", 
            min_value=0, 
            max_value=200, 
            value=25,
            help="Amount spent on fruit products"
        )
        spend_meat = st.number_input(
            "Meat Spending ($)", 
            min_value=0, 
            max_value=2000, 
            value=150,
            help="Amount spent on meat products"
        )
        spend_fish = st.number_input(
            "Fish Spending ($)", 
            min_value=0, 
            max_value=300, 
            value=40,
            help="Amount spent on fish products"
        )
        spend_sweets = st.number_input(
            "Sweets Spending ($)", 
            min_value=0, 
            max_value=300, 
            value=25,
            help="Amount spent on sweet products"
        )
        spend_gold = st.number_input(
            "Gold Products Spending ($)", 
            min_value=0, 
            max_value=400, 
            value=45,
            help="Amount spent on premium/gold products"
        )
    
    # ========================================
    # COLUMN 3: Purchase Channels & Engagement
    # ========================================
    with col3:
        st.subheader("📱 Purchase Channels")
        num_web_purchases = st.number_input(
            "Web Purchases", 
            min_value=0, 
            max_value=30, 
            value=4,
            help="Number of purchases made through website"
        )
        num_catalog_purchases = st.number_input(
            "Catalog Purchases", 
            min_value=0, 
            max_value=30, 
            value=3,
            help="Number of purchases made through catalog"
        )
        num_store_purchases = st.number_input(
            "Store Purchases", 
            min_value=0, 
            max_value=15, 
            value=6,
            help="Number of in-store purchases"
        )
        num_discount_purchases = st.number_input(
            "Discount Purchases", 
            min_value=0, 
            max_value=20, 
            value=2,
            help="Number of purchases with discounts"
        )
        web_visits_last_month = st.number_input(
            "Web Visits Last Month", 
            min_value=0, 
            max_value=25, 
            value=5,
            help="Number of website visits in the last month"
        )
        
        st.subheader("📢 Campaign Engagement")
        accepted_campaign_1 = st.checkbox("Accepted Campaign 1")
        accepted_campaign_2 = st.checkbox("Accepted Campaign 2")
        accepted_campaign_3 = st.checkbox("Accepted Campaign 3")
        accepted_campaign_4 = st.checkbox("Accepted Campaign 4")
        accepted_campaign_5 = st.checkbox("Accepted Campaign 5")
        accepted_last_campaign = st.checkbox("Accepted Last Campaign")
    
    # ========================================
    # PREDICT BUTTON
    # ========================================
    st.markdown("---")
    predict_button = st.button("🔍 Predict Customer Segment", type="primary", width='stretch')
    
    if predict_button:
        with st.spinner("🤖 Analyzing customer profile..."):
            # Create input dataframe with exact column names from data.csv
            input_data = pd.DataFrame({
                'birth_year': [birth_year],
                'annual_income': [annual_income],
                'num_children': [num_children],
                'num_teenagers': [num_teenagers],
                'spend_wine': [spend_wine],
                'spend_fruits': [spend_fruits],
                'spend_meat': [spend_meat],
                'spend_fish': [spend_fish],
                'spend_sweets': [spend_sweets],
                'spend_gold': [spend_gold],
                'num_web_purchases': [num_web_purchases],
                'num_catalog_purchases': [num_catalog_purchases],
                'num_store_purchases': [num_store_purchases],
                'num_discount_purchases': [num_discount_purchases],
                'web_visits_last_month': [web_visits_last_month],
                'accepted_campaign_1': [int(accepted_campaign_1)],
                'accepted_campaign_2': [int(accepted_campaign_2)],
                'accepted_campaign_3': [int(accepted_campaign_3)],
                'accepted_campaign_4': [int(accepted_campaign_4)],
                'accepted_campaign_5': [int(accepted_campaign_5)],
                'accepted_last_campaign': [int(accepted_last_campaign)],
                'days_since_last_purchase': [50],  # Default value
                'has_recent_complaint': [0],  # Default value
                'education_level': [education_level],
                'marital_status': [marital_status]
            })
            
            # Engineer features
            input_data = create_features(input_data)
            
            # Add encoded education
            input_data['education_encoded'] = input_data['education_level'].apply(encode_education)
            
            # Add marital status dummies
            input_data = create_marital_status_dummies(input_data)
            
            # Prepare for prediction (now returns DataFrame)
            X_input = prepare_input_features(input_data, feature_info['numeric_cols'])
            X_scaled = scaler.transform(X_input)
            
            # Make prediction
            # model.predict() returns the ACTUAL cluster ID (0, 1, or 3)
            # NOT an index (0, 1, or 2)
            prediction = int(model.predict(X_scaled)[0])  # This is the actual cluster ID
            
            # Get probabilities (these ARE indexed 0, 1, 2)
            probabilities = model.predict_proba(X_scaled)[0]
            
            # Find which index corresponds to this cluster ID
            model_classes = feature_info.get('model_classes', list(feature_info['cluster_names'].keys()))
            predicted_class_idx = list(model_classes).index(prediction)  # Convert cluster ID to index
            
            # Display results
            st.success("✅ Prediction Complete!")
            st.markdown("---")
            
            # Get cluster information
            cluster_name = feature_info['cluster_names'][prediction]
            cluster_data = cluster_info[str(prediction)]
            
            # ========================================
            # RESULTS DISPLAY
            # ========================================
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🎯 Predicted Segment", 
                    cluster_name,
                    help="The customer segment this profile belongs to"
                )
            
            with col2:
                st.metric(
                    "✨ Confidence", 
                    f"{probabilities[predicted_class_idx]*100:.1f}%",
                    help="Model's confidence in this prediction"
                )
            
            with col3:
                st.metric(
                    "👥 Segment Size", 
                    f"{cluster_data['size']:,} customers",
                    help="Total customers in this segment"
                )
            
            with col4:
                st.metric(
                    "💰 Avg Segment Spend", 
                    f"${cluster_data['avg_spend']:,.0f}",
                    help="Average spending of customers in this segment"
                )
            
            # ========================================
            # PROBABILITY DISTRIBUTION CHART
            # ========================================
            st.markdown("---")
            st.subheader("📈 Prediction Confidence Distribution")
            
            # Build probability dataframe using actual cluster IDs from model_classes
            model_classes = feature_info.get('model_classes', list(feature_info['cluster_names'].keys()))
            prob_data = []
            
            for idx, prob in enumerate(probabilities):
                cluster_id = model_classes[idx]  # Get actual cluster ID from index
                cluster_name_item = feature_info['cluster_names'][cluster_id]
                prob_data.append({
                    'Segment': cluster_name_item,
                    'Probability': prob * 100
                })
            
            prob_df = pd.DataFrame(prob_data).sort_values('Probability', ascending=False)
            
            fig = px.bar(
                prob_df, 
                x='Segment', 
                y='Probability',
                title='Probability Distribution Across All Segments',
                labels={'Probability': 'Probability (%)'},
                color='Probability',
                color_continuous_scale='Blues',
                text='Probability'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_title="Customer Segment",
                yaxis_title="Probability (%)"
            )
            st.plotly_chart(fig, width='stretch')
            
            # ========================================
            # CUSTOMER PROFILE SUMMARY
            # ========================================
            st.markdown("---")
            st.subheader("👤 Customer Profile Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Demographics:**")
                st.write(f"• Age: {2014 - birth_year} years old")
                st.write(f"• Education: {education_level}")
                st.write(f"• Marital Status: {marital_status}")
                st.write(f"• Annual Income: ${annual_income:,}")
                st.write(f"• Family Size: {num_children + num_teenagers} dependents")
            
            with col2:
                st.markdown("**Shopping Behavior:**")
                total_spend = spend_wine + spend_fruits + spend_meat + spend_fish + spend_sweets + spend_gold
                total_purchases = num_web_purchases + num_catalog_purchases + num_store_purchases
                campaigns_accepted = sum([accepted_campaign_1, accepted_campaign_2, accepted_campaign_3, 
                                         accepted_campaign_4, accepted_campaign_5, accepted_last_campaign])
                
                st.write(f"• Total Spending: ${total_spend:,}")
                st.write(f"• Total Purchases: {total_purchases}")
                st.write(f"• Campaigns Accepted: {campaigns_accepted}/6")
                st.write(f"• Preferred Channel: {'Web' if num_web_purchases >= num_store_purchases else 'Store'}")

# ============================================
# BATCH UPLOAD PROCESSING
# ============================================
else:  # Batch Upload
    st.header("📁 Batch Customer Segmentation")
    
    st.markdown("""
    Upload a CSV file with customer data. The file should contain the following columns:
    - `birth_year`, `education_level`, `marital_status`, `annual_income`
    - `num_children`, `num_teenagers`, `spend_wine`, `spend_fruits`, `spend_meat`, 
      `spend_fish`, `spend_sweets`, `spend_gold`
    - `num_web_purchases`, `num_catalog_purchases`, `num_store_purchases`
    - `accepted_campaign_1` through `accepted_campaign_5`, `accepted_last_campaign`
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="File should match the format of data.csv"
    )
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded **{len(df_upload):,}** customers")
            
            # Show preview
            with st.expander("👀 Preview Uploaded Data (First 10 rows)"):
                st.dataframe(df_upload.head(10), width='stretch')
            
            # Process button
            if st.button("🚀 Process All Customers", type="primary", width='stretch'):
                with st.spinner("🤖 Processing customers... This may take a moment."):
                    # Engineer features
                    df_processed = create_features(df_upload)
                    
                    # Add encoded features
                    df_processed['education_encoded'] = df_processed['education_level'].apply(encode_education)
                    df_processed = create_marital_status_dummies(df_processed)
                    
                    # Prepare for prediction
                    X_batch = prepare_input_features(df_processed, feature_info['numeric_cols'])
                    X_scaled = scaler.transform(X_batch)
                    
                    # Make predictions
                    predictions = model.predict(X_scaled)  # These are actual cluster IDs (0, 1, or 3)
                    probabilities = model.predict_proba(X_scaled)
                    
                    # Get model classes for indexing
                    model_classes = feature_info.get('model_classes', list(feature_info['cluster_names'].keys()))
                    
                    # Get confidence scores (need to find the right index for each prediction)
                    confidence_scores = []
                    for i, pred in enumerate(predictions):
                        pred_idx = list(model_classes).index(pred)  # Find index of this cluster ID
                        confidence_scores.append(probabilities[i][pred_idx] * 100)
                    
                    # Add results to dataframe
                    df_upload['predicted_cluster'] = predictions
                    df_upload['cluster_name'] = [feature_info['cluster_names'][int(p)] for p in predictions]
                    df_upload['confidence'] = confidence_scores

                    # Display results
                    st.success("✅ Segmentation Complete!")
                    st.markdown("---")
                    
                    # ========================================
                    # SUMMARY METRICS
                    # ========================================
                    st.subheader("📊 Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Customers", f"{len(df_upload):,}")
                    
                    with col2:
                        st.metric("Avg Confidence", f"{df_upload['confidence'].mean():.1f}%")
                    
                    with col3:
                        most_common = df_upload['cluster_name'].mode()[0]
                        st.metric("Most Common Segment", most_common)
                    
                    with col4:
                        unique_segments = df_upload['cluster_name'].nunique()
                        st.metric("Unique Segments", unique_segments)
                    
                    # ========================================
                    # DISTRIBUTION CHARTS
                    # ========================================
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Segment Distribution")
                        segment_counts = df_upload['cluster_name'].value_counts()
                        
                        fig_pie = px.pie(
                            values=segment_counts.values,
                            names=segment_counts.index,
                            title='Customer Distribution Across Segments',
                            hole=0.3
                        )
                        st.plotly_chart(fig_pie, width='stretch')
                    
                    with col2:
                        st.subheader("📈 Confidence Distribution")
                        
                        fig_hist = px.histogram(
                            df_upload,
                            x='confidence',
                            nbins=30,
                            title='Distribution of Prediction Confidence',
                            labels={'confidence': 'Confidence (%)'},
                            color_discrete_sequence=['#1f77b4']
                        )
                        fig_hist.update_layout(showlegend=False)
                        st.plotly_chart(fig_hist, width='stretch')
                    
                    # ========================================
                    # DETAILED RESULTS TABLE
                    # ========================================
                    st.markdown("---")
                    st.subheader("📋 Detailed Results")
                    
                    # Select columns to display
                    display_cols = ['cluster_name', 'confidence']
                    if 'customer_id' in df_upload.columns:
                        display_cols.insert(0, 'customer_id')
                    if 'birth_year' in df_upload.columns:
                        display_cols.append('birth_year')
                    if 'annual_income' in df_upload.columns:
                        display_cols.append('annual_income')
                    
                    # Format confidence as percentage
                    display_df = df_upload[display_cols].copy()
                    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(display_df, width='stretch', height=400)
                    
                    # ========================================
                    # DOWNLOAD RESULTS
                    # ========================================
                    st.markdown("---")
                    st.subheader("📥 Download Results")
                    
                    csv = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Complete Results (CSV)",
                        data=csv,
                        file_name="segmentation_results.csv",
                        mime="text/csv",
                        width='stretch'
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV matches the expected format with correct column names.")

# ============================================
# SIDEBAR - SEGMENT INFORMATION
# ============================================
st.sidebar.markdown("---")
st.sidebar.header("📊 Segment Overview")

for cluster_id, info in sorted(cluster_info.items(), key=lambda x: int(x[0])):
    total_customers = sum([c['size'] for c in cluster_info.values()])
    percentage = (info['size'] / total_customers) * 100
    
    with st.sidebar.expander(f"🎯 {info['name']}"):
        st.write(f"**Size:** {info['size']:,} customers ({percentage:.1f}%)")
        st.write(f"**Avg Income:** ${info['avg_income']:,.0f}")
        st.write(f"**Avg Spend:** ${info['avg_spend']:,.0f}")

# ============================================
# FOOTER
# ============================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; font-size: 0.8rem; color: #666;'>
    <p>Retail Buyer Segmentation System</p>
    <p>Model Accuracy: 97.8%</p>
    <p>Built with Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)