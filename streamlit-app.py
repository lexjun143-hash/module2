import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from snowflake.snowpark.context import get_active_session

# Get Snowflake session
session = get_active_session()

# Load data from Snowflake
@st.cache_data
def load_data(_session):
    """Load reviews_with_sentiment data from Snowflake"""
    query = """
    SELECT 
        ORDER_ID,
        FILENAME,
        PRODUCT,
        REVIEW_DATE,
        SHIPPING_DATE,
        CARRIER,
        TRACKING_NUMBER,
        LATITUDE,
        LONGITUDE,
        STATUS,
        DELIVERY_DAYS,
        LATE,
        REGION,
        REVIEW_TEXT,
        SENTIMENT_SCORE
    FROM M2.SCHEMA.REVIEWS_WITH_SENTIMENT
    """
    df = _session.sql(query).to_pandas()
    return df

# Load the dataset
df = load_data(session)

# Main title
st.title("📦 Avalanche Product Dashboard")
st.markdown("### Analyze Customer Sentiment and Shipping Performance")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Product filter
products = ['All'] + sorted(df['PRODUCT'].dropna().unique().tolist())
selected_product = st.sidebar.multiselect(
    "Select Product(s)",
    products,
    default=['All']
)

# Region filter
regions = ['All'] + sorted(df['REGION'].dropna().unique().tolist())
selected_region = st.sidebar.multiselect(
    "Select Region(s)",
    regions,
    default=['All']
)

# Carrier filter
carriers = ['All'] + sorted(df['CARRIER'].dropna().unique().tolist())
selected_carrier = st.sidebar.multiselect(
    "Select Carrier(s)",
    carriers,
    default=['All']
)

# Delivery status filter
delivery_statuses = ['All'] + sorted(df['STATUS'].dropna().unique().tolist())
selected_status = st.sidebar.multiselect(
    "Select Delivery Status",
    delivery_statuses,
    default=['All']
)

# Apply filters
filtered_df = df.copy()

if 'All' not in selected_product:
    filtered_df = filtered_df[filtered_df['PRODUCT'].isin(selected_product)]

if 'All' not in selected_region:
    filtered_df = filtered_df[filtered_df['REGION'].isin(selected_region)]

if 'All' not in selected_carrier:
    filtered_df = filtered_df[filtered_df['CARRIER'].isin(selected_carrier)]

if 'All' not in selected_status:
    filtered_df = filtered_df[filtered_df['STATUS'].isin(selected_status)]

# Display key metrics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Reviews", len(filtered_df))

with col2:
    avg_sentiment = filtered_df['SENTIMENT_SCORE'].mean()
    st.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}")

with col3:
    late_percentage = (filtered_df['LATE'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("Late Deliveries", f"{late_percentage:.1f}%")

with col4:
    avg_delivery_days = filtered_df['DELIVERY_DAYS'].mean()
    st.metric("Avg Delivery Days", f"{avg_delivery_days:.1f}")

# Data Preview
st.markdown("---")
st.subheader("📊 Data Preview")
st.dataframe(filtered_df.head(10), use_container_width=True)

# Visualize Sentiment by Region
st.markdown("---")
st.subheader("🌍 Average Sentiment Score by Region")

# Calculate average sentiment by region
sentiment_by_region = filtered_df.groupby('REGION')['SENTIMENT_SCORE'].mean().sort_values()

# Create matplotlib plot
fig, ax = plt.subplots(figsize=(10, 6))
sentiment_by_region.plot(kind='barh', ax=ax, color='skyblue')
ax.set_xlabel('Average Sentiment Score')
ax.set_ylabel('Region')
ax.set_title('Average Sentiment Score by Region')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='x', alpha=0.3)

st.pyplot(fig)

# Highlight Delivery Issues
st.markdown("---")
st.subheader("⚠️ Delivery Issues with Negative Sentiment")

# Filter for negative sentiment and delivery issues
negative_reviews = filtered_df[
    (filtered_df['SENTIMENT_SCORE'] < 0) & 
    (filtered_df['LATE'] == True)
]

# Group by region and product
issues_summary = negative_reviews.groupby(['REGION', 'PRODUCT']).agg({
    'ORDER_ID': 'count',
    'SENTIMENT_SCORE': 'mean',
    'DELIVERY_DAYS': 'mean',
    'STATUS': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
}).reset_index()

issues_summary.columns = ['Region', 'Product', 'Issue Count', 'Avg Sentiment', 'Avg Delivery Days', 'Most Common Status']
issues_summary = issues_summary.sort_values('Issue Count', ascending=False)

st.dataframe(issues_summary, use_container_width=True)

# Additional visualizations
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Sentiment Trends Over Time")
    filtered_df['REVIEW_DATE'] = pd.to_datetime(filtered_df['REVIEW_DATE'])
    sentiment_over_time = filtered_df.groupby(filtered_df['REVIEW_DATE'].dt.to_period('M'))['SENTIMENT_SCORE'].mean()
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sentiment_over_time.plot(ax=ax2, marker='o', linewidth=2, color='green')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Average Sentiment Score')
    ax2.set_title('Sentiment Trends Over Time')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

with col2:
    st.subheader("🚚 Carrier Performance")
    carrier_performance = filtered_df.groupby('CARRIER').agg({
        'SENTIMENT_SCORE': 'mean',
        'LATE': lambda x: (x.sum() / len(x) * 100)
    }).round(2)
    
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    carrier_performance['SENTIMENT_SCORE'].plot(kind='bar', ax=ax3, color='coral')
    ax3.set_xlabel('Carrier')
    ax3.set_ylabel('Average Sentiment Score')
    ax3.set_title('Carrier Performance by Sentiment')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    st.pyplot(fig3)

# Chatbot Assistant using Cortex (Optional - requires Cortex enabled)
st.markdown("---")

# Check if Cortex is available
try:
    test_query = "SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large2', 'test') as response"
    session.sql(test_query).collect()
    cortex_available = True
except:
    cortex_available = False

if cortex_available:
    st.subheader("💬 AI Assistant - Ask Questions About Your Data")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about customer sentiment or shipping performance..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        data_summary = f"""
        You are analyzing customer review and shipping data with the following summary:
        - Total reviews: {len(filtered_df)}
        - Average sentiment score: {filtered_df['SENTIMENT_SCORE'].mean():.2f}
        - Late delivery rate: {(filtered_df['LATE'].sum() / len(filtered_df) * 100):.1f}%
        - Average delivery days: {filtered_df['DELIVERY_DAYS'].mean():.1f}
        - Regions: {', '.join(filtered_df['REGION'].unique())}
        - Products: {', '.join(filtered_df['PRODUCT'].unique())}
        - Carriers: {', '.join(filtered_df['CARRIER'].unique())}
        
        Top issues by region:
        {issues_summary.head(5).to_string()}
        
        User question: {prompt}
        
        Provide a helpful, concise answer based on this data.
        """
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Use Snowflake Cortex via SQL
                    cortex_query = f"""
                    SELECT SNOWFLAKE.CORTEX.COMPLETE(
                        'mistral-large2',
                        '{data_summary.replace("'", "''")}'
                    ) as response
                    """
                    result = session.sql(cortex_query).collect()
                    response = result[0]['RESPONSE']
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    st.info("💡 **AI Chatbot Not Available**: Snowflake Cortex is not enabled for this account. Contact your Snowflake administrator to enable Cortex features.")

# Footer
st.markdown("---")
st.markdown("**Avalanche Product Dashboard** | Powered by Snowflake Cortex & Streamlit")