import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the datasets with semicolon delimiter
combined_df = pd.read_csv("C:/Users/FPT SHOP/Documents/Zalo Received Files/combined_data3.csv", delimiter=";")
hunonic_df = pd.read_csv("C:/Users/FPT SHOP/Documents/Zalo Received Files/hunonic8.csv", delimiter=";")

# Set up custom CSS for the app
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #ADD8E6;  /* Light blue background for the entire web app */
    }
    .main {
        background-color: #ADD8E6; /* Light blue background for the main content area */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green background for buttons */
        color: white;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Set up the Streamlit app layout
st.title("Kiểm tra giá sản phẩm")

# Sidebar: Search bar for product selection
st.sidebar.markdown("<h1 style='text-align: center; font-size: 30px;'>Tìm kiếm sản phẩm</h1>", unsafe_allow_html=True)
search_term = st.sidebar.text_input("Nhập tên sản phẩm:")

# Filter dataset based on search term
filtered_products = combined_df[combined_df['Tên sản phẩm'].str.contains(search_term, case=False, na=False)]
product_selected = st.sidebar.selectbox("Chọn sản phẩm", filtered_products['Tên sản phẩm']) if not filtered_products.empty else None

if product_selected:
    # Get selected product data from both combined and Hunonic datasets
    selected_combined = combined_df[combined_df['Tên sản phẩm'] == product_selected]
    selected_hunonic = hunonic_df[hunonic_df['Tên sản phẩm'] == product_selected]

    # Display the selected product's price from the combined dataset
    st.write(f"### Giá của sản phẩm {product_selected}:")
    combined_price = selected_combined['Giá hiện tại'].values[0]
    st.write(f"**Giá hiện tại (Tổng hợp):** {combined_price} VNĐ")

    # Check if the product exists in the Hunonic dataset
    if not selected_hunonic.empty:
        hunonic_price = selected_hunonic['Giá hiện tại'].values[0]
        st.write(f"**Giá hiện tại (Hunonic):** {hunonic_price} VNĐ")

        # Compare prices to determine if it's "Phá Giá"
        if combined_price < hunonic_price:
            st.write("**Kết quả: Phá Giá**")
        else:
            st.write("**Kết quả: Không Phá Giá**")
    else:
        st.write("**Sản phẩm không có trong dữ liệu Hunonic.**")

# Create buttons to display the two datasets
dataset_option = st.sidebar.selectbox("Chọn dataset", ["Hiển thị Data tổng hợp", "Hiển thị Hunonic Data"])

# Function to compare and highlight differences
def highlight_price_difference(combined, hunonic):
    # Merge data on 'Tên sản phẩm'
    merged_df = pd.merge(combined, hunonic, on='Tên sản phẩm', suffixes=('_combined', '_hunoic'))

    # Apply condition to highlight rows where combined current price is less than hunoic price
    def highlight(row):
        return ['background-color: lightgreen' if row['Giá hiện tại_combined'] < row['Giá hiện tại_hunoic'] else '' for _ in row]

    # Sort the highlighted rows first
    sorted_df = merged_df.sort_values(by=['Giá hiện tại_combined'], ascending=True)
    
    # Create a dataframe of only the filtered rows
    filtered_df = merged_df[merged_df['Giá hiện tại_combined'] < merged_df['Giá hiện tại_hunoic']]
    
    # Return both the styled dataframe and the filtered dataframe
    return sorted_df.style.apply(highlight, axis=1), filtered_df

# Show dataset based on selection
if dataset_option == "Hiển thị Data tổng hợp":
    st.write("Dữ liệu Tổng hợp:")
    st.dataframe(combined_df)

    # Add comparison button
    if st.button("So sánh với Hunonic"):
        # Perform the comparison and get both the highlighted and filtered dataframes
        highlighted_df, filtered_df = highlight_price_difference(combined_df, hunonic_df)
        
        st.write("Dữ liệu sau khi so sánh:")
        st.dataframe(highlighted_df)  # Display the highlighted dataframe

        st.write("Các sản phẩm có giá thị trường tổng hợp thấp hơn Hunonic:")  # Display the filtered dataframe below
        st.dataframe(filtered_df)

        # Optionally: You can provide a download button for the filtered dataframe
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Tải xuống CSV",
            data=csv,
            file_name='highlighted_products.csv',
            mime='text/csv',
        )

        # Create visualizations with Plotly
        st.write("### Biểu đồ so sánh giá giữa Thị trường và Hunonic")
        
        # Bar Chart for current prices
        bar_chart_df = pd.merge(combined_df, hunonic_df, on='Tên sản phẩm', suffixes=('_thitruong', '_hunonic'))
        fig_bar = px.bar(bar_chart_df, 
                         x="Tên sản phẩm", 
                         y=["Giá hiện tại_thitruong", "Giá hiện tại_hunonic"], 
                         title="So sánh giá hiện tại",
                         labels={"value": "Giá hiện tại", "variable": "Dataset"})
        st.plotly_chart(fig_bar)

        # Scatter Plot for price comparison
        scatter_df = pd.merge(combined_df, hunonic_df, on='Tên sản phẩm', suffixes=('_thitruong', '_hunonic'))
        fig_scatter = px.scatter(scatter_df, 
                                 x="Giá hiện tại_thitruong", 
                                 y="Giá hiện tại_hunonic", 
                                 color="Tên sản phẩm",
                                 title="Phân phối giá Thị Trường vs Hunonic",
                                 labels={"Giá hiện tại_thitruong": "Giá Thị Trường", "Giá hiện tại_hunonic": "Giá Hunonic"})
        st.plotly_chart(fig_scatter)
        
        # Box Plot for price distribution with custom colors
        fig_box = px.box(bar_chart_df, 
                 y=["Giá hiện tại_thitruong", "Giá hiện tại_hunonic"], 
                 title="Box plot phân phối giá giữa Thị trường và Hunonic",
                 labels={"value": "Giá hiện tại", "variable": "Loại giá"},
                 color_discrete_map={"Giá hiện tại_thitruong": "red", "Giá hiện tại_hunonic": "blue"})  # Use red and blue for contrast
        st.plotly_chart(fig_box)

        # Histogram for price frequency distribution with custom colors
        fig_hist = px.histogram(bar_chart_df, 
                        x=["Giá hiện tại_thitruong", "Giá hiện tại_hunonic"], 
                        title="Histogram tần suất giá Thị Trường và Hunonic",
                        labels={"value": "Giá hiện tại", "variable": "Loại giá"},
                        barmode='overlay',  # Overlay both histograms to compare easily
                        color_discrete_map={"Giá hiện tại_thitruong": "red", "Giá hiện tại_hunonic": "blue"})  # Red for Thị Trường, blue for Hunonic
        st.plotly_chart(fig_hist)

elif dataset_option == "Hiển thị Hunonic Data":
    st.write("Dữ liệu Hunonic:")
    st.dataframe(hunonic_df)

# Tiền xử lý dữ liệu cho mô hình dự đoán
def preprocess_data(df):
    df['Giá gốc'] = pd.to_numeric(df['Giá gốc'], errors='coerce')
    df['Giá hiện tại'] = pd.to_numeric(df['Giá hiện tại'], errors='coerce')
    df = df.dropna()  # Loại bỏ các giá trị thiếu
    return df

# Huấn luyện mô hình Linear Regression
combined_df = preprocess_data(combined_df)
X = combined_df[['Giá gốc']]
y = combined_df['Giá hiện tại']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Thêm chức năng dự đoán vào cuối
st.write("### Dự đoán giá hiện tại dựa trên giá gốc")
gia_goc_input = st.number_input("Nhập giá gốc:")
if st.button("Dự đoán giá hiện tại"):
    gia_du_doan = model.predict([[gia_goc_input]])[0]
    st.write(f"Giá hiện tại dự đoán: {gia_du_doan:.2f} VNĐ")
