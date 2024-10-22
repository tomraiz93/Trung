import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the datasets with semicolon delimiter
combined_df = pd.read_csv("combined_data3.csv", delimiter=";")
hunonic_df = pd.read_csv("hunonic8.csv", delimiter=";")

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
    merged_df = pd.merge(combined, hunonic, on='Tên sản phẩm', suffixes=('_combined', '_hunoic'))
    filtered_df = merged_df[merged_df['Giá hiện tại_combined'] < merged_df['Giá hiện tại_hunoic']]
    return filtered_df

# Show dataset based on selection
if dataset_option == "Hiển thị Data tổng hợp":
    st.write("Dữ liệu Tổng hợp:")
    st.dataframe(combined_df)

    if st.button("So sánh với Hunonic"):
        filtered_df = highlight_price_difference(combined_df, hunonic_df)
        
        st.write("Các sản phẩm có giá thị trường tổng hợp thấp hơn Hunonic:")
        st.dataframe(filtered_df)

        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Tải xuống CSV",
            data=csv,
            file_name='highlighted_products.csv',
            mime='text/csv',
        )

        # Tạo biểu đồ so sánh giá giữa Thị trường và Hunonic bằng Matplotlib
        st.write("### Biểu đồ so sánh giá giữa Thị trường và Hunonic")
        bar_chart_df = pd.merge(combined_df, hunonic_df, on='Tên sản phẩm', suffixes=('_thitruong', '_hunonic'))

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        index = range(len(bar_chart_df))
        ax.bar(index, bar_chart_df['Giá hiện tại_thitruong'], bar_width, label='Giá Thị Trường', color='red')
        ax.bar([i + bar_width for i in index], bar_chart_df['Giá hiện tại_hunonic'], bar_width, label='Giá Hunonic', color='blue')
        
        ax.set_xlabel('Tên sản phẩm')
        ax.set_ylabel('Giá hiện tại')
        ax.set_title('So sánh giá hiện tại giữa Thị trường và Hunonic')
        ax.set_xticks([i + bar_width / 2 for i in index])
        ax.set_xticklabels(bar_chart_df['Tên sản phẩm'], rotation=90)
        ax.legend()

        st.pyplot(fig)

        # Box plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=[bar_chart_df['Giá hiện tại_thitruong'], bar_chart_df['Giá hiện tại_hunonic']], ax=ax)
        ax.set_xticklabels(['Giá Thị Trường', 'Giá Hunonic'])
        ax.set_ylabel('Giá hiện tại')
        ax.set_title('Phân phối giá giữa Thị trường và Hunonic')
        st.pyplot(fig)

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
