import pandas as pd

def show_df(data):
    # Nếu truyền vào là Series, chuyển nó thành DataFrame để dùng được .style
    if isinstance(data, pd.Series):
        data = data.to_frame()
        
    return data.style.set_table_styles([
        {'selector': 'th, td', 'props': [
            ('text-align', 'left'), 
            ('white-space', 'normal'),
            ('max-width', '800px')
        ]}
    ]).set_properties(**{'text-align': 'left'})

def print_header(title):
    # Tạo chuỗi dấu bằng dài bằng độ dài tiêu đề cộng thêm khoảng đệm
    border = "=" * 55
    print("\n" + border)
    print(f" {title}")
    print(border + "\n")