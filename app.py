# import streamlit as st
# import pandas as pd
# import time
# from datetime import datetime



# ts = time.time()
# date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
# timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

# df = pd.read_csv("Attendance/Attendance_" + date + ".csv")


# st.dataframe(df.style.highlight_max(axis=0))



import streamlit as st
import pandas as pd
import time 
import plotly.express as px 
import os
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Attendance Analytics", layout="wide")

# --- 2. HELPER FUNCTIONS ---
def load_data(date_str):
    """Loads CSV for a specific date."""
    file_path = f"Attendance/Attendance_{date_str}.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def load_all_data():
    """Loads ALL CSV files to find student history."""
    all_files = [f for f in os.listdir("Attendance") if f.endswith('.csv')]
    all_data = []
    for file in all_files:
        df = pd.read_csv(f"Attendance/{file}")
        # Extract date from filename (Attendance_11-01-2026.csv)
        date_str = file.replace("Attendance_", "").replace(".csv", "")
        df['Date'] = date_str
        all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# --- 3. SIDEBAR ---
st.sidebar.title("🛠️ Control Panel")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2921/2921222.png", width=100) # Optional Icon

# Feature: Auto Refresh
auto_refresh = st.sidebar.checkbox("🔴 Live Auto-Refresh (2s)", value=False)
if auto_refresh:
    time.sleep(2)
    st.rerun()

# Feature: Date Picker
ts = time.time()
default_date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
selected_date = st.sidebar.text_input("📅 View Date (DD-MM-YYYY)", default_date)

# --- 4. MAIN LAYOUT (TABS) ---
st.title("📷 Smart Attendance Dashboard")
tab1, tab2, tab3 = st.tabs(["📋 Daily View", "📈 Analytics", "🔍 Student History"])

# --- TAB 1: DAILY VIEW ---
with tab1:
    st.header(f"Attendance for {selected_date}")
    df_today = load_data(selected_date)
    
    if df_today is not None and not df_today.empty:
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Present", len(df_today))
        col2.metric("Last Entry", df_today.iloc[-1]['TIME'])
        col3.metric("First Entry", df_today.iloc[0]['TIME'])
        
        # Style the table
        st.dataframe(df_today.style.background_gradient(cmap="Blues"), use_container_width=True)
        
        # Download
        csv = df_today.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report", csv, f"Attendance_{selected_date}.csv", "text/csv")
    else:
        st.info(f"No records found for {selected_date}. Class hasn't started yet?")

# --- TAB 2: ANALYTICS (GRAPHS) ---
with tab2:
    st.header("📊 Attendance Trends")
    if df_today is not None and not df_today.empty:
        
        # Logic: Extract Hour from Time (e.g., "09:15:00" -> "09")
        try:
            df_today['Hour'] = pd.to_datetime(df_today['TIME'], format='%H:%M-%S').dt.hour
            
            # Simple Bar Chart
            arrival_counts = df_today['Hour'].value_counts().sort_index()
            st.bar_chart(arrival_counts)
            st.caption("This graph shows how many students arrived during each hour.")
            
        except Exception as e:
            st.warning("Could not generate graph. Time format might be different.")
    else:
        st.warning("No data available to plot.")

# --- TAB 3: STUDENT SEARCH (HISTORY) ---
with tab3:
    st.header("🔍 Search Student History")
    
    # Load HUGE dataset (all files)
    all_df = load_all_data()
    
    if not all_df.empty:
        # Get unique student names
        student_list = all_df['NAME'].unique().tolist()
        selected_student = st.selectbox("Select a Student", student_list)
        
        if selected_student:
            # Filter data for that student
            student_history = all_df[all_df['NAME'] == selected_student]
            
            st.write(f"Showing history for **{selected_student}**:")
            st.dataframe(student_history[['Date', 'TIME', 'NAME']], use_container_width=True)
            
            st.success(f"{selected_student} has attended {len(student_history)} classes in total.")
    else:
        st.error("No attendance records found in the system yet.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Pratik")

