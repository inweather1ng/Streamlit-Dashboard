
import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from datetime import date

# 1. Page Configuration (Must be first)
st.set_page_config(
    page_title="Folsom Lake Dashboard",
    page_icon="💧",
    layout="wide"
    
)




@st.cache_data
def load_data():
    # 1. Load and clean Reservoir data (2000 - 2026)
    reservoir = pd.read_csv('Raw_data/reservoir data.csv', thousands=',')
    for col in ['STORAGE AF', 'DC PUMP CFS', 'LK EVAP AF']:
        if col in reservoir.columns:
            reservoir[col] = pd.to_numeric(reservoir[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
    
    reservoir['DATE'] = pd.to_datetime(reservoir['DATE']).dt.date
    start_date = pd.to_datetime("2000-11-01").date()
    end_date = pd.to_datetime("2026-05-21").date()
    reservoir = reservoir[(reservoir['DATE'] > start_date) & (reservoir['DATE'] < end_date)]
    
    # 2. Load and clean Outflow data

    precip = pd.read_csv('Raw_data/rainfall.csv',skiprows=2)
    precip['Dates'] = pd.to_datetime(precip['Date'].astype(str), format='%Y%m',errors="coerce").dt.date
    precip = precip.rename(columns={"Value": "Precip"}).drop(columns=['Date'], errors='ignore')
    change_strings = reservoir['STORAGE AF'] = reservoir['STORAGE AF'].astype(str).str.replace(',','',regex=False)


    outflow = pd.read_csv("Raw_data/outflow.csv", thousands=",")
    outflow['VALUE'] = pd.to_numeric(outflow['VALUE'].astype(str).str.replace(",", "", regex=False), errors="coerce")
    outflow['Dates'] = pd.to_datetime(outflow['OBS DATE'], format='%Y%m%d %H%M', errors="coerce").dt.date
    outflow = outflow.rename(columns={"VALUE": "Outflow"}).drop(columns=['OBS DATE'], errors='ignore')
    
    # 3. Fetch and clean CDEC SWE data (Alpha Station)
    url_target = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=ALP&SensorNums=3&dur_code=D&Start=2000-11-01&End=2026-06-30"
    swedata = pd.read_csv(url_target)
    swedata['VALUE'] = pd.to_numeric(swedata['VALUE'], errors='coerce')
    swedata['Dates'] = pd.to_datetime(swedata['OBS DATE'], format='%Y%m%d %H%M', errors='coerce').dt.date
    swedata = swedata.rename(columns={"VALUE": "SWE_Value"}).drop(columns=['OBS DATE'], errors='ignore')

    # 4. Load and clean Inflow data (⚠️ Short timeline: 2020 - 2026)
    inflow = pd.read_csv("Raw_data/Folsom Lake Dam and Powerplant Observed Daily Average Lake_Reservoir Inflow cfs (2025-04-01 - 2026-05-21).csv", skiprows=7)
    
    inflow.columns = inflow.columns.str.strip()
    
    
    inflow['Dates'] = pd.to_datetime(inflow['Datetime (UTC)'], errors="coerce").dt.date
    inflow['Inflow_Result'] = pd.to_numeric(inflow['Result'], errors="coerce")
    
    start_inflow = pd.Timestamp('2020-01-01').date()
    end_inflow = pd.Timestamp('2026-05-21').date()
    inflow = inflow[(inflow['Dates'] > start_inflow) & (inflow['Dates'] < end_inflow)]
    inflow = inflow[['Dates', 'Inflow_Result']] # Keep only what we need
    
    # 5. Sequentially merge dataframes using LEFT joins to protect historical data
    # Standardize names for joining
    reservoir = reservoir.rename(columns={'DATE': 'Dates'})
    
    # Build master using left joins so 2000-2020 data isn't deleted by the shorter inflow dataset
    masterdata = pd.merge(reservoir, swedata, on='Dates', how='left')
    masterdata = pd.merge(masterdata, outflow, on='Dates', how='left')
    masterdata = pd.merge(masterdata, inflow, on='Dates', how='left')
    masterdata = pd.merge(masterdata,precip,on='Dates',how='left')
    
    # 6. Feature Engineering (Derived Date Columns)
    masterdata["CALYEARDATE"] = pd.to_datetime(masterdata["Dates"])
    masterdata["year"] = masterdata['CALYEARDATE'].dt.year
    
    masterdata["month"] = masterdata['CALYEARDATE'].dt.month
    
    # Calculate Water Year (Oct 1 - Sept 30)
    masterdata["water year"] = masterdata["year"]
    islatemonth = masterdata['month'].isin([10, 11, 12])
    masterdata.loc[islatemonth, 'water year'] = masterdata['year'] + 1

    return masterdata


df = load_data()
@st.cache_data
def convert(dataframeresult):
    return dataframeresult.to_csv(index=False).encode("utf-8")
csvdata = convert(df)
st.download_button(
    label="Download master data here",
    data=csvdata,
    file_name="dataframe.csv",
    mime="text/csv"

                   )
variable_options = {
    'Snow water equivalent (inches)': 'SWE_Value',
    "Reservoir Storage (Acre-Feet)": "STORAGE AF",
    "Inflow (Cubic feet/Second": "Inflow_Result",
    "Outflow (Cubic feet/Second": "DC PUMP CFS",
    "Precipitation (inches)":"Precip"
}
tab1,tab2,tab3,tab5,tab6,tab7,tab8 = st.tabs(["Analysis of all variables","Inflow Analysis","Evaporation analysis","Yearly Outflow","Yearly Inflow","Reservoir minimum and maximum","Precipitation yearly average"])
with tab1:
# --- Sidebar Controls (ONLY ONE CONFIGURATION BLOCK HERE) ---
    st.sidebar.header("Dashboard Controls")
    min_wy = int(df['water year'].min())
    max_wy = int(df['water year'].max())

    selected = st.sidebar.slider(
        "Select Water Year",
        min_value=min_wy,
        max_value=max_wy,
        value=max_wy
    )



    st.sidebar.markdown("---")
    st.sidebar.subheader("Axis Configuration")

    x_select = st.sidebar.selectbox(
        "Select X axis variable", 
        options=list(variable_options.keys()),
        index=0,
        key="x_axis_unique"
    )

    y_select = st.sidebar.selectbox(
        "Select Y axis variable",
        options=list(variable_options.keys()),
        index=1,
        key="y_axis_unique"
    )

# Convert friendly names to dataframe columns
    x_column = variable_options[x_select]
    y_column = variable_options[y_select]


    filtered_df = df[df['water year'] == selected]
    filtered_df[x_column] = pd.to_numeric(filtered_df[x_column], errors='coerce')
    filtered_df[y_column] = pd.to_numeric(filtered_df[y_column], errors='coerce')
    corlrelation = f"{filtered_df[x_column].corr(filtered_df[y_column]):.2f}"
    label = (f" With an r of {corlrelation}")
    # --- Main Page UI Layout ---
    st.title('Folsom Lake Water Dashboard')
    st.markdown(f"Viewing telemetry data for the **{selected}** water year")
    st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        max_swe = filtered_df['SWE_Value'].max()
        val_swe = f"{max_swe:.1f} in" if pd.notna(max_swe) else "No Data"
        st.metric(label="Peak Snow water equivalent", value=val_swe,)
    with c2:
        max_storage = filtered_df['STORAGE AF'].max()
        val_storage = f"{int(float(max_storage)):,} AF" if pd.notna(max_storage) else "No Data"
        st.metric(label="Peak Storage Volume", value=val_storage)
    with c3:
        max_inflow = filtered_df['Inflow_Result'].max()
        val_inflow = f"{int(max_inflow):,} cfs" if pd.notna(max_inflow) else "No Data"
        st.metric(label="Peak Daily Inflow Rate", value=val_inflow)
    with c4:
        max_outflow = filtered_df['DC PUMP CFS'].max()
        val_outflow = f"{int(max_outflow):,} cfs" if pd.notna(max_outflow) else "No Data"
        st.metric(label="Peak Outflow", value=val_outflow)
    st.write("---")
    st.subheader(f"How does {x_select} affect {y_select}?")

    if filtered_df.empty:
        st.warning(f"No concurrent records found for Water Year {selected}. Try switching the slider to 2025 or 2026.")
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.regplot(
        data=filtered_df,
        x=x_column,
        y=y_column,
        ax=ax1,
        color='#1f77b4',
        scatter_kws={'alpha': 0.6},
        line_kws={'color': 'red', 'linewidth': 2, 'label': 'trendline'},

    )
    ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
                                
    ax1.set_title(f"{x_select} vs. {y_select} in Water Year {selected}{label}", fontsize=12, fontweight='bold')
    ax1.set_ylabel(y_select)
    ax1.set_xlabel(x_select)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    st.pyplot(fig)
    st.write("Which variables had the best correlation?")
with tab2:
    st.header("Inflow Monthly Averages")
    averages = []
    months = []
    
    month = df['month'].tolist()
    months = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    df['month'] = pd.to_numeric(df['month'],errors="coerce").map(months)
    order=list(months.values())
    df['month'] = pd.Categorical(df['month'],categories=order,ordered=True)
    monthly_averages=df.groupby('month')['Inflow_Result'].mean().reset_index()
    monthly_averages.rename(columns={'Inflow_Result': 'Average Inflow'}, inplace=True)
    
    figure,theax=plt.subplots()
    st.subheader("Average inflow trends by month")
    sns.lineplot(
        data=monthly_averages,
        x='month',
        y='Average Inflow',
        ax=theax,
        label=("yearly average inflow")
        


    )
    theax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    theax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    theax.set_title("Inflow graph over by month")
    theax.set_xlabel("Month")
    theax.set_ylabel("Inflow trend (Cubic Feet/Second)")
    theax.grid(True)
    st.pyplot(figure)
    st.write("A very Interesting trend was detected in the monthly inflow trends. The wettest month is usually January or February. In the graph, you can see that the highest average inflow occured in January. Since inflow directly corresponds with precipitation runoff, an increase in inflow in january correlates clearly with the amount of precipitation in january. Following january, inflow decreases in the summer and fall. Due to the low snowpack, inflow does not gain rapidly in the winter until January. ")
with tab3:
    x=6
    
    st.header("Evaporation Exploration")
    months = {1:"January",2:"February",3:"MArch",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
    monthly_averages=df.groupby('month')['Inflow_Result'].mean().reset_index()
    monthly_averages.rename(columns={'Inflow_Result': 'Average Inflow'}, inplace=True)
    evaporation_monthly = df.groupby('month')['LK EVAP AF'].mean().reset_index()
    evaporation_monthly.rename(columns={'LK EVAP AF':'Average Evaporation'},inplace=True)
    figure2,theaxis=plt.subplots()
    st.subheader("evaporation trends")
    sns.lineplot(data=evaporation_monthly,x="month",y="Average Evaporation",ax=theaxis,label="Outflow average")
    theaxis.set_title("Evaporation by monthly averages")
    theaxis.set_xlabel("Month")
    theaxis.set_ylabel("Average Evaporation (Acre-Feet)")
    theaxis.grid(True)
    theaxis.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    theaxis.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    st.pyplot(figure2)
    st.write("Evaporation trends peak in the late summer and reach a bottom in the winter. In the summer, the monthly evaporation can increase up to 6 times the evaporation during winter. This trend seems directly associated with temperature trends, however it is a more severe reaction to summer weather. This makes evaporation an easy prediction metric for lake level studies relating to seasonal changes.")
with tab5:
    st.header("Year-Over-Year Outflow Trends")
    
    # FIX: Grouping by 'year' (derived from your 26-year master data timeline)
    outflow_yearly = df.groupby('year')['Outflow'].mean().reset_index()
        
    
    figure4, axis4 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=outflow_yearly, x='year', y='Outflow', ax=axis4, marker="s", color="green")
    axis4.set_title("Total Outflow Trends YOY (2021 - 2026)")
    axis4.set_xlabel("Year")
    axis4.set_ylabel("Total Outflow Volume(Cubic feet/Second)")
    axis4.grid(True, linestyle='--', alpha=0.5)
    axis4.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    axis4.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    st.pyplot(figure4)
    st.write("On a yearly basis, ouflow trends peak during rainy winters. Examples include the wet season of 2023. This serves as another accurate prediction metric for lake level as it has stable trends that match greatly with the peak rainfall events.")
with tab6:
        #st.header("Year-Over-Year Inflow Trends")
    inflow2 = pd.read_csv("Raw_data/inflow.csv")
    
    inflow2['VALUE'] = pd.to_numeric(inflow2['VALUE'], errors='coerce')
    inflow2['Dates'] = pd.to_datetime(inflow2['OBS DATE'], format='%Y%m%d %H%M', errors='coerce').dt.date
    inflow2['Dates'] = pd.to_datetime(inflow2['Dates'])
    inflow2['theyear'] = inflow2['Dates'].dt.year
    inflow2=inflow2.dropna()
    inflow2 = inflow2.rename(columns={"VALUE": "Inflow_Value"}).drop(columns=['OBS DATE'], errors='ignore')
    # FIX: Grouping by 'year' (derived from your 26-year master data timeline)
    inflow_yearly = inflow2.groupby('theyear')['Inflow_Value'].mean().reset_index()
    figure7,axis7=plt.subplots(figsize=(10,4))
    sns.lineplot(data=inflow_yearly,x='theyear',y='Inflow_Value')
    axis7.set_title("Inflow trends over year")
    axis7.set_xlabel("year")
    axis7.set_ylabel("Inflow (Cubic Feet/Second)")
    axis7.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    axis7.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    st.pyplot(figure7)
    st.write("Inflow values are heavily associated with storm runoff during the rainy season. Resultantly, inflow trends over a larger basis have peaked during major rain events in the central California area. For example, the rainy season of 2016 and 2017 marked the greatest inflow seen in the past 20 plus years. Inflow relates well with precipitation, but doesn't mirror lake level because it does not take into account the previous winter's precipitation. It is instead a way of detecting real-time trends instead.")   
    
        #figure4, axis4 = plt.subplots(figsize=(10, 4))
        #sns.lineplot(data=inflow_yearly, x='year', y='Inflow_Result', ax=axis4, marker="s", color="green")
        #axis4.set_title("Total Outflow Trends YOY (2000 - 2026)")
        # 
        #axis4.s#et_x#label("Year")
    
    axis4.grid(True, linestyle='--', alpha=0.5)
        #st.pyplot(figure4)
with tab7:
    st.header("lake level bottoming over time")
    idx_min = df.groupby("year")['STORAGE AF'].idxmin()
    
    reservoir_min = df.loc[idx_min, ['year','month','STORAGE AF']].reset_index(drop=True)
    st.dataframe(reservoir_min)

    figure5,axis5 = plt.subplots(figsize=(10,4))
    sns.lineplot(data=reservoir_min,x='year',y='STORAGE AF',ax=axis5)
    axis5.set_title("Reservoir level bottoming")
    axis5.set_xlabel("Year")
    axis5.set_ylabel("Min storage (Acre-Feet)")
    axis5.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    axis5.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    st.pyplot(figure5)
    st.write("Lake level bottoming over the past 5 years has been increasing as california has been moved out of the drought it was facing, reflected in the increase since 2021. The month of the bottomming has been january over the past 2 years, while it was reviously in the fall. This is an unexpected trend, as usally the lowest lake level is in the fall. It is important to note that the 2026 value will get lower as the year goes on.")
    
with tab8:
    st.header("Precipitation averages YOY")
    precipitation_averages = df.groupby("year")['Precip'].sum().reset_index()
    figure6,axis6 = plt.subplots(figsize=(10,4))
    sns.lineplot(data=precipitation_averages,x='year',y='Precip',ax=axis6)
    axis6.set_title("precipitation averages over time")
    axis6.set_xlabel("Precipitation average(inches)")
    axis6.set_ylabel("precipitation")
    axis6.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    axis6.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    st.pyplot(figure6)
    df_clean_precip = df.dropna(subset=['Precip'])
    st.write("Precipitation averages reached a bottom in 2022, and then epaked in 2023, reflecting the large rainfall event.")
# Now idxmin will not encounter all-NA groups (unless a year had zero valid records)
    idx_min = df_clean_precip.groupby("year")['Precip'].idxmin(skipna=True)
    precip_min = df_clean_precip.loc[idx_min, ['year', 'month', 'Precip']].reset_index(drop=True)

    st.dataframe(precip_min)
    st.write("Mimimum precipitation typically ocurs in July or August, as seen in the table.")

 

    

