import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
st.set_page_config(
    page_title="Unveiling VULnerability",
    layout="centered",
    page_icon=":brain:",  # Brain emoji for icon
)
#st.beta_set_background("/Users/zakariaelkhilani/Documents/Work/Vscode/streamlt/Image1.jpeg"  # Replace with your image path)

# Title text
st.markdown("<h1 style='text-align: center;'>Unveiling VULnerability: An Interactive Exploration</h1>", unsafe_allow_html=True)


# Introduction text
st.markdown(
    """
    Delve into the world of student VULnerability with this interactive data analysis tool. This application caters to both professionals and those interested in gaining insights into factors that may impact student well-being.

    **Explore Data from Multiple Sources:**

    - **Neuropsychologie Station:** Analyze data prepared by neuropsychologists, offering valuable perspectives on student cognitive function.
    - **Psychologie Station:** Gain insights from data compiled by psychologists, shedding light on students' mental and emotional well-being.
    - **Wellness Program:** Examine a comprehensive dataset merging information from both stations for a holistic view of student VULnerability.

    **Interactive Features:**

    - **Intuitive Data Upload:** Upload CSV files with ease and let the application guide you based on the chosen data source.
    - **Enhanced Data Exploration:** Employ various visualizations to uncover relationships and trends within the data.
    - **In-Depth Statistical Analysis:** Access key descriptive statistics and missing value information.
    - **Correlational Insights:** Identify potential connections between factors influencing VULnerability.
    """
)
# User selection for data analysis type


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


def update_cross_tab(variable):
    cross_tab = pd.crosstab(df[variable], df[last_column])

    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(cross_tab, annot=True, cmap='coolwarm', fmt="d", ax=ax)

    # Adjust font sizes for better readability
    ax.set_title(f'Cross-Tabulation of {variable} and {last_column}', fontsize=18)
    ax.set_xlabel('Cognitive Performances', fontsize=18)
    ax.set_ylabel(variable, fontsize=18)
    ax.tick_params(labelsize=12)  # Adjust tick label size

    st.pyplot(fig)

def update_cross_tab2(variable):
    if variable is not None:
        cross_tab = pd.crosstab(df[variable], df[last_column])

        fig, ax = plt.subplots(figsize=(18, 10))
        sns.countplot(data=df, x=variable, hue=last_column, palette='coolwarm')

        # Adjust font sizes and rotation for better readability
        plt.title(f'Cross-Tabulation of {variable} and {last_column}', fontsize=18)
        plt.xlabel(variable, fontsize=18)
        plt.ylabel('Count', fontsize=18)
        plt.legend(title=last_column, title_fontsize=16)  # Adjust legend title size
        plt.xticks(rotation=45)
        plt.tick_params(labelsize=12)  # Adjust tick label size

        st.pyplot(fig)

def update_cross_tab3(variable):
    fig, ax = plt.subplots(figsize=(18, 18))
    ax = sns.countplot(data=df, x=variable, palette='coolwarm')

    # Adjust font sizes for better readability
    plt.title(f'Frequency of {variable}', fontsize=18)
    plt.xlabel(variable, fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.tick_params(labelsize=15)  # Adjust tick label size

    total = len(df[variable])
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=15)

    st.pyplot(fig)

def plot_corr_matrix():
    plt.figure(figsize=(18, 18))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")

    # Adjust font sizes for better readability
    plt.title('Correlation Matrix', fontsize=16)
    ax = plt.gca()  # Get current axis
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

    st.pyplot(plt)  # Display the plot using Streamlit

# Function to plot target variable correlations
def plot_target_correlations():
    target_correlations = corr_matrix[last_column].drop(last_column)
    plt.figure(figsize=(10, 6))
    target_correlations.plot(kind='bar', color='skyblue')

    # Adjust font sizes for better readability
    plt.title(f'Correlations with {last_column}',fontsize=20 )
    plt.xlabel('Features', fontsize=20)
    plt.ylabel('Correlation',fontsize=20)
    plt.xticks(rotation=45)
    plt.tick_params(labelsize=13)  # Adjust tick label size

    st.pyplot(plt)  # Display the plot using Streamlit
# Function for neuropsychological analysis (replace with your actual implementation)
def find_strong_correlations(corr_matrix, threshold=0.2):
    # Find correlations greater than the threshold
    strong_correlations = corr_matrix[corr_matrix.abs() > threshold]

    # Drop NA values and the diagonal, as they are not relevant for plotting
    strong_correlations = strong_correlations.dropna(axis=0, how='all').dropna(axis=1, how='all')

    return strong_correlations
def plot_strong_correlations(strong_correlations):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=strong_correlations.index, y=strong_correlations.values)
    plt.title('Strong Correlations with Target Variable (VULnerability_total)')
    plt.xlabel('Feature')
    plt.ylabel('Correlation')
    plt.xticks(rotation=45)
    st.pyplot() 
def neuropsychological_analysis(data):
    # ... your code for analysis using data ...
    st.write("Neuropsychological Analysis Results...")  # Placeholder for results

# Function for psychological analysis (replace with your actual implementation)
def psychological_analysis(data):
    # ... your code for analysis using data ...
    st.write("Psychological Analysis Results...")  # Placeholder for results

# File selection sidebar
def get_data_file_message(analysis_type):
    """
    Returns a message specific to the chosen analysis type,
    guiding the user on the expected data format.
    """
    if analysis_type == "Neuropsychological Data Analysis":
        return "Choose File (CSV)"
    elif analysis_type == "Psychological Data Analysis":
        return "Choose Psychological Data File (CSV)"
    elif analysis_type == "Full Wellness Program Data Analysis":
        return "Choose Full Wellness Program Data File (CSV)"
    else:
        return "Please select an analysis type."  # Default message

st.header("Data Selection and Analysis")

analysis_type = st.radio(
    "Select Analysis Type",
    ("Neuropsychological Data Analysis", "Psychological Data Analysis", "Full Wellness Program Data Analysis"),
    key="analysis_type"  # Unique key for radio button state persistence
)

if analysis_type:  # Check if a valid analysis type is chosen
    data_file_message = get_data_file_message(analysis_type)
    uploaded_file = st.file_uploader(data_file_message, type=['csv'])

    if uploaded_file is not None:
        st.write("**Analysis Type:**", analysis_type)
        # Replace this with your specific analysis logic based on data type
        # (e.g., using libraries like pandas for data manipulation and visualization)
        # ...
        #st.write("**Analysis Results:** (placeholder for insights and visualizations)")
else:
    st.write("Please select an analysis type to upload data.")




############
#######
####

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    dff=df.copy()
    if uploaded_file.name == "Neuropsychological_Data.csv":
        st.title("Neuropsychology station")
        last_column=df.columns[-1] 
        n_rows = st.slider('Choose number of rows to display', min_value=5, max_value=len(df), step=1)
        columns_toshow = st.multiselect('Select columns to display', df.columns.tolist(), default=df.columns.tolist())
        st.write(df[:n_rows][columns_toshow])

        categorical_variables = [col for col in df.columns if df[col].dtype == 'object' and col != 'Cognitive performances']
        categorical_variable = [col for col in df.columns if df[col].dtype == 'object']

        tab1, tab2,tab3,tab4, tab5 = st.tabs(["Numbers","Graphs","Percentages","Histogram", "General Statistics"] )

        with tab1:
            st.subheader("Cross-Tabulation Heatmap")
            variable_dropdown = st.selectbox('Select Variable:', df.columns.drop('Cognitive performances'), key="heatmap_variable")
            update_cross_tab(variable_dropdown)
        with tab2:
            st.subheader("Cross-Tabulation 2")
            variable_dropdown1 =st.selectbox('Select Variable:', categorical_variables, key="countplot_variable")
            update_cross_tab2(variable_dropdown1)
        with tab3:
            st.subheader("Categorical statistic")
            variable_dropdown2 = st.selectbox('Select Variable:', categorical_variable, key="categorical_variable")
            update_cross_tab3(variable_dropdown2)

        with tab4:
            st.subheader("Histogram")
            histogram_feature = st.selectbox('Select feature for histogram', df.select_dtypes(include=np.number).columns.tolist())
            fig_hist = px.histogram(df, x=histogram_feature)
            st.plotly_chart(fig_hist)
        with tab5:
            st.subheader("General Statistics")

            # Use a radio button to allow users to choose
            stat_choice = st.radio("Choose Statistics to Display:", 
                                ("Number of students", "Neuropsychological Variables", "Descriptive Statistics","Missing Values"))

            if stat_choice == "Number of students":
                st.write("Number of students:", "<h1>{}</h1>".format(len(df)), unsafe_allow_html=True)
            elif stat_choice == "Neuropsychological Variables":
                st.write("Neuropsychological Variables", "<h1>{}</h1>".format(len(df.columns)), unsafe_allow_html=True)
            elif stat_choice == "Descriptive Statistics":
                st.write("Descriptive Statistics:")
                st.write(df.describe())           
            elif stat_choice=="Missing Values":
                st.write("Missing Values:")
                st.write(df.isnull().sum())

        st.subheader("Neuropsychological Variables Analysis")

        Variable_neuro = ['Score Memory', 'Score Visual Attention', 'Score Visual Construction', 'Score Executive']  # Assuming these are in your dataframe

        # Check if there are neuropsychological variables in the data
        if any(var in df.columns for var in Variable_neuro):
            variables_neuro = [var for var in Variable_neuro if var in df.columns]  # Filter existing variables

            # Select plot type
            plot_type = st.radio("Choose Plot Type:", ("Histograms", "Boxplots", "Frequency Plots"))

            # Select numerical variables (assuming these are the ones you want for histograms)
            numerical_neuro = [var for var in variables_neuro if df[var].dtype != 'object']  # Filter numerical variables

            if plot_type == "Histograms":
                # Histograms
                fig = plt.figure(figsize=(12, 8))
                for i, var in enumerate(numerical_neuro, 1):
                    plt.subplot(2, 2, i)
                    sns.histplot(df[var], bins=20, kde=True)
                    plt.title(f'Distribution of {var}')
                    plt.xlabel(var)
                    plt.ylabel('Frequency')
                plt.tight_layout()
                st.pyplot(fig)
            elif plot_type == "Boxplots":
                # Boxplots
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
                axes = axes.flatten()

                for i, var in enumerate(variables_neuro):
                    ax = axes[i]
                    ax.boxplot(df[var])
                    ax.set_title(f'Box Plot of {var}')
                    ax.set_ylabel(var)

                plt.tight_layout()
                st.pyplot(fig)
            elif plot_type == "Frequency Plots":
                # Frequency plots for specific variables
                performance_variables = ['Performance Memory', 'Performance Visual Attention', 'Performance Visual Construction', 'Performance Executive']
                performance_variables = [var for var in performance_variables if var in df.columns]  # Filter existing performance variables

                # Set up subplots
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 20))
                axes = axes.flatten()

                # Visualize frequency of each category using bar plots
                for i, var in enumerate(performance_variables):
                    ax = axes[i]
                    sns.countplot(data=df, x=var, ax=ax)
                    ax.set_title(f'Frequency of {var}')
                    ax.set_xlabel(var)
                    ax.set_ylabel('Count')
                    ax.tick_params(axis='x', rotation=45)

                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.write("No neuropsychological variables found in the data.")

        st.subheader("Correlations")
        df=df.drop(["N°"], axis=1)
        encode_categorical = st.checkbox("Enable Label Encoding for Categorical Features")

        # ... rest of your code ...

        if encode_categorical:
            # Initialize LabelEncoder
            label_encoder = LabelEncoder()

            # Iterate over each column
            for col in df.select_dtypes(include='object'):  # Select categorical columns
                # Transform the column using LabelEncoder
                df[col] = label_encoder.fit_transform(df[col])

        plot_type = st.radio("Choose Correlation Plot:", ("Correlation Matrix", "Target Variable Correlations","Correlation with VULnerabilité"))
        corr_matrix = df.corr()
        if plot_type == "Correlation Matrix":
            plot_corr_matrix()
        elif plot_type == "Target Variable Correlations":
            plot_target_correlations()
        elif plot_type=="Correlation with VULnerabilité":
            VUL_correlations = corr_matrix['Cognitive performances'].sort_values(ascending=False)
            st.write("**Correlations with VULnerabilité (sorted by absolute value):**")
            for feature, correlation in VUL_correlations.items():
                st.write(f"- {feature}: {correlation:.2f}")
        else:
            st.write("Invalid plot type selected.")



                           
                                                            #######################################
                                                            #######################################
                                                            #######################################                                
                                                            #######################################
                                                            #######################################
                                                            #######################################


    elif uploaded_file.name == "Psychological_Data.csv":
        last_column=df.columns[-1]
        st.title("Psychology station")
        n_rows = st.slider('Choose number of rows to display', min_value=5, max_value=len(df), step=1)
        columns_toshow = st.multiselect('Select columns to display', df.columns.tolist(), default=df.columns.tolist())
        st.write(df[:n_rows][columns_toshow])

        categorical_variables = [col for col in df.columns if df[col].dtype == 'object' and col != 'Vulnerability Psychique']
        categorical_variable = [col for col in df.columns if df[col].dtype == 'object']
        tab1, tab2,tab3,tab4, tab5 = st.tabs(["Numbers","Graphs","Percentages","Histogram", "General Statistics"] )

        with tab1:
            st.subheader("Cross-Tabulation Heatmap")
            variable_dropdown = st.selectbox('Select Variable:', df.columns.drop('Vulnerability Psychique'), key="heatmap_variable")
            update_cross_tab(variable_dropdown)
        with tab2:
            st.subheader("Cross-Tabulation 2")
            variable_dropdown1 =st.selectbox('Select Variable:', categorical_variables, key="countplot_variable")
            update_cross_tab2(variable_dropdown1)
        with tab3:
            st.subheader("Categorical statistic")
            variable_dropdown2 = st.selectbox('Select Variable:', categorical_variable, key="categorical_variable")
            update_cross_tab3(variable_dropdown2)

        with tab4:
            st.subheader("Histogram")
            histogram_feature = st.selectbox('Select feature for histogram', df.select_dtypes(include=np.number).columns.tolist())
            fig_hist = px.histogram(df, x=histogram_feature)
            st.plotly_chart(fig_hist)
        with tab5:
            st.subheader("General Statistics")

            # Use a radio button to allow users to choose
            stat_choice = st.radio("Choose Statistics to Display:", 
                                ("Number of students", "Psychological Variables", "Descriptive Statistics","Missing Values"))

            if stat_choice == "Number of students":
                st.write("Number of students:", "<h1>{}</h1>".format(len(df)), unsafe_allow_html=True)
            elif stat_choice == "Psychological Variables":
                st.write("Psychological Variables:", "<h1>{}</h1>".format(len(df.columns)), unsafe_allow_html=True)
            elif stat_choice == "Descriptive Statistics":
                st.write("Descriptive Statistics:")
                st.write(df.describe())           
            elif stat_choice=="Missing Values":
                st.write("Missing Values:")
                st.write(df.isnull().sum())

        st.subheader("Psychological Variables Analysis")

        Variable_psy = ['Force Du Moi ','Relation De Lindividu Avec Le Monde Extérieu ','perception / monde exterieur', 'Connexion Sociale ','Structure Pensées Et Concept De Soi ']  # Assuming these are in your dataframe

        st.subheader("Correlations")
        encode_categorical = st.checkbox("Enable Label Encoding for Categorical Features")

        # ... rest of your code ...

        if encode_categorical:
            # Initialize LabelEncoder
            label_encoder = LabelEncoder()

            # Iterate over each column
            for col in df.select_dtypes(include='object'):  # Select categorical columns
                # Transform the column using LabelEncoder
                df[col] = label_encoder.fit_transform(df[col])

        plot_type = st.radio("Choose Correlation Plot:", ("Correlation Matrix", "Target Variable Correlations","Correlation with VULnerabilité"))
        corr_matrix = df.corr()
        if plot_type == "Correlation Matrix":
            plot_corr_matrix()
        elif plot_type == "Target Variable Correlations":
            plot_target_correlations()
        elif plot_type=="Correlation with VULnerabilité":
            VUL_correlations = corr_matrix['Vulnerability Psychique'].sort_values(ascending=False)
            st.write("**Correlations with VULnerabilité (sorted by absolute value):**")
            for feature, correlation in VUL_correlations.items():
                st.write(f"- {feature}: {correlation:.2f}")
        else:
            st.write("Invalid plot type selected.")







                                                            #######################################
                                                            #######################################
                                                            #######################################                                
                                                            #######################################
                                                            #######################################
                                                            #######################################








########STation psy#####

    elif uploaded_file.name == "Welnnes_program.csv":
        st.title("Wellness Programm full Data")
        last_column=df.columns[-1] 
        
        n_rows = st.slider('Choose number of rows to display', min_value=5, max_value=len(df), step=1)
        columns_toshow = st.multiselect('Select columns to display', df.columns.tolist(), default=df.columns.tolist())
        st.write(df[:n_rows][columns_toshow])
        st.subheader("General Statistics")

        stat_choice = st.radio("Choose Statistics to Display:", ("Number of Students", "Variables", "Descriptive Statistics", "Missing Values"))

        if stat_choice == "Number of Students":
            st.write("Number of Students:", "<h1>{}</h1>".format(len(df)), unsafe_allow_html=True)
        elif stat_choice == "Variables":
            st.write("Variables:", "<h1>{}</h1>".format(len(df.columns)), unsafe_allow_html=True)
        elif stat_choice == "Descriptive Statistics":
            st.write("Descriptive Statistics:")
            st.write(df.describe())           
        elif stat_choice == "Missing Values":
            st.write("Missing Values:")
            st.write(df.isnull().sum())
            
        df.dropna(subset=['Vulnerability Psychique'], inplace=True)
        categorical_variables = [col for col in df.columns if df[col].dtype == 'object' and col != 'vunlerability_total']
        categorical_variable = [col for col in df.columns if df[col].dtype == 'object']
        tab1, tab2,tab3,tab4 = st.tabs(["Numbers","Cross-Tabulation","Percentages","Histogram"] )
        tabs = ["Numbers", "Graphs", "Percentages", "Histogram"]
     

        with tab1:
            st.subheader("Cross-Tabulation Heatmap")
            variable_dropdown = st.selectbox('Select Variable:', df.columns.drop('vulnerability_total'), key="heatmap_variable")
            update_cross_tab(variable_dropdown)
        with tab2:
            st.subheader("Neuro vs Psy ")
            num_rows_different = df[df['Cognitive performances'] != df['Vulnerability Psychique']].shape[0]

            counts = {'Normal_VULnerable': 0, 'VULnerable_Normal': 0,
                    'A_risque_VULnerable': 0, 'VULnerable_A_risque': 0,
                    'A_risque_Normal': 0, 'Normal_A_risque': 0}
            for index, row in df.iterrows():
                cognitive_performance = row['Cognitive performances']
                VULnerability = row['Vulnerability Psychique']
                # Check conditions and update counts
                if cognitive_performance == 'Normal' and VULnerability == 'VULnérable':
                    counts['Normal_VULnerable'] += 1
                elif cognitive_performance == 'VULnérable' and VULnerability == 'Normal':
                    counts['VULnerable_Normal'] += 1
                elif cognitive_performance == 'A risque' and VULnerability == 'VULnérable':
                    counts['A_risque_VULnerable'] += 1
                elif cognitive_performance == 'VULnérable' and VULnerability == 'A risque':
                    counts['VULnerable_A_risque'] += 1
                elif cognitive_performance == 'A risque' and VULnerability == 'Normal':
                    counts['A_risque_Normal'] += 1
                elif cognitive_performance == 'Normal' and VULnerability == 'A risque':
                    counts['Normal_A_risque'] += 1


            # Display options and results
            selected_option = st.radio("Choose Statistics to Display:",
                                        ("Cross Table",
                                        "Number of students with different Cognitive Performances and VULnerability",
                                        "Normal_VULnerable",
                                        "VULnerable_Normal",
                                        "A_risque_VULnerable",
                                        "VULnerable_A_risque",
                                        "A_risque_Normal",
                                        "Normal_A_risque"))

            # Display results based on selected option
            if selected_option == "Cross Table":
                        st.subheader("Cross-tabulation and Heatmap")
                        # Assuming filtered_df is your filtered DataFrame
                        cross_table = pd.crosstab(df['Cognitive performances'], df['Vulnerability Psychique'])

                        # Plotting the heatmap for better visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.heatmap(cross_table, annot=True, fmt='d', cmap='viridis', cbar=True, linewidths=.5, ax=ax)
                        ax.set_title('Cross-Tabulation between Cognitive Performance and VUL')
                        ax.set_xlabel('Vulnerability Psychique')
                        ax.set_ylabel('Cognitive Performance')
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                        st.pyplot(fig)
            elif selected_option == "Number of students with different Cognitive Performances and VULnerability":
                st.write(f"Number of students with different Cognitive Performances and VULnerability: {num_rows_different}")
            else:
                # Display count based on the selected option (e.g., "Normal_VULnerable")
                st.write(f"Number of Students where {selected_option}: {counts[selected_option]}")
        with tab3:
            st.subheader("Categorical statistic")
            variable_dropdown2 = st.selectbox('Select Variable:', categorical_variable, key="categorical_variable")
            update_cross_tab3(variable_dropdown2)

        with tab4:
            st.subheader("Histogram")
            histogram_feature = st.selectbox('Select feature for histogram', df.select_dtypes(include=np.number).columns.tolist())
            fig_hist = px.histogram(df, x=histogram_feature)
            st.plotly_chart(fig_hist)
        st.subheader("Correlations")
        df=df.drop(["N°",'Percentile Memory','Percentile Visual Attention', 'Percentile Visual Construction','Percentile Executive'], axis=1)
        encode_categorical = st.checkbox("Enable Label Encoding for Categorical Features")

        # ... rest of your code ...

        if encode_categorical:
            # Initialize LabelEncoder
            label_encoder = LabelEncoder()

            # Iterate over each column
            for col in df.select_dtypes(include='object'):  # Select categorical columns
                # Transform the column using LabelEncoder
                df[col] = label_encoder.fit_transform(df[col])

        plot_type = st.radio("Choose Correlation Plot:", ("Correlation Matrix", "Target Variable Correlations","Correlation with VULnerabilité"))
        corr_matrix = df.corr()
        if plot_type == "Correlation Matrix":
            plot_corr_matrix()
        elif plot_type == "Target Variable Correlations":
            plot_target_correlations()
        elif plot_type == "Correlation with VULnerabilité":
            threshold = st.slider("Correlation Threshold (default: 0.2)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
            # Filter strong correlations
            correlation_matrix=df.corr()
            sorted_correlation_matrix = correlation_matrix['VULnerability_total'].abs().sort_values(ascending=False)
            strong_correlations = sorted_correlation_matrix[sorted_correlation_matrix > threshold ]

            fig, ax = plt.subplots(figsize=(18, 10))

            # Create the seaborn barplot using the figure's axis
            sns.barplot(x=strong_correlations.index, y=strong_correlations.values, ax=ax)

            # Set plot title and labels
            ax.set_title('Correlation with Target Variable (VULnerability_total)')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Correlation')

            # Rotate x-axis labels (optional)
            plt.xticks(rotation=45)  # Rotate x-axis labels outside Streamlit plot call

            # Display the plot in Streamlit
            st.pyplot(fig)
        else:
            st.write("Invalid plot type selected.")
            
      
       
