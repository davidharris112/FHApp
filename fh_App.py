# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# used chat gpt for general syntax help

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif')




with st.sidebar:
    st.subheader("Fetal Health Fatures Input")
    csv = st.file_uploader("Upload Your Data", type=["csv"])
    st.warning("Ensure your data follows the format outlined below")
    st.expander("Sample Data Format").write(pd.read_csv('fetal_health_user.csv'))


# Button to choose model type
input_type = st.sidebar.radio("Select Model", ("Decision Tree", "Random Forest", "AdaBoost", "Soft Voting Classifier"))

if input_type=="Decision Tree":
    # Load the pre-trained model from the pickle file
    dt_pickle = open('decision_tree_fh.pickle', 'rb') 
    clf = pickle.load(dt_pickle) 
    dt_pickle.close()
if input_type=="Random Forest":
    rf_pickle = open('random_forest_fh.pickle', 'rb') 
    clf = pickle.load(rf_pickle) 
    rf_pickle.close()
if input_type=="AdaBoost":
    Ada_pickle = open('AdaBoost_fh.pickle', 'rb')
    clf = pickle.load(Ada_pickle)
    Ada_pickle.close()
if input_type=="Soft Voting Classifier":
    SV_pickle = open('SoftVote_fh.pickle', 'rb')
    clf = pickle.load(SV_pickle)
    SV_pickle.close()
    #print("Soft Voting Classifier not implemented yet")

st.sidebar.markdown(f":blue-badge[:material/check: You selected {input_type}]")


st.write("Utilize our advanced Machine Learning application to predict fetal health classifications.")

if csv is None:
    st.info("Please upload data to proceed")
    st.stop()
if csv is not None:
    st.success("CSV Uploaded Successfully")
    fh_df = pd.read_csv(csv)

if input_type=="Decision Tree":
    st.subheader("Predicting Fetal Health Class Using Random Forest Model")
    # add Prediction Probability Column 
    for index, row in fh_df.iterrows():

        baseline_value = fh_df.loc[index, 'baseline value']
        accelerations = fh_df.loc[index, 'accelerations']
        fetal_movement = fh_df.loc[index, 'fetal_movement']
        uterine_contractions = fh_df.loc[index, 'uterine_contractions']
        light_decelerations = fh_df.loc[index, 'light_decelerations']
        severe_decelerations = fh_df.loc[index, 'severe_decelerations']
        prolongued_decelerations = fh_df.loc[index, 'prolongued_decelerations']
        abnormal_short_term_variability = fh_df.loc[index, 'abnormal_short_term_variability']
        mean_value_of_short_term_variability = fh_df.loc[index, 'mean_value_of_short_term_variability']
        percentage_of_time_with_abnormal_long_term_variability = fh_df.loc[index, 'percentage_of_time_with_abnormal_long_term_variability']
        mean_value_of_long_term_variability = fh_df.loc[index, 'mean_value_of_long_term_variability']
        histogram_width = fh_df.loc[index, 'histogram_width']
        histogram_min = fh_df.loc[index, 'histogram_min']
        histogram_max = fh_df.loc[index, 'histogram_max']
        histogram_number_of_peaks = fh_df.loc[index, 'histogram_number_of_peaks']
        histogram_number_of_zeroes = fh_df.loc[index, 'histogram_number_of_zeroes']
        histogram_mode = fh_df.loc[index, 'histogram_mode']
        histogram_mean = fh_df.loc[index, 'histogram_mean']
        histogram_median = fh_df.loc[index, 'histogram_median']
        histogram_variance = fh_df.loc[index, 'histogram_variance']
        histogram_tendency = fh_df.loc[index, 'histogram_tendency']
        
        features = [[baseline_value,accelerations,fetal_movement,uterine_contractions,light_decelerations,severe_decelerations,prolongued_decelerations,abnormal_short_term_variability,mean_value_of_short_term_variability,percentage_of_time_with_abnormal_long_term_variability,mean_value_of_long_term_variability,histogram_width,histogram_min,histogram_max,histogram_number_of_peaks,histogram_number_of_zeroes,histogram_mode,histogram_mean,histogram_median,histogram_variance,histogram_tendency]] 

        # Using predict() with new data provided by the csv
        new_prediction = clf.predict([[baseline_value,accelerations,fetal_movement,uterine_contractions,light_decelerations,severe_decelerations,prolongued_decelerations,abnormal_short_term_variability,mean_value_of_short_term_variability,percentage_of_time_with_abnormal_long_term_variability,mean_value_of_long_term_variability,histogram_width,histogram_min,histogram_max,histogram_number_of_peaks,histogram_number_of_zeroes,histogram_mode,histogram_mean,histogram_median,histogram_variance,histogram_tendency]]) 


        #  Predict class probabilities
        proba = clf.predict_proba(features)[0]
        # Get probability of the predicted class
        class_index = list(clf.classes_).index(new_prediction[0])
        predicted_class_proba = proba[class_index]


        # Store the predicted species & probability
        fh_df.loc[index, "Predicted Fetal Health"] = new_prediction[0]
        fh_df.loc[index, "Predicted Probability"] = predicted_class_proba


    # NOTE used chat gpt for these 
    fh_df["Predicted Fetal Health"] = fh_df["Predicted Fetal Health"].replace({
        1: "Normal",
        2: "Suspect",
        3: "Pathological"
    })
    color_map = {"Normal": "lime", "Suspect": "yellow", "Pathological": "orange"}
    # 
    display_df = fh_df[['baseline value', 'accelerations', 'fetal_movement',
                        'uterine_contractions', 'light_decelerations',
                        'severe_decelerations', 'prolongued_decelerations',
                        'abnormal_short_term_variability',
                        'mean_value_of_short_term_variability',
                        'percentage_of_time_with_abnormal_long_term_variability',
                        'mean_value_of_long_term_variability', 'histogram_width',
                        'histogram_min', 'histogram_max',
                        'histogram_number_of_peaks', 'histogram_number_of_zeroes',
                        'histogram_mode', 'histogram_mean', 'histogram_median',
                        'histogram_variance', 'histogram_tendency',
                        'Predicted Fetal Health', 'Predicted Probability']]

    # 
    styled_df = display_df.style.applymap(
        lambda val: f'background-color: {color_map.get(val, "white")}',
        subset=['Predicted Fetal Health']
    )

    with st.expander("Predicted Fetal Health"):
        st.dataframe(styled_df, use_container_width=True)

        # Showing additional items in tabs
    st.subheader("Model Performance")
    tab1, tab2, tab3, tab4 = st.tabs(["Decision Tree", "Feature Importance", "Confusion Matrix", "Classification Report"])

    # Tab 1: Visualizing Decision Tree
    with tab1:
        st.image('dt_visual.svg')
        st.caption("Visualization of the Decision Tree used in prediction.")

    # Tab 2: Feature Importance Visualization
    with tab2:
        st.write("### Feature Importance")
        st.image('feature_imp.svg')

    # Tab 3: Confusion Matrix
    with tab3:
        st.write("### Confusion Matrix")
        st.image('confusion_mat.svg')    

    # Tab 4: Classification Report
    with tab4:
        st.write("### Classification Report")
        report_df = pd.read_csv('class_report.csv', index_col = 0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

if input_type=="Random Forest":
    for index, row in fh_df.iterrows():

        baseline_value = fh_df.loc[index, 'baseline value']
        accelerations = fh_df.loc[index, 'accelerations']
        fetal_movement = fh_df.loc[index, 'fetal_movement']
        uterine_contractions = fh_df.loc[index, 'uterine_contractions']
        light_decelerations = fh_df.loc[index, 'light_decelerations']
        severe_decelerations = fh_df.loc[index, 'severe_decelerations']
        prolongued_decelerations = fh_df.loc[index, 'prolongued_decelerations']
        abnormal_short_term_variability = fh_df.loc[index, 'abnormal_short_term_variability']
        mean_value_of_short_term_variability = fh_df.loc[index, 'mean_value_of_short_term_variability']
        percentage_of_time_with_abnormal_long_term_variability = fh_df.loc[index, 'percentage_of_time_with_abnormal_long_term_variability']
        mean_value_of_long_term_variability = fh_df.loc[index, 'mean_value_of_long_term_variability']
        histogram_width = fh_df.loc[index, 'histogram_width']
        histogram_min = fh_df.loc[index, 'histogram_min']
        histogram_max = fh_df.loc[index, 'histogram_max']
        histogram_number_of_peaks = fh_df.loc[index, 'histogram_number_of_peaks']
        histogram_number_of_zeroes = fh_df.loc[index, 'histogram_number_of_zeroes']
        histogram_mode = fh_df.loc[index, 'histogram_mode']
        histogram_mean = fh_df.loc[index, 'histogram_mean']
        histogram_median = fh_df.loc[index, 'histogram_median']
        histogram_variance = fh_df.loc[index, 'histogram_variance']
        histogram_tendency = fh_df.loc[index, 'histogram_tendency']
        
        # Using predict() with new data provided by the csv
        new_prediction = clf.predict([[baseline_value,accelerations,fetal_movement,uterine_contractions,light_decelerations,severe_decelerations,prolongued_decelerations,abnormal_short_term_variability,mean_value_of_short_term_variability,percentage_of_time_with_abnormal_long_term_variability,mean_value_of_long_term_variability,histogram_width,histogram_min,histogram_max,histogram_number_of_peaks,histogram_number_of_zeroes,histogram_mode,histogram_mean,histogram_median,histogram_variance,histogram_tendency]]) 
        features = [[baseline_value,accelerations,fetal_movement,uterine_contractions,light_decelerations,severe_decelerations,prolongued_decelerations,abnormal_short_term_variability,mean_value_of_short_term_variability,percentage_of_time_with_abnormal_long_term_variability,mean_value_of_long_term_variability,histogram_width,histogram_min,histogram_max,histogram_number_of_peaks,histogram_number_of_zeroes,histogram_mode,histogram_mean,histogram_median,histogram_variance,histogram_tendency]] 

        #  Predict class probabilities
        proba = clf.predict_proba(features)[0]

        # Get probability of the predicted class
        class_index = list(clf.classes_).index(new_prediction[0])
        predicted_class_proba = proba[class_index]


        # Store the predicted species & the probability
        fh_df.loc[index, "Predicted Fetal Health"] = new_prediction[0]
        fh_df.loc[index, "Predicted Probability"] = predicted_class_proba

    
    # NOTE used chat gpt for these 
    fh_df["Predicted Fetal Health"] = fh_df["Predicted Fetal Health"].replace({
        1: "Normal",
        2: "Suspect",
        3: "Pathological"
    })
    color_map = {"Normal": "lime", "Suspect": "yellow", "Pathological": "orange"}
    # 
    display_df = fh_df[['baseline value', 'accelerations', 'fetal_movement',
                        'uterine_contractions', 'light_decelerations',
                        'severe_decelerations', 'prolongued_decelerations',
                        'abnormal_short_term_variability',
                        'mean_value_of_short_term_variability',
                        'percentage_of_time_with_abnormal_long_term_variability',
                        'mean_value_of_long_term_variability', 'histogram_width',
                        'histogram_min', 'histogram_max',
                        'histogram_number_of_peaks', 'histogram_number_of_zeroes',
                        'histogram_mode', 'histogram_mean', 'histogram_median',
                        'histogram_variance', 'histogram_tendency',
                        'Predicted Fetal Health', 'Predicted Probability']]

    # 
    styled_df = display_df.style.applymap(
        lambda val: f'background-color: {color_map.get(val, "white")}',
        subset=['Predicted Fetal Health']
    )

    with st.expander("Predicted Fetal Health"):
        st.dataframe(styled_df, use_container_width=True)

        # Showing additional items in tabs
    st.subheader("Model Performance")
# Showing additional items in tabs
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Residuals Histogram", "Predicited vs Actual"])


    # Tab 1: Feature Importance Visualization
    with tab1:
        st.write("### Feature Importance")
        st.image('feature_imp_rf.svg')

    # Tab 2: Confusion Matrix
    with tab2:
        st.write("### Confusion Matrix")
        st.image('confusion_mat_rf.svg')    

    # Tab 3: Classification Report
    with tab3:
        st.write("### Classification Report")
        report_df = pd.read_csv('class_report_rf.csv', index_col = 0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

if input_type=="AdaBoost":
    st.subheader("Predicting Fetal Health Class Using AdaBoost Model")
    # add Prediction Probability Column 
    for index, row in fh_df.iterrows():

        baseline_value = fh_df.loc[index, 'baseline value']
        accelerations = fh_df.loc[index, 'accelerations']
        fetal_movement = fh_df.loc[index, 'fetal_movement']
        uterine_contractions = fh_df.loc[index, 'uterine_contractions']
        light_decelerations = fh_df.loc[index, 'light_decelerations']
        severe_decelerations = fh_df.loc[index, 'severe_decelerations']
        prolongued_decelerations = fh_df.loc[index, 'prolongued_decelerations']
        abnormal_short_term_variability = fh_df.loc[index, 'abnormal_short_term_variability']
        mean_value_of_short_term_variability = fh_df.loc[index, 'mean_value_of_short_term_variability']
        percentage_of_time_with_abnormal_long_term_variability = fh_df.loc[index, 'percentage_of_time_with_abnormal_long_term_variability']
        mean_value_of_long_term_variability = fh_df.loc[index, 'mean_value_of_long_term_variability']
        histogram_width = fh_df.loc[index, 'histogram_width']
        histogram_min = fh_df.loc[index, 'histogram_min']
        histogram_max = fh_df.loc[index, 'histogram_max']
        histogram_number_of_peaks = fh_df.loc[index, 'histogram_number_of_peaks']
        histogram_number_of_zeroes = fh_df.loc[index, 'histogram_number_of_zeroes']
        histogram_mode = fh_df.loc[index, 'histogram_mode']
        histogram_mean = fh_df.loc[index, 'histogram_mean']
        histogram_median = fh_df.loc[index, 'histogram_median']
        histogram_variance = fh_df.loc[index, 'histogram_variance']
        histogram_tendency = fh_df.loc[index, 'histogram_tendency']
        
        features = [[baseline_value,accelerations,fetal_movement,uterine_contractions,light_decelerations,severe_decelerations,prolongued_decelerations,abnormal_short_term_variability,mean_value_of_short_term_variability,percentage_of_time_with_abnormal_long_term_variability,mean_value_of_long_term_variability,histogram_width,histogram_min,histogram_max,histogram_number_of_peaks,histogram_number_of_zeroes,histogram_mode,histogram_mean,histogram_median,histogram_variance,histogram_tendency]] 

        # Using predict() with new data provided by the csv
        new_prediction = clf.predict([[baseline_value,accelerations,fetal_movement,uterine_contractions,light_decelerations,severe_decelerations,prolongued_decelerations,abnormal_short_term_variability,mean_value_of_short_term_variability,percentage_of_time_with_abnormal_long_term_variability,mean_value_of_long_term_variability,histogram_width,histogram_min,histogram_max,histogram_number_of_peaks,histogram_number_of_zeroes,histogram_mode,histogram_mean,histogram_median,histogram_variance,histogram_tendency]]) 


        #  Predict class probabilities
        proba = clf.predict_proba(features)[0]
        # Get probability of the predicted class
        class_index = list(clf.classes_).index(new_prediction[0])
        predicted_class_proba = proba[class_index]


        # Store the predicted species & probability
        fh_df.loc[index, "Predicted Fetal Health"] = new_prediction[0]
        fh_df.loc[index, "Predicted Probability"] = predicted_class_proba


    # NOTE used chat gpt for these 
    fh_df["Predicted Fetal Health"] = fh_df["Predicted Fetal Health"].replace({
        1: "Normal",
        2: "Suspect",
        3: "Pathological"
    })
    color_map = {"Normal": "lime", "Suspect": "yellow", "Pathological": "orange"}
    # 
    display_df = fh_df[['baseline value', 'accelerations', 'fetal_movement',
                        'uterine_contractions', 'light_decelerations',
                        'severe_decelerations', 'prolongued_decelerations',
                        'abnormal_short_term_variability',
                        'mean_value_of_short_term_variability',
                        'percentage_of_time_with_abnormal_long_term_variability',
                        'mean_value_of_long_term_variability', 'histogram_width',
                        'histogram_min', 'histogram_max',
                        'histogram_number_of_peaks', 'histogram_number_of_zeroes',
                        'histogram_mode', 'histogram_mean', 'histogram_median',
                        'histogram_variance', 'histogram_tendency',
                        'Predicted Fetal Health', 'Predicted Probability']]

    # 
    styled_df = display_df.style.applymap(
        lambda val: f'background-color: {color_map.get(val, "white")}',
        subset=['Predicted Fetal Health']
    )

    with st.expander("Predicted Fetal Health"):
        st.dataframe(styled_df, use_container_width=True)

        # Showing additional items in tabs
    st.subheader("Model Performance")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

    # Tab 1: Feature Importance Visualization
    with tab1:
        st.write("### Feature Importance")
        st.image('feature_imp_Ada.svg')

    # Tab 2: Confusion Matrix
    with tab2:
        st.write("### Confusion Matrix")
        st.image('confusion_mat_Ada.svg')    

    # Tab 3: Classification Report
    with tab3:
        st.write("### Classification Report")
        report_df = pd.read_csv('class_report_Ada.csv', index_col = 0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")


if input_type=="Soft Voting Classifier":
    st.subheader("Predicting Fetal Health Class Using Soft Voting Classifier Model")
    # add Prediction Probability Column 
    for index, row in fh_df.iterrows():

        baseline_value = fh_df.loc[index, 'baseline value']
        accelerations = fh_df.loc[index, 'accelerations']
        fetal_movement = fh_df.loc[index, 'fetal_movement']
        uterine_contractions = fh_df.loc[index, 'uterine_contractions']
        light_decelerations = fh_df.loc[index, 'light_decelerations']
        severe_decelerations = fh_df.loc[index, 'severe_decelerations']
        prolongued_decelerations = fh_df.loc[index, 'prolongued_decelerations']
        abnormal_short_term_variability = fh_df.loc[index, 'abnormal_short_term_variability']
        mean_value_of_short_term_variability = fh_df.loc[index, 'mean_value_of_short_term_variability']
        percentage_of_time_with_abnormal_long_term_variability = fh_df.loc[index, 'percentage_of_time_with_abnormal_long_term_variability']
        mean_value_of_long_term_variability = fh_df.loc[index, 'mean_value_of_long_term_variability']
        histogram_width = fh_df.loc[index, 'histogram_width']
        histogram_min = fh_df.loc[index, 'histogram_min']
        histogram_max = fh_df.loc[index, 'histogram_max']
        histogram_number_of_peaks = fh_df.loc[index, 'histogram_number_of_peaks']
        histogram_number_of_zeroes = fh_df.loc[index, 'histogram_number_of_zeroes']
        histogram_mode = fh_df.loc[index, 'histogram_mode']
        histogram_mean = fh_df.loc[index, 'histogram_mean']
        histogram_median = fh_df.loc[index, 'histogram_median']
        histogram_variance = fh_df.loc[index, 'histogram_variance']
        histogram_tendency = fh_df.loc[index, 'histogram_tendency']
        
        features = [[baseline_value,accelerations,fetal_movement,uterine_contractions,light_decelerations,severe_decelerations,prolongued_decelerations,abnormal_short_term_variability,mean_value_of_short_term_variability,percentage_of_time_with_abnormal_long_term_variability,mean_value_of_long_term_variability,histogram_width,histogram_min,histogram_max,histogram_number_of_peaks,histogram_number_of_zeroes,histogram_mode,histogram_mean,histogram_median,histogram_variance,histogram_tendency]] 

        # Using predict() with new data provided by the csv
        new_prediction = clf.predict([[baseline_value,accelerations,fetal_movement,uterine_contractions,light_decelerations,severe_decelerations,prolongued_decelerations,abnormal_short_term_variability,mean_value_of_short_term_variability,percentage_of_time_with_abnormal_long_term_variability,mean_value_of_long_term_variability,histogram_width,histogram_min,histogram_max,histogram_number_of_peaks,histogram_number_of_zeroes,histogram_mode,histogram_mean,histogram_median,histogram_variance,histogram_tendency]]) 


        #  Predict class probabilities
        proba = clf.predict_proba(features)[0]
        # Get probability of the predicted class
        class_index = list(clf.classes_).index(new_prediction[0])
        predicted_class_proba = proba[class_index]


        # Store the predicted species & probability
        fh_df.loc[index, "Predicted Fetal Health"] = new_prediction[0]
        fh_df.loc[index, "Predicted Probability"] = predicted_class_proba


    # NOTE used chat gpt for these 
    fh_df["Predicted Fetal Health"] = fh_df["Predicted Fetal Health"].replace({
        1: "Normal",
        2: "Suspect",
        3: "Pathological"
    })
    color_map = {"Normal": "lime", "Suspect": "yellow", "Pathological": "orange"}
    # 
    display_df = fh_df[['baseline value', 'accelerations', 'fetal_movement',
                        'uterine_contractions', 'light_decelerations',
                        'severe_decelerations', 'prolongued_decelerations',
                        'abnormal_short_term_variability',
                        'mean_value_of_short_term_variability',
                        'percentage_of_time_with_abnormal_long_term_variability',
                        'mean_value_of_long_term_variability', 'histogram_width',
                        'histogram_min', 'histogram_max',
                        'histogram_number_of_peaks', 'histogram_number_of_zeroes',
                        'histogram_mode', 'histogram_mean', 'histogram_median',
                        'histogram_variance', 'histogram_tendency',
                        'Predicted Fetal Health', 'Predicted Probability']]

    # 
    styled_df = display_df.style.applymap(
        lambda val: f'background-color: {color_map.get(val, "white")}',
        subset=['Predicted Fetal Health']
    )

    with st.expander("Predicted Fetal Health"):
        st.dataframe(styled_df, use_container_width=True)

    #     # Showing additional items in tabs
    # st.subheader("Model Performance")
    # tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

    # # Tab 1: Feature Importance Visualization
    # with tab1:
    #     st.write("### Feature Importance")
    #     st.image('feature_imp_Ada.svg')

    # # Tab 2: Confusion Matrix
    # with tab2:
    #     st.write("### Confusion Matrix")
    #     st.image('confusion_mat_Ada.svg')    

    # # Tab 3: Classification Report
    # with tab3:
    #     st.write("### Classification Report")
    #     report_df = pd.read_csv('class_report_Ada.csv', index_col = 0).transpose()
    #     st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
    #     st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")