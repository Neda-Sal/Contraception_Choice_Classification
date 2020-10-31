import shap
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt

from pre_process import num_of_vars, change_vars, make_binary
from compare_models import train_val_test, logistic_report, knn_report, rfc_report, gb_report, compare

import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px


st.write('''# Women's Contracetption Choices

## The respondants in our dataframe are women who were married at least once. Here is just a snapshot of the dataframe.

''')

data = pd.read_csv('Final_model_df_renamed_cols.csv')
st.dataframe(data.head())

st.write(
'''
## The final model chosen was a GradientBoostingClassifier. 
### Final Model Confusion Matrix
'''
)

X_test3 = pd.read_pickle('X_test3.pkl')
y_test3 = pd.read_pickle('y_test3.pkl')


#load in model
with open("final_gcb.pkl", "rb") as f:
    final_GBC = pickle.load(f)


#print confusion matrix
fig, ax = plt.subplots()
plt.title('GradientBoostingClassifier Confusion Matrix')
plot_confusion_matrix(final_GBC, X_test3, y_test3, ax=ax)
st.pyplot(fig);


#Tableau stuff
# graph = open('figures/Plotly_all_logloss_2020-10-27.html', 'r', encoding='utf-8')
# source_code= graph.read()
# components.html(source_code, height = 600, width = 800)


st.write(
'''
### Feature Importance
'''
)

sorted_feats = pd.read_csv('sorted_impt_feats_final_model.csv')

fig2, ax = plt.subplots()
plt.title('Model Feature Importance')
plt.barh(sorted_feats['Features'], sorted_feats['Scores'])
plt.xlabel('Relative Impact on Model Output')
st.pyplot(fig2);




# import plotly.express as px
# fig3 = px.scatter_3d(data, x='Age', y='Num_living_children', z='desire_for_more_kids', color='Current_Method', opacity=0.7, labels={
#                      "Num_living_children": "Num of Living Children",
#                      "desire_for_more_kids": "Wants More Kids (Y/N)"}, 
#     title = 'Method choice by Age, Number of Living Children, Desire for More Children')
# st.plotly_chart(fig3);

# #data['Current_Method'].astype(int).astype(str)

# fig2 = px.scatter_3d(data, x='Age', y='Num_living_children', z='freq_of_intercourse',color='Current_Method', opacity=0.6,  labels={
#                      "Num_living_children": "Num of Living Children",
#                      "freq_of_intercourse": "Frequency of Intercourse"},title = 'Method choice by Age, Number of Living Children, Frequency of Intercourse')
# st.plotly_chart(fig2);




st.write(
'''
## Make a prediction!
'''
)

age = st.slider('Current Age:', 0, 100,25)
age_at_first_marriage = st.slider('Age at first marriage:', 0, age, value = 0)
media = st.slider('Media Exposure Level: (0 = No Exposure and 3 = Very Exposed)', 0, 3, 2)
num_kids = st.slider('Number of kids born:', 0, 20, value=0)
age_at_first_intercourse = st.number_input('Age at first intercourse:', value=20)
freq = st.number_input('Frequency of intercourse: (times per week)', value=4)


education = st.selectbox('Education Level:', ('0 No Education','1 Primary', '2 Secondary', '3 Higher'))
literacy = st.selectbox('Literacy Level:', ('0 Cannot Read','1 Reads with difficulty', '2 Reads Easily'))

hubbs_education = st.selectbox('Husband\'s Education Level:', ('0 No Education','1 Primary', '2 Secondary', '3 Higher'))
hubbs_literacy = st.selectbox('Husband\'s Literacy Level:', ('0 Cannot Read','1 Reads with difficulty', '2 Reads Easily'))

knows_methods = st.selectbox('Method Awareness:', ['0 Not aware of any methods','1 Aware of short term methods', '2 Aware of long term methods'])

st.write('''
Only select the boxes that are **True** for your demographic
''')
living_w_kids = st.checkbox('Children currently live in residence.', value=True)
knows_cylce = st.checkbox('Well-informed about ovulatory cylce.', value=True)
knows_source = st.checkbox('Well-informed on where to get contraception.', value=True)
wants_kids = st.checkbox('Wants kids within 2 years.', value=True)
worked_before_marriage = st.checkbox('Worked before marriage.', value=True)
working_now = st.checkbox('Currently working.', value=True)
heard_FP = st.checkbox('Heard a family practice message within the last month.', value=True)
transportation = st.checkbox('Has transportation.', value=True)


woman = np.array([age, num_kids, num_kids, age_at_first_marriage, age_at_first_intercourse, freq, 0, int(living_w_kids), int(knows_cylce), int(knows_source),  int(wants_kids), int(worked_before_marriage), int(working_now), int(heard_FP), int(transportation), media, int(education[0]), int(literacy[0]), int(hubbs_education[0]), int(hubbs_literacy[0]), int(knows_methods[0]) ]).reshape(1,-1)

#make the prediction with a few assumtions (i.e. all kids are still alive)
pred = final_GBC.predict(woman)


proba_0 = final_GBC.predict_proba(woman)[0,0]
proba_1 = final_GBC.predict_proba(woman)[0,1]

st.write(f'''
    **Associated Probability of No Use/Short-Term Use:** {proba_0:.2f}  
    **Associated Probability of Long-Term Use:** {proba_1:.2f}
''')

if int(pred) == 0:
    st.write('''
    ### Prediction:  
    **Not using contraception, or using short term contraception**  
    ''')

else:
    st.write(''' 
    ### **Prediction**:  
    **Using long term contraception**
    ''')
 #   st.write(f'Probability: ${pred:.2f}')

