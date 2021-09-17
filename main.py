# load libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import streamlit as st
#import SessionState
#import pickle
#import lightgbm
#import lightgbm as lgbm
# from Functions import*


# # picture and title
# col1, mid, col2 = st.columns([5, 1, 15])
# with col1:
#     path = "data/picture.PNG"
#     st.image(path, width=150)

# with col2:
#     st.title('Bank Loan Dashboard')
#     st.write('Credit Default Risk Prediction')

# # Load data
# def load_data():
#     application = pd.read_csv('application_test_sample.csv', index_col=0)
#     with open('data/train_data_final.pkl', 'rb') as f:
#         train_data_final = pickle.load(f)
#     application_list = train_data_final
#     bureau_sample_list = pd.read_csv('bureau_sample_list.csv')
#     bureau = pd.read_csv('bureau_test_fe.csv', index_col=0)
#     prev_application = pd.read_csv('prev_application_test_fe.csv', index_col=0)
#     prev_sample_list = pd.read_csv('prev_sample_list.csv')
#     data_sample = pd.read_csv('data_sample.csv')
#     data_compare_clients = pd.read_csv('data_compare_clients.csv')
#     data_pp = pd.read_csv('data_pp.csv')

#     return data_pp, data_sample, application, application_list, bureau, bureau_sample_list, prev_sample_list, prev_application, data_compare_clients

# data_pp, data_sample, application, application_list, bureau, bureau_sample_list, prev_sample_list, prev_application, data_compare_clients = load_data()

# # Select a sample of clients
# application_sample = application_list.head(50)
# client_list = application_sample['SK_ID_CURR'].tolist()


# # Prepare data for this set of clients
# #application_sample_fe = preprocess_application(application_sample)
# application_sample_fe = application.head(100)
# bureau_list = bureau_sample_list.head(100)
# bureau_sample = bureau_list[bureau_list['SK_ID_CURR'].isin(client_list)]

# #prev_application_sample = prev_application[prev_application['SK_ID_CURR'].isin(client_list)]
# prev_list = prev_sample_list.head(100)
# prev_application_sample = prev_list[prev_list['SK_ID_CURR'].isin(client_list)]

# #data = merge_data(application_sample_fe, bureau_sample, prev_application_sample)
# data = data_sample
# data = data.fillna(data.median())

# # Select one client ID
# col1, col2 = st.sidebar.columns([2,1])
# with col1:
#     st.title('Client Selection')


# col1, col2, col3 = st.sidebar.columns([2,1,1])
# with col1:
#     select_client = st.selectbox('Select a Client', client_list)

# with col3:
#     # Set session counter for reset option
#     session = SessionState.get(run_id=0)
#     if st.button("Reset"):
#         session.run_id += 1

# # Set labels for variables
# name_dict = {'NAME_INCOME_TYPE': 'Income Type',
#              'NAME_EDUCATION_TYPE': 'Education Type',
#              'NAME_FAMILY_STATUS': 'Family Status',
#              'CNT_CHILDREN': "Nb Children",
#              'AMT_INCOME_TOTAL': 'Total Income',
#              'AMT_CREDIT': 'Credit Amount',
#              'AMT_ANNUITY': 'Annuity',
#              'AMT_GOODS_PRICE': 'Goods Price',
#              'BUREAU_COUNT': 'Bureau Credit Nb Loans',
#              'BUREAU_ACTIVE_LOANS_PCT': 'Bureau Credit  % Active Loans',
#              'PREVAPP_SK_ID_PREV_COUNT': 'Nb Previous Application',
#              }


# # Store initial values
# init_dict = dict()
# for var in name_dict.keys():
#     init_dict[var] = data.loc[data['SK_ID_CURR'] == select_client, var].max()

# # Map education name
# education_dict = {0 : 'Lower secondary', 1:  'Secondary / secondary special',
#                   2: 'Incomplete higher', 3: 'Higher education', 4: 'Academic degree' }
# init_dict['NAME_EDUCATION_TYPE'] = education_dict[init_dict['NAME_EDUCATION_TYPE']]

# # display client parameters
# st.markdown('#')
# st.markdown('** Client Details **')

# # Categorical variables (static)
# col1, col2, col3, col4 = st.columns(4)
# cols = [col1, col2, col3, col4]
# for i, var in enumerate(['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN']):
#         with cols[i]:
#             st.markdown(name_dict[var])
#             st.info(init_dict[var])

# col1, col2, col3, col4 = st.columns(4)
# cols = [col1, col2, col3]
# for i, var in enumerate(['BUREAU_COUNT', 'BUREAU_ACTIVE_LOANS_PCT', 'PREVAPP_SK_ID_PREV_COUNT']):
#         with cols[i]:
#             st.markdown(name_dict[var])
#             st.info(int(init_dict[var]))


# # Numerical variables with sliders
# select_dict = dict()
# col5, col6, col7, col8 = st.columns(4)
# cols = [col5, col6, col7, col8]
# keys = [0, 10, 20, 30, 40]
# for i, var in enumerate(['AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE','AMT_CREDIT', 'AMT_ANNUITY']):
#     with cols[i]:
#         select_dict[var] = st.slider(name_dict[var],
#                                 int(0),
#                                 int(round(data[var].max(), -3)),
#                                 int(init_dict[var]),
#                                 key=str(keys[i] + session.run_id)
#                                 )

# # Display model result
# st.markdown('#')
# #st.markdown('** Scoring **')

# # load client's data
# #X = data
# X = data[data['SK_ID_CURR'] == select_client]

# # update slider values in client's data
# for var in select_dict.keys():
#     X.loc[X['SK_ID_CURR']==select_client, var] = select_dict[var]
# X['CREDIT_TERM'] = select_dict['AMT_CREDIT'] /(select_dict['AMT_ANNUITY']+ 0.00001)
# X['ANNUITY_INCOME_RATIO'] = select_dict['AMT_ANNUITY'] / (select_dict['AMT_INCOME_TOTAL']+ 0.00001)
# X['INCOME_ANNUITY_DIFF'] = select_dict['AMT_INCOME_TOTAL'] - select_dict['AMT_ANNUITY']
# X['CREDIT_GOODS_RATIO'] = select_dict['AMT_CREDIT'] / (select_dict['AMT_GOODS_PRICE'] + 0.00001)

# # Apply preprocessing
# #X_pp = preprocessing_data(X)
# X_pp = data_pp


# # final model tuned with Pycaret
# #model_lgbm = lgbm.LGBMClassifier(bagging_fraction=0.7, bagging_freq=6, boosting_type='gbdt', 
# #                                 class_weight='balanced', colsample_bytree=1.0,
# #                                 feature_fraction=0.5, importance_type='split', learning_rate=0.1,
# #                                 max_depth=-1, min_child_samples=66, min_child_weight=0.001,
# #                                 min_split_gain=0.4, n_estimators=90, n_jobs=-1, num_leaves=90,
# #                                 objective=None, random_state=123, reg_alpha=0.0005,
# #                                 reg_lambda=0.1, silent=True, subsample=1.0,
# #                                 subsample_for_bin=200000, subsample_freq=0)


# # load model
# model = pickle.load(open('lgbm_model.pkl', 'rb'))
# #model = model_lgbm
# #prevision = model.predict_proba(X_pp)[0,1]
# #prevision = model.predict_proba(X_pp, [0,1])
# #prevision = model.predict(X_pp)[0,1]
# #prevision = model_lgbm.fit(data_pp, data['TARGET'])


# # Set threshold
# default_threshold = 0.516
# col1, col2, col3 = st.columns((2,1,1))
# with col2:
#     threshold = st.slider('Threshold',0.0, 1.0, default_threshold, key=str(keys[4] + session.run_id))
#------------------------------------------------------------------------
# mon 2eme level
#------------------------------------------------------------------------------
# display score and risk proba
# with col1:
#     if prevision < threshold:
#         st.success('CREDIT GRANTED')
#     elif prevision >= threshold:
#         st.warning('CREDIT DENIED')

#     st.write("Default Risk : {} %".format(round(prevision * 100)))

# # display comparison graph
# st.sidebar.title('Statistics')

# # Select pop to compare to
# col1, col2 = st.sidebar.columns([1,1])
# with col1:
#     pop = st.selectbox('Compare Client to', ['All Clients', 'Non Defaulters', 'Defaulters'])


# Select a variable
# param_name_dict = {
#                       'External Source 2': 'EXT_SOURCE_2',
#                       'Credit Term': 'CREDIT_TERM',
#                       'External Source 1': 'EXT_SOURCE_1',
#                       'Credit Goods Ratio': 'CREDIT_GOODS_RATIO',
#                       'Bureau Active Loan %': 'BUREAU_ACTIVE_LOANS_PCT',
#                       'Days Employed Ratio': 'DAYS_EMPLOYED_RATIO',
#                       'Bureau Debt Credit Ratio': 'BUREAU_RATIO_DEBT_CREDIT',
#                       'Age': 'DAYS_BIRTH',
#                       'Goods Price': 'AMT_GOODS_PRICE',
#                       'Annuity Income Ratio':'ANNUITY_INCOME_RATIO',
# }

# param_list = list(param_name_dict.keys())

# # get selected variable
# param = st.sidebar.selectbox('Select Parameter', param_list, 7)

# # get client value for this variable
# x = data[data['SK_ID_CURR'] == select_client][param_name_dict[param]].item()

# if param == 'Age':
#     x = -x/365

# # Displot of selected variable
# df = data_compare_clients
# df = df.fillna(df.median())
# df['DAYS_BIRTH'] = -df['DAYS_BIRTH']/365

# pop_dict = {'Defaulters' : 1,
#             'Non Defaulters':0
#             }

# if pop != 'All Clients':
#     fig, ax = plt.subplots(figsize = (4,3))
#     sns.histplot(x = param_name_dict[param], data = df[df['TARGET']== pop_dict[pop]],
#              kde = True, linewidth=0,
#              color = 'darkgreen', bins = 30, ax = ax, alpha = 0.3)
#     mean = df[df['TARGET']== pop_dict[pop]][param_name_dict[param]].mean()
# else :
#     fig, ax = plt.subplots(figsize = (4,3))
#     sns.histplot(x = param_name_dict[param], data = df,
#              kde = True, linewidth=0,
#              color = 'darkgreen', bins = 30, ax = ax, alpha = 0.3)
#     mean = df[param_name_dict[param]].mean()
# plt.xlabel(param)

# indicate client data
# plt.axvline(x= x, c = '#F37768')
# fig.patch.set_alpha(0)
# st.sidebar.pyplot(fig)



# -*- coding: utf-8 -*-

#from functions import barcharts, gaugechart, load_data, comparisonchart

#--------------------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache
def load_data(filename):
    df = pd.read_csv(filename, index_col=0)
    return df

def gaugechart(key_data):
    val = key_data['proba']
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value= val,
        domain={'x': [0, 1], 'y': [0,1]},
        title={'text': "Default Probability"}))
    fig.update_traces(number_valueformat = ".1%",gauge_axis_tickmode='array',gauge_axis_range=[0,1])
    fig.update_layout(autosize=False,width=600,height=400)
    if val < 0.4:
        fig.update_traces(gauge_bar_color = 'green')
    else:
        if val > 0.7:
            fig.update_traces(gauge_bar_color = 'red')
        else: fig.update_traces(gauge_bar_color = 'orange')
    return fig

def barcharts(df, key_data):
    # MAKE SUBPLOTS
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[2,2],
        specs=[[{"type": "bar"},{"type": "bar"}]],
        subplot_titles=("Against all clients","Against similar clients"),
        vertical_spacing=0.1, horizontal_spacing=0.09)
     # STACKED BAR
    # Get probability distribution
    proba_groups = pd.DataFrame(df.proba_group.value_counts())
    proba_groups = proba_groups.sort_index()

    # Get probability distribution within the client's cluster
    proba_groups2 = pd.DataFrame(df[df['cluster'] == key_data['cluster']].proba_group.value_counts())
    proba_groups2 = proba_groups2.sort_index()

    #Create a list with only the client proba bins_values for charting
    client_proba = ['','','','','','','','','','']
    client_proba2 = client_proba.copy()
    position = proba_groups.index.tolist().index(key_data['proba_bin'])
    client_proba[position] = proba_groups.loc[key_data['proba_bin'],'proba_group']
    client_proba2[position] = proba_groups2.loc[key_data['proba_bin'], 'proba_group']

    proba = key_data['proba']

    # plot params
    labels = proba_groups.columns

    fig.update_layout(autosize=False,
                      width=1200,
                      height=400)
    for i, label_name in enumerate(labels):
        x = df.iloc[:, i].index
        # bar chart to represent clients distribution by probability bin
        fig.add_trace(go.Bar(x=proba_groups.index,
                             y=proba_groups.iloc[:, i],
                             name='Credit Risk Score Bin',
                             hovertemplate='<b>Proba: %{x}%</b><br>#Nb of clients: %{y:,.0f}',
                             legendgroup='grp2',
                             showlegend=True),
                             row=1, col=1)
        # bar chart to represent client position in the distribution
        fig.add_trace(go.Scatter(x=proba_groups.index,
                                 y=client_proba,
                                 mode='markers',
                                 marker_size = 20,
                                 marker_symbol='star-dot',
                                 name='Credit Risk Score',
                                 hovertemplate=f"<b>Client's risk probability: {proba*100:.2f}%</b>",
                                 showlegend=False),
                                 row=1, col=1)

        # bar chart to represent clients distribution by probability bin, against similar clients (in the same cluster)
        fig.add_trace(go.Bar(x=proba_groups2.index,
                             y=proba_groups2.iloc[:, i],
                             name='Credit Risk Score Bin',
                             hovertemplate='<b>Proba: %{x}%</b><br>#Nb of clients: %{y:,.0f}',
                             legendgroup='grp2',
                             showlegend=True),
                             row = 1, col = 2)

        # bar chart to represent client position in the cluster distribution
        fig.add_trace(go.Scatter(x=proba_groups.index,
                                 y=client_proba2,
                                 mode='markers',
                                 marker_size = 20,
                                 marker_symbol='star-dot',
                                 marker_color='yellow',
                                 name='Credit Risk Score',
                                 hovertemplate=f"<b>Client's risk probability: {proba*100:.2f}%</b>",
                                 showlegend=False),
                                 row=1, col=2)
    fig.update_yaxes(title_text='Nb of clients', linecolor='grey', mirror=True,
                     title_standoff=0, gridcolor='grey', gridwidth=0.1,
                     zeroline=False,
                     row=1, col=1)
    fig.update_xaxes(title_text='Credit Default Probability',linecolor='grey', mirror=True,
                     row=1, col=1)
    fig.update_xaxes(title_text='Credit Default Probability',linecolor='grey', mirror=True,
                     row=1, col=2)

    return fig

def comparisonchart(df, key_data):

    # Data prep
    ed_df = df[df['NAME_EDUCATION_TYPE'] == key_data['education_type']].copy()
    gender_df = df[df['CODE_GENDER'] == key_data['gender']].copy()
    age_df = df[df['age_group'] == key_data['age_group']].copy()
    debt_df = df[df['debt_group'] == key_data['debt_group']].copy()

    age_proba = age_df['proba'].mean()
    ed_proba = ed_df['proba'].mean()
    gender_proba = gender_df['proba'].mean()
    debt_proba = debt_df['proba'].mean()


    list_proba = [age_proba,ed_proba,gender_proba,debt_proba]
    list_color = []

    j=0
    for i in list_proba:
        if i < 0.45:
            list_color.append('green')
        elif i > 0.6:
            list_color.append('red')
        else:
            list_color.append('orange')
        j+=1

    age_color = list_color[0]
    ed_color = list_color[1]
    gender_color = list_color[2]
    debt_color = list_color[3]

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[2,2],
        subplot_titles=("Within the same Age Group","With the same Education Level", "With a similar Debt/Income ratio", "With the same Gender"),
        specs=[[{'type' : 'domain'}, {'type' : 'domain'}],
               [{'type' : 'domain'},{'type' : 'domain'}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    fig.add_trace(go.Indicator(
        value = age_proba,
        gauge={'bar': {'color': age_color}},
        delta = {'reference': 2*age_proba-key_data['proba'], 'valueformat': '.2%', 'increasing.color':'red', 'decreasing.color':'green'}),
        row = 1,
        col = 1
    )

    fig.update_traces(number_valueformat=".1%", gauge_axis_tickmode='array', gauge_axis_range=[0, 1])

    fig.add_trace(go.Indicator(
        value=ed_proba,
        gauge={'bar': {'color': ed_color}},
        delta={'reference': 2*ed_proba-key_data['proba'], 'valueformat':'.2%', 'increasing.color':'red', 'decreasing.color':'green'}),
        row=1,
        col=2
    )

    fig.update_traces(number_valueformat=".1%", gauge_axis_tickmode='array', gauge_axis_range=[0, 1])

    fig.add_trace(go.Indicator(
        value=debt_proba,
        gauge={'bar': {'color': debt_color}},
        delta={'reference': 2*debt_proba-key_data['proba'], 'valueformat':'.2%', 'increasing.color':'red', 'decreasing.color':'green'}),
        row=2,
        col=1
    )

    fig.update_traces(number_valueformat=".1%", gauge_axis_tickmode='array', gauge_axis_range=[0, 1])

    fig.add_trace(go.Indicator(
        value=gender_proba,
        gauge={'bar':{'color' : gender_color}},
        delta={'reference': 2*gender_proba-key_data['proba'], 'valueformat':'.2%', 'increasing.color':'red', 'decreasing.color':'green'}),
        row=2,
        col=2
    )

    fig.update_traces(number_valueformat=".1%", gauge_axis_tickmode='array', gauge_axis_range=[0, 1])

    fig.update_layout(
        template={'data': {'indicator': [{'mode': "number+delta+gauge"}]}},
        autosize=False,
        width=1000,
        height=800
    )

    return fig




#--------------------------
# ICON & TITLE
#--------------------------

#st.set_page_config(page_title="Home Credit - Client Scoring",
#                   page_icon=":money_with_wings:",
 #                  layout='wide')

#--------------------------
# VARIABLES & DATA PREP
#--------------------------

# Will only run once if already cached

df = load_data('customers_data.csv')

# Add a selectbox to the sidebar:

st.sidebar.header('Filter clients :')

age_category_list = ['All'] + np.unique(df['age_group']).tolist()
age_category = st.sidebar.selectbox("Age Group", age_category_list)

gender_list = ['All'] + np.unique(df['CODE_GENDER']).tolist()
gender = st.sidebar.selectbox("Gender", gender_list)

ed_level_list = ['All'] + np.unique(df['NAME_EDUCATION_TYPE']).tolist()
ed_level = st.sidebar.selectbox("Education Level", ed_level_list)

# Saving selections in a global filter

if (age_category != 'All'):
    age_filter = (df['age_group'] == age_category)
else: age_filter = (df['age_group'].isin(age_category_list))

if (ed_level != 'All'):
    ed_filter = (df['NAME_EDUCATION_TYPE'] == ed_level)
else: ed_filter = (df['NAME_EDUCATION_TYPE'].isin(ed_level_list))

if (gender != 'All'):
    gender_filter = (df['CODE_GENDER'] == gender)
else: gender_filter = (df['CODE_GENDER'].isin(gender_list))

client_list = df[age_filter & ed_filter & gender_filter]["SK_ID_CURR"].tolist()

st.sidebar.markdown("-------------------------------------------------")

st.sidebar.header('Select a client :')
select_client = st.sidebar.selectbox('Choose client ID',client_list)

client_index = df[df["SK_ID_CURR"] == select_client].index[0]

key_data = {
    'client_id': df.loc[client_index,'SK_ID_CURR'],
    'gender': df.loc[client_index,'CODE_GENDER'],
    'age': df.loc[client_index,'age'],
    'age_group': df.loc[client_index, 'age_group'],
    'debt_ratio': df.loc[client_index, 'debt_ratio'],
    'debt_group': df.loc[client_index, 'debt_group'],
    'nb_children': df.loc[client_index,'CNT_CHILDREN'],
    'family_status': df.loc[client_index,'NAME_FAMILY_STATUS'],
    'education_type': df.loc[client_index,'NAME_EDUCATION_TYPE'],
    'income_tot': df.loc[client_index,'AMT_INCOME_TOTAL'],
    'occupation_type': df.loc[client_index, 'OCCUPATION_TYPE'],
    'income_type': df.loc[client_index, 'NAME_INCOME_TYPE'],
    'owns_car': df.loc[client_index,'FLAG_OWN_CAR'],
    'owns_realty': df.loc[client_index,'FLAG_OWN_REALTY'],
    'credit_amount': df.loc[client_index,'AMT_CREDIT'],
    'goods_price': df.loc[client_index,'AMT_GOODS_PRICE'],
    'annuity': df.loc[client_index,'AMT_ANNUITY'],
    'target': df.loc[client_index,'target'],
    'cluster': df.loc[client_index,'cluster'],
    'proba': df.loc[client_index,'proba'],
    'proba_bin':df.loc[client_index,'proba_group']
}


filtered_df = df[df['SK_ID_CURR'] == select_client]

#--------------------------
# PAGE TITLE
#--------------------------
st.write('# **HOME CREDIT APPLICATION - RISK EVALUATION**')
st.write('')
st.write('')
st.write('')

st.markdown("-------------------------------------------------")

#--------------------------
# CLIENT DESCRIPTION
#--------------------------

#with st.beta_container():
with st.container():
    #col1, col2, col3 = st.beta_columns(3)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("## **Client Details**")
        st.markdown(f"**Gender:** {key_data['gender']}")
        st.markdown(f"**Age:** {key_data['age']:.0f} yo")
        st.markdown(f"**Nb children:   ** {key_data['nb_children']:.0f}")
        st.markdown(f"**Family Status:   ** {key_data['family_status']}")
        st.markdown(f"**Education Type:   ** {key_data['education_type']}")

    with col2:
        st.write("## **Occupation & Properties**")
        st.markdown(f"**Occupation Type:   ** {key_data['occupation_type']}")
        st.markdown(f"**Income Type:   ** {key_data['income_type']}")
        st.markdown(f"**Annual Income:   ** ${key_data['income_tot']:,.0f}")
        st.markdown(f"**Owns a car:   ** {key_data['owns_car']}")
        st.markdown(f"**Owns realty:   ** {key_data['owns_realty']}")

    with col3:
        st.markdown("## **Credit Application**")
        st.markdown(f"**Credit Amount:   ** ${key_data['credit_amount']:,.2f}")
        st.markdown(f"**Goods Price:   ** ${key_data['goods_price']:,.2f}")
        st.markdown(f"**Annuity:   ** ${key_data['annuity']:,.2f}")

st.markdown("-------------------------------------------------")

#--------------------------
# CREDIT SCORE RESULT
#--------------------------

st.write('')
st.write('')
st.write('')
st.write('')

#with st.beta_container():
with st.container():
    #cola, colb = st.beta_columns(2)
    cola, colb = st.columns(2)

    with cola:
        st.write('## **Credit Risk Value :**',unsafe_allow_html=True)
        #st.markdown(f"**Credit Score:** {key_data['proba']:,.2%}")
        fig2 = gaugechart(key_data)
        st.write(fig2)

    with colb:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        if key_data['target'] ==1:
            st.markdown("## Credit Application is **risky** & likely to **default**")
        else:
            st.markdown("## Credit Application is **safe** & likely to be **approved**")
            st.balloons()

#--------------------------
# CREDIT SCORE COMPARED TO OTHER CLIENTS
#--------------------------

st.markdown("-------------------------------------------------")


st.write("## **Where does the client stand?**")
st.write("*Comparison is made with clients filtered using the left panel*")
fig = barcharts(df[df["SK_ID_CURR"].isin(client_list)], key_data)
st.write(fig)

#--------------------------
# CREDIT SCORE COMPARED TO OTHER CLIENTS
#--------------------------

st.markdown("-------------------------------------------------")


st.write('')
st.write('## **Score comparison with similar clients :** ')
st.write("*Comparison is made with clients filtered using the left panel*")
st.write('')

fig3 = comparisonchart(df[df["SK_ID_CURR"].isin(client_list)],key_data)
st.write(fig3)