# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 02:41:19 2023

@author: chris
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import Dash, html, dash_table, dcc
from dash.dependencies import Input, Output
import base64
import io

# Incorporate data
diabetic = pd.read_csv('diabetic_data.csv')

""" Clean the dataset """

diabetic = diabetic.replace("?", np.nan)
# Drop the unknown races (3 lines)
diabetic = diabetic.drop(diabetic[diabetic['gender'] == 'Unknown/Invalid'].index)
# Drop the columns that are not useful due to many reasons we explained in our Jupiter Notebook and PowerPoint
diabetic.drop(['weight', 'payer_code', 'medical_specialty', 'examide', 'citoglipton','acetohexamide', 'troglitazone', 'glipizide-metformin', 'glimepiride-pioglitazone','metformin-rosiglitazone', 'metformin-pioglitazone', 'tolbutamide','repaglinide', 'nateglinide', 'chlorpropamide', 'acarbose', 'miglitol', 'tolazamide','glyburide-metformin'], axis=1, inplace=True)

# Replace NaN race by the most frequent
diabetic['race'].fillna(diabetic['race'].mode()[0], inplace=True)

# Drop the percentage of missing value that is over 95%
df_null = diabetic.isnull().sum(axis=1) * 100 / diabetic.shape[1]
diabetic = diabetic.drop(df_null[df_null > 95].index, axis=0)

#target value 
diabetic_target = diabetic.replace({"NO":0,
                    "<30":1,
                    ">30":0})

""" Visualisation  """

# Drop the columns named 'diag_1', 'diag_2', and 'diag_3'
diabetic_dropped = diabetic.drop(["diag_1", "diag_2", "diag_3"], axis=1)

# Create a list of columns with string type
string_columns = diabetic_dropped.select_dtypes(include='object').columns.tolist()

# Obtenez les colonnes qui ne sont pas de type chaîne de caractères
numeric_columns = diabetic.select_dtypes(include=['number']).columns

# Créez un subplot pour chaque colonne numérique
fig = make_subplots(rows=len(numeric_columns), cols=1, subplot_titles=numeric_columns)

# Ajoutez les traces de chaque colonne au subplot correspondant
for i, col in enumerate(numeric_columns, 1):
    fig.add_trace(go.Histogram(x=diabetic[col], nbinsx=50), row=i, col=1)

# Mettez à jour la mise en page du subplot
fig.update_layout(
    height=500 * len(numeric_columns),
    showlegend=False
)

# Mettez à jour les titres des axes
for i, col in enumerate(numeric_columns, 1):
    fig.update_xaxes(title_text=col, row=i, col=1)
    fig.update_yaxes(title_text='Frequency', row=i, col=1)
    
def generate_plot_overview():
    fig, ax = plt.subplots(figsize=(20, 15))
    diabetic.drop(["patient_nbr"], axis=1).hist(bins=50, figsize=(20, 15), ax=ax)
    plt.close(fig)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return html.Img(src='data:image/png;base64,{}'.format(encoded_image))

def generate_plots_readmitted():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1: Distribution
    sns.countplot(x="readmitted", data=diabetic_target, ax=axes[0])
    axes[0].set_title("Distribution of 'Readmitted' Values")

    # Plot 2: Proportion
    colors = plt.cm.Set2.colors
    diabetic_target["readmitted"].value_counts().plot.pie(autopct="%.1f%%", colors=colors, ax=axes[1])
    axes[1].set_title("Proportion of 'Readmitted' Values")

    # Save the figure to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Close the figure
    plt.close(fig)

    return html.Div(
        children=[
            html.H2("Distribution and Proportion of 'Readmitted' Values", style={'text-align': 'center', 'font-family': 'Arial', 'color': 'brown'}),
            html.Div(
                children=[
                    html.Img(src="data:image/png;base64,{}".format(encoded_image))
                ],
                style={"display": "flex", "justify-content": "center"}
            ),
        ],
        style={"display": "flex", "justify-content": "center", "align-items": "center", "flex-direction": "column", "height": "100vh"}
    )

def generate_correlation_heatmap():
    selected_features = ['time_in_hospital','num_lab_procedures','num_procedures','number_emergency', "number_inpatient", 'num_medications', 'number_diagnoses', 'readmitted']
    correlation_matrix = diabetic_target[selected_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title('Correlation Heatmap with Readmission Status')

    # Save the figure to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Close the figure
    plt.close()

    return html.Div(
        children=[
            html.H2("Correlation Heatmap with Readmission Status", style={'text-align': 'center', 'font-family': 'Arial', 'color': 'brown'}),
            html.Div(
                children=[
                    html.Img(src="data:image/png;base64,{}".format(encoded_image))
                ],
                style={"display": "flex", "justify-content": "center"}
            ),
        ],
        style={"display": "flex", "justify-content": "center", "align-items": "center", "flex-direction": "column", "height": "100vh"}
    )

# Graph options for the dropdown
graph_options = [
    {'label': 'Count of Readmitted Cases by Time in Hospital', 'value': 'count_by_time_in_hospital'},
    {'label': 'Box Plot of Number of Diagnoses by Readmission Status', 'value': 'boxplot_number_diagnoses'},
    {'label': 'Count of Readmitted Cases by Race', 'value': 'count_by_race'},
    {'label': 'Box Plot of Number of Lab Procedures by Readmission Status', 'value': 'boxplot_lab_procedures'},
    {'label': 'Histogram of Total Procedures/Medications by Readmission Status', 'value': 'histogram_total_procedures_medications'},
    {'label': 'Count of Readmitted Cases by Admission Type ID', 'value': 'count_by_admission_type_id'},
    {'label': 'Box Plot of Time in Hospital by Age and Readmission Status', 'value': 'boxplot_time_in_hospital_by_age'}
]

""" Application """

# Initialize the app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div(
    [
        html.H1(
            "Welcome to our Diabetes Dashboard",
            style={
                'text-align': 'center',
                'font-family': 'Arial',
                'color': 'white',
                'background-color': 'brown',
                'padding': '10px',
                'margin-bottom': '20px'
            }
        ),
        html.Div(
            [
                html.H2("Patient data table", style={'text-align': 'center', 'font-family': 'Arial', 'color': 'brown'}),
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": col, "id": col} for col in diabetic_target.columns],
                    data=diabetic_target.to_dict('records'),
                    page_size=10,
                    style_cell={'text-align': 'left'},
                    style_header={'background-color': 'brown', 'color': 'white', 'font-weight': 'bold'},
                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'background-color': 'whitesmoke'}], 
                    style_table={'overflowX': 'auto'}
                ),
                html.Div(
                    [
                        html.Button("Download CSV of our clean dataset", id="btn_csv",  style={'background-color': 'brown', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'font-size': '16px', 'cursor': 'pointer'}),
                        dcc.Download(id="download-dataframe-csv")
                    ],
                    style={'text-align': 'center', 'margin': '20px', 'border': '2px solid brown', 'padding': '10px'}
                ),
                
            ],
            style={'border': '2px solid brown', 'padding': '10px'}
        ),
        html.Div(
            [
                html.H2("Select Column for Visualization of string characteristics Proportion among Patient", style={'text-align': 'center', 'font-family': 'Arial', 'color': 'brown'}),
                dcc.Dropdown(
                    id='string-column-dropdown',
                    options=[{'label': col, 'value': col} for col in string_columns],
                    value=string_columns[0],
                    style={'width': '300px', 'margin': '0 auto'}
                ),
                html.H3(id='selected-column', style={'textAlign': 'center', 'marginTop': '20px'}),
                html.Div(id='graph-container', style={'textAlign': 'center'})
            ],
            # style={'border': '2px solid brown', 'padding': '10px'}
        ),
        html.Div(
            children=[
                html.H2("Numerical variables Overview", style={'text-align': 'center', 'font-family': 'Arial', 'color': 'brown'}),
                generate_plot_overview()
            ]
        ),
        html.Div(
            [
                html.H2("Select Column for Visualization of numeric characteristics Repartition among Patient",
                        style={'text-align': 'center', 'font-family': 'Arial', 'color': 'brown'}),
                dcc.Dropdown(
                    id='numeric-column-dropdown',
                    options=[{'label': col, 'value': col} for col in numeric_columns],
                    value=numeric_columns[0],
                    style={'width': '300px', 'margin': '0 auto'}
                ),
                html.Div(
                    dcc.Graph(id='numeric-graph')
                )
            ],
        ),
        html.Div(
            children=[
                generate_plots_readmitted()
            ],
            style={"display": "flex", "justify-content": "center", "align-items": "center", "flex-direction": "column", "height": "100vh", "overflow": "auto"}
        ),
        html.Div(
            children=[
                generate_correlation_heatmap()
            ],
            style={"display": "flex", "justify-content": "center", "align-items": "center", "flex-direction": "column", "height": "100vh", "overflow": "auto"}
        ),
        html.Div(children=[
            html.H2("Focus on our target value readmitted", style={'text-align': 'center', 'font-family': 'Arial', 'color': 'brown'}),
            dcc.Dropdown(
                id='graph-readmitted-dropdown',
                options=graph_options,
                value=graph_options[0]['value']
            ),
            html.Div(id='graph-readmitted-container')
        ]),        
    ],
    style={'border': '2px solid brown', 'padding': '10px'}
)

@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("btn_csv", "n_clicks")],
    prevent_initial_call=True
)
def download_dataframe(n_clicks):
    return dcc.send_data_frame(diabetic.to_csv, "mydfDiabetic.csv")

@app.callback(
    Output('graph-container', 'children'),
    [Input('string-column-dropdown', 'value')]
)
def update_graph(selected_column):
    graph_figure = px.pie(
        values=diabetic_dropped[selected_column].value_counts(),
        names=diabetic_dropped[selected_column].unique(),
        title=selected_column
    )
    graph = dcc.Graph(figure=graph_figure)
    return graph


@app.callback(
    Output('numeric-graph', 'figure'),
    [Input('numeric-column-dropdown', 'value')]
)
def update_numeric_graph(selected_column):
    fig = go.Figure(go.Histogram(x=diabetic[selected_column], nbinsx=50))
    fig.update_layout(
        title=f"Histogram of {selected_column}",
        xaxis_title=selected_column,
        yaxis_title='Frequency'
    )
    return fig

# Callback to display the graphs based on the dropdown selection
@app.callback(
    dash.dependencies.Output('graph-readmitted-container', 'children'),
    [dash.dependencies.Input('graph-readmitted-dropdown', 'value')]
)
def display_graph(selected_graph):
    
    if selected_graph == 'count_by_time_in_hospital':
        cross_tab = pd.crosstab(diabetic_target['time_in_hospital'], diabetic_target['readmitted'])
        bar_chart = go.Bar(x=cross_tab.index, y=cross_tab[1], name='Readmitted', marker_color='red')
        bar_chart_no = go.Bar(x=cross_tab.index, y=cross_tab[0], name='Not Readmitted', marker_color='blue')
        data = [bar_chart, bar_chart_no]
        layout = go.Layout(title='Count of Readmitted Cases by Time in Hospital',
                           xaxis=dict(title='Time in Hospital'),
                           yaxis=dict(title='Count'))
        fig = go.Figure(data=data, layout=layout)
        return dcc.Graph(figure=fig)
    
    elif selected_graph == 'boxplot_number_diagnoses':
        fig = go.Figure()
        fig.add_trace(go.Box(x=diabetic_target['readmitted'], y=diabetic_target['number_diagnoses'], name='Number of Diagnoses'))
        fig.update_layout(title='Box Plot of Number of Diagnoses by Readmission Status',
                          xaxis=dict(title='Readmission Status'),
                          yaxis=dict(title='Number of Diagnoses'))
        return dcc.Graph(figure=fig)
    
    elif selected_graph == 'count_by_race':
        cross_tab = pd.crosstab(diabetic_target['race'], diabetic_target['readmitted'])
        bar_chart = go.Bar(x=cross_tab.index, y=cross_tab[1], name='Readmitted', marker_color='red')
        bar_chart_no = go.Bar(x=cross_tab.index, y=cross_tab[0], name='Not Readmitted', marker_color='blue')
        data = [bar_chart, bar_chart_no]
        layout = go.Layout(title='Count of Readmitted Cases by Race',
                           xaxis=dict(title='Race'),
                           yaxis=dict(title='Count'))
        fig = go.Figure(data=data, layout=layout)
        return dcc.Graph(figure=fig)
    
    elif selected_graph == 'boxplot_lab_procedures':
        fig = go.Figure()
        fig.add_trace(go.Box(x=diabetic_target['readmitted'], y=diabetic_target['num_lab_procedures'], name='Number of Lab Procedures'))
        fig.update_layout(title='Box Plot of Number of Lab Procedures by Readmission Status',
                          xaxis=dict(title='Readmission Status'),
                          yaxis=dict(title='Number of Lab Procedures'))
        return dcc.Graph(figure=fig)
    
    elif selected_graph == 'histogram_total_procedures_medications':
        diabetic_target['total_procedures_medications'] = diabetic_target['num_procedures'] + diabetic_target['num_medications']
        fig = px.histogram(diabetic_target, x='total_procedures_medications', color='readmitted',
                   labels={'total_procedures_medications': 'Total Procedures/Medications'},
                   title='Distribution of Total Procedures/Medications by Readmission Status')
        return dcc.Graph(figure=fig)

    elif selected_graph == 'count_by_admission_type_id':
        cross_tab = pd.crosstab(diabetic_target['admission_type_id'], diabetic_target['readmitted'])
        bar_chart = go.Bar(x=cross_tab.index, y=cross_tab[1], name='Readmitted', marker_color='red')
        bar_chart_no = go.Bar(x=cross_tab.index, y=cross_tab[0], name='Not Readmitted', marker_color='blue')
        data = [bar_chart, bar_chart_no]
        layout = go.Layout(title='Count of Readmitted Cases by Admission Type ID',
                           xaxis=dict(title='Admission Type ID'),
                           yaxis=dict(title='Count'))
        fig = go.Figure(data=data, layout=layout)
        return dcc.Graph(figure=fig)

    elif selected_graph == 'boxplot_time_in_hospital_by_age':
        fig = px.box(diabetic_target, x='age', y='time_in_hospital', color='readmitted',
                     title='Boxplot of Time in Hospital by Age and Readmission Status',
                     labels={'age': 'Age', 'time_in_hospital': 'Time in Hospital', 'readmitted': 'Readmission Status'})
        return dcc.Graph(figure=fig)
    
    elif selected_graph == 'pairplot_diagnoses_lab_procedures':
        variables = ['number_diagnoses', 'num_lab_procedures']
        index_variable = 'number_diagnoses'
        target_variable = 'readmitted'
    
        group_labels = diabetic[target_variable].unique()
        scatter_matrix = ff.create_scatterplotmatrix(
            diabetic[variables], diag='histogram', index_vals=diabetic[index_variable],
            colormap='Viridis', colormap_type='cat', height=700, width=700
        )
    
        # Assign colors based on group labels
        colors = {group_labels[0]: 'blue', group_labels[1]: 'red'}
    
        for i in range(len(scatter_matrix['data'])):
            scatter_matrix['data'][i]['marker']['color'] = [colors[val] for val in diabetic[target_variable]]
    
        scatter_matrix.update_layout(title='Pair Plot of number of diagnoses and lab procedures by Readmission Status', title_x=0.5)
        
        fig = scatter_matrix
        return dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)