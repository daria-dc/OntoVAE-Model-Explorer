###========================================================================###
##                     RECOUNT3 GO-VAE MODEL DASH APP                       ##
###========================================================================###



### SETTING UP ENVIRONMENT ###

#-----------------------------------------------------------------------
# import libraries

import pandas as pd
import numpy as np
import json
import itertools
import umap

import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection

from igraph import Graph

pio.templates.default = "plotly_dark"

W3 = "https://www.w3schools.com/w3css/4/w3.css"
FA = "https://use.fontawesome.com/releases/v5.12.1/css/all.css"
#-----------------------------------------------------------------------



### IMPORTING NECESSARY DATA ###

#-----------------------------------------------------------------------
# load latent space
z = np.load('data/GTEx_samples_latent_space_embedding_RVAE-484.npy')

# load genes that were used for model training
genes = pd.read_csv('data/recount3_depth3_min_30_max_1000_trimmed_GO_genes.csv', header=None)
genes = genes.iloc[:,1].tolist()

# load GTEx annotation
annot = pd.read_csv('data/recount_brain_GTEx_TCGA_annot.csv')
annot = annot[annot.study == 'GTEx'].reset_index(drop=True)

# load GO term activations from the decoder
GO_act = np.load('data/GTEx_quant_norm_trimmed_ensemble100_avg_GO_activations.npy')

# load annotation of GO terms
GO_annot = pd.read_csv('data/GO_BP_at_least_depth3_min_30_max_1000_trimmed_annot.csv', sep=';')
GO_annot['GO'] = GO_annot[['GO_ID', 'GO_term']].agg(' | '.join, axis=1)
roots = GO_annot[GO_annot.depth == 3].GO_ID.tolist()


# load precomputed GO term Wang semantic similarities
wsem_sim = np.load('data/go_ids_depth3_trimmed_wang_sem_sim.npy')

# load data for default display
umap_default = pd.read_csv('data/RVAE-484_UMAP_results.csv', sep=';')
wilcox_default = pd.read_csv('data/trimmed_GTEx_ensemble100_brain_vs_rest_GO_node_activations.csv', sep=';')

# load GO graph
with open('data/GO_BP_recount3_genes_depth3_min_30_max_1000_trimmed_graph.json', 'r') as jfile:
    go_graph = json.load(jfile)
#-----------------------------------------------------------------------



### DEFINING HELPER FUNCTIONS

#-----------------------------------------------------------------------
# function to create scatter plot from UMAP

def create_scatter_plot(data, color):
    '''
    Input
    data: data from the precomputed UMAP
    color: the variable by which user wishes to color the plot ('study', 'tissue', 'disease')
    Output
    fig: the UMAP scatter plot colored by color
    '''
    fig = px.scatter(
        data, 
        x='UMAP 1', 
        y='UMAP 2',
        color=data[color],
        color_discrete_sequence=px.colors.qualitative.Dark24, 
        labels={'color': color}
    )   

    fig.update_traces(marker=dict(size=2.5))
    fig.update_layout({'plot_bgcolor': 'black'})

    return fig


# functions to retrieve common ancestors from list of GO IDs

def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            new_paths = find_all_paths(graph, node, end, path)
            for p in new_paths: 
                paths.append(p)
    return paths

def get_ancestors(node, roots, graph):
    paths = [find_all_paths(graph, node, root, []) for root in roots]
    paths_filt = [path for path in paths if len(path) > 0]
    paths_filt = list(itertools.chain.from_iterable(paths_filt))
    ancestors = {x for l in paths_filt for x in l}
    return ancestors if len(ancestors) > 0 else {node}

def get_comm_ancestors(leaves, roots, graph):
    ancestors = [get_ancestors(node, roots, graph) for node in leaves]
    common_ancestors = set.intersection(*ancestors)
    return common_ancestors



# function to compute cytoscape graph from Wilcoxon results

def get_cytoscape_components(group, wsem_sim):
    '''
    Input 
    group: the sorted results of the Wilcoxon test for one goup
    wsem_sim: the precomputed Wang semantic similarities
    Output
    nodes + edges: the elements for drawing the graph
    stylesheet: the stylesheet for the group
    '''

    # filter and sort the Wang sem sims to match the sorted GO terms of the group
    group_sims = wsem_sim[group.index.to_numpy(),:]
    group_sims = group_sims[:,group.index.to_numpy()]

    # apply a threshold and set similarity values below to 0
    group_sims[group_sims < 0.3] = 0

    # create the graph and retrieve coordinates
    graph = Graph.Weighted_Adjacency(group_sims, mode='undirected', loops=False)
    layout = graph.layout()
    group['x'], group['y'] = np.array(layout.coords).T

    # retrieve edge information
    edge_df = graph.get_edge_dataframe()
    es = [(group.GO_id.iloc[edge_df.source[i]], group.GO_id.iloc[edge_df.target[i]], edge_df.weight[i]) for i in range(edge_df.shape[0])]

    # perform community clustering and add community membership to the nodes
    community = graph.community_multilevel()
    group['community'] = community.membership

    # create color mapping for communities
    n_comm = len(np.unique(np.array(community.membership)))
    if n_comm <= 24:
        colors = px.colors.qualitative.Dark24[:n_comm] 
    else:
        colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Dark24[:n_comm-24]
    color_map = dict(zip(list(set(community.membership)), colors))
    group['color'] = [color_map[community.membership[i]] for i in range(group.shape[0])]

    # extract community members
    comm_members = {i: group[group.community == i].GO_id.tolist() for i in group.community.unique()}
    comm_members = {k:v for k,v in comm_members.items() if len(v) > 1}

    # get community representatives (members with most genes)
    representatives = []
    for vals in comm_members.values():
        representatives.append(group[group.GO_id.isin(vals)].sort_values('genes', ascending=False).iloc[0,:].GO_id)
    group['representative'] = np.where(group['GO_id'].isin(representatives), True, False)

    # get their common ancestors
    comm_ancestors = {k: get_comm_ancestors(leaves, roots, go_graph) for k,leaves in comm_members.items()}

    # get representative labels
    rep_labels = []
    for k, v in comm_ancestors.items():
        if len(v) == 1:
            rep_labels.append(GO_annot[GO_annot.GO_ID.isin(v)].GO_term.iloc[0])
        elif len(v) == 0:
            rep_labels.append(GO_annot[GO_annot.GO_ID.isin(comm_members[k])].sort_values('genes', ascending=False).iloc[0,:].GO_term)
        else:
            rep_labels.append(GO_annot[GO_annot.GO_ID.isin(v)].sort_values(['depth', 'genes'], ascending=[False, False]).iloc[0,:].GO_term)
    
    rep_dict = dict(zip(representatives, rep_labels))
    group['rep_label'] = group['GO_id']
    group['rep_label'] = group['rep_label'].map(rep_dict)

    # get rep labels for hover
    rep_dict2 = dict(zip(list(comm_ancestors.keys()), rep_labels))
    group['rep_label_hover'] = group['community']
    group['rep_label_hover'] = group['rep_label_hover'].map(rep_dict2).fillna('None')

    # create graph nodes and edges for cytoscape
    nodes = [{'data': 
                {'id': group.GO_id.iloc[i], 
                'label': group.GO_term.iloc[i],
                'stat': group.statistic.iloc[i],
                'pval': group.pval.iloc[i],
                'qval': group.qval.iloc[i],
                'depth': group.depth.iloc[i],
                'genes': np.log(group.genes.iloc[i] + 2)*10,
                'representative': group.representative.iloc[i],
                'rep_label': group.rep_label.iloc[i],
                'rep_label_hover': group.rep_label_hover.iloc[i]},
             'position': {'x': group.x.iloc[i]*50, 'y': group.y.iloc[i]*50},
             'classes': str(group.community.iloc[i]),
             'grabbable': True,
             'selectable': True} for i in range(group.shape[0])]

    edges = [{'data': {'source': e[0], 'target': e[1], 'weight': e[2]*5},
          'classes': str( group[group.GO_id == e[0]].community.iloc[0])} for e in es]

    # create stylehseet
    stylesheet = [
                    {
                        'selector': 'node',
                        'style': {
                            'width': 'data(genes)',
                            'height': 'data(genes)'
                        }
                    },

                    {
                        'selector': 'edge',
                        'style': {
                            'width': 'data(weight)'
                        }
                    },

                    {
                        'selector': '[representative == True]',
                        'style': {
                            'label': 'data(rep_label)',
                            'font-size': '20px'
                        }                    
                    }

    ] 

    stylesheet.extend(
        [
                    {
                        'selector': '.' + str(key),
                        'style': {
                            'background-color': color_map[key],
                            'line-color': color_map[key],
                            'color': color_map[key]
                        }
                    }
                    for key in color_map
        ]
    )

    return nodes + edges, stylesheet
#-----------------------------------------------------------------------



### INITIALIZATION OF THE APP

#-----------------------------------------------------------------------
# Initialize the app
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, W3, FA])
server = app.server
#app.config.suppress_callback_exceptions = True
#-----------------------------------------------------------------------



### DESIGNING THE APP LAYOUT

#-----------------------------------------------------------------------
# create app layout


app.layout = html.Div(
                [ 
                    dbc.NavbarSimple(
                        children=[
                            dbc.NavItem(
                                dbc.NavLink(
                                    children=[
                                        html.I(className='fab fa-github button-icon w3-large'),
                                        "Source Code"
                                    ],
                                    href="https://github.com/daria-dc/GO-VAE-Model-Explorer",
                                )
                            ),
                        ],
                        brand="GO-VAE Model Explorer",
                        brand_href="#",
                        color="dark",
                        dark=True,
                        sticky='top'
                    ),

    html.Br(),

    dbc.Container(
        [

            # Dropdown menus to select samples that UMAP will be performed on
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Markdown('Select available studies, tissues and diseases to inspect their clustering in the latent space. When nothing is selected, all samples are represented.'),
                            width={'size': 6},
                    ),
                    
                    dbc.Col(
                        children=[
                            dcc.Dropdown(id = 'study',
                                options = [{'label': i, 'value': i} for i in annot.study.unique()],
                                placeholder="Select one or more studies...",
                                multi = True,
                                value = []),
                        
                            dcc.Dropdown(id = 'tissue',
                                options = [{'label': i, 'value': i} for i in annot.tissue.unique()],
                                placeholder="Select one or more tissues...",
                                multi = True,
                                value = []),
    
                            dcc.Dropdown(id = 'disease',
                                options = [{'label': i, 'value': i} for i in annot.disease.unique()],
                                placeholder="Select one or more diseases...",
                                multi = True,
                                value =[])
                        ], width={'size': 6},
                    )
                ], #no_gutters = True
            ),

            html.Button(
                id = 'umap-button',
                children=[
                    html.I(className="fas fa-redo w3-large button-icon"), #, style={'padding-right': 10}),
                    "Re-run UMAP"
                ], className="w3-button w3-green w3-round"
            ),

            html.Br(),
            html.Br(),

            # Items that allow to pick by which variable UMAP should be colored
            dbc.Row(
                [   
                    dbc.Col(
                        html.P('Color by'),
                        width = {'size': 1, 'offset': 0}
                    ),

                    dbc.Col(
                        dcc.RadioItems(id = 'color_select_1',
                            options = [{'label': 'tissue', 'value': 'tissue'},
                                    {'label': 'study', 'value': 'study'},
                                    {'label': 'disease', 'value': 'disease'}],
                            value = 'tissue',
                            inputStyle={'margin-right': '3px', 'margin-left': '10px'}
                        ),
                        #width = {'size': 4, 'offset': 1}
                    ),

                    dbc.Col(
                        html.P('Color by'),
                        width = {'size': 1, 'offset': 0}
                    ),

                    dbc.Col(
                        dcc.RadioItems(id = 'color_select_2',
                            options = [{'label': 'tissue', 'value': 'tissue'},
                                    {'label': 'study', 'value': 'study'},
                                    {'label': 'disease', 'value': 'disease'}],
                            value = 'study',
                            inputStyle={'margin-right': '3px', 'margin-left': '10px'}
                        ),
                        #width = {'size': 4, 'offset': 7}
                    ),
                ]
            ),

            # UMAP graph of the latent space colored by two different variables
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id = 'UMAP_cluster_1', figure = {}),
                        width = {'size': 6},
                        #style={'backgroundColor': 'black'}
                        ),

                    dbc.Col(dcc.Graph(id = 'UMAP_cluster_2', figure = {}),
                        width = {'size': 6},
                        #style={'backgroundColor': 'black'}
                        ),
                ], 
            ),

            html.Br(),

            html.Hr(),

            dbc.Row(dbc.Col(dcc.Markdown('To compare GO term activations between groups you can select two groups to run a Wilcoxon test between them for each GO term. The graph below each group shows the top terms for this group, clustered by their Wang semantic similarities. If tissue or disease are only selected for one group, the comparison will be performed against all remaining tissues or diseases, respectively.'),
                            width={'size': 12},
                            ),
                    ),

            # Dropdown menus to select groups between which to perform Wilcoxon test for GO node activations
            dbc.Row(
                [
                    dbc.Col(dcc.Dropdown(id = 'study-group1',
                            options = [{'label': i, 'value': i} for i in annot.study.unique()],
                            placeholder="Select one or more studies...",
                            multi = True,
                            value = ['GTEx']),
                            width = {'size': 6}
                            ),

                    dbc.Col(dcc.Dropdown(id = 'study-group2',
                            options = [{'label': i, 'value': i} for i in annot.study.unique()],
                            placeholder="Select one or more studies...",
                            multi = True,
                            value = ['GTEx']),
                            width = {'size': 6}
                            ),
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(dcc.Dropdown(id = 'tissue-group1',
                            options = [{'label': i, 'value': i} for i in annot.tissue.unique()],
                            placeholder="Select one or more tissues...",
                            multi = True,
                            value = ['Brain']),
                            width = {'size': 6}
                            ),

                    dbc.Col(dcc.Dropdown(id = 'tissue-group2',
                            options = [{'label': i, 'value': i} for i in annot.tissue.unique()],
                            placeholder="Select one or more tissues...",
                            multi = True,
                            value = []),
                            width = {'size': 6}
                            ),
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(dcc.Dropdown(id = 'disease-group1',
                            options = [{'label': i, 'value': i} for i in annot.disease.unique()],
                            placeholder="Select one or more diseases...",
                            multi = True,
                            value =[]),
                            width = {'size': 6}
                            ),
                    
                    dbc.Col(dcc.Dropdown(id = 'disease-group2',
                            options = [{'label': i, 'value': i} for i in annot.disease.unique()],
                            placeholder="Select one or more diseases...",
                            multi = True,
                            value =[]),
                            width = {'size': 6}
                            ),

                ]
            ),

            html.Br(),

            html.Button(
                id = 'wilcox-button',
                children=[
                    html.I(className="fas fa-redo w3-large button-icon"), #, style={'padding-right': 10}),
                    "Perform Wilcoxon test"
                ], className="w3-button w3-green w3-round"
            ),

            #html.Button('Click here to perform Wilcoxon test', id='wilcox-button'),

            html.Br(),
            html.Br(),


            dbc.Row(
                [
                    dbc.Col(
                        html.P('Top GO terms'),
                        width = {'size': 1, 'offset': 0}
                    ),

                    dbc.Col(
                        dcc.RangeSlider(
                            id='cyto-slider1',
                            marks = {
                                0: '0',
                                100: '100',
                                200: '200',
                                300: '300',
                                400: '400',
                                500: '500'
                            },
                            min=0,
                            max=500,
                            step=50,
                            value=[0,100],
                        ),
                    ),

                    dbc.Col(
                        html.P('Top GO terms'),
                        width = {'size': 1, 'offset': 0}
                    ),
                    
                    dbc.Col(
                        dcc.RangeSlider(
                            id='cyto-slider2',
                            marks = {
                                0: '0',
                                100: '100',
                                200: '200',
                                300: '300',
                                400: '400',
                                500: '500'
                            },
                            min=0,
                            max=500,
                            step=50,
                            value=[0,100],
                        ),
                    ),
                ]
            ),

            # Cytoscape graph representations of GO terms
            dbc.Row(
                [
                    dbc.Col(
                        cyto.Cytoscape(
                            id='go-graph1',
                            minZoom=0.5,
                            maxZoom=2,
                            layout={'name': 'preset'},
                            style={'width': '100%', 'height': '500px', 'background-color': 'black'},
                            elements=[], 
                            stylesheet=[]         
                        ),
                    ),

                    dbc.Col(
                        cyto.Cytoscape(
                            id='go-graph2',
                            minZoom=0.5,
                            maxZoom=2,
                            layout={'name': 'preset'},
                            style={'width': '100%', 'height': '500px', 'background-color': 'black'},
                            elements=[], 
                            stylesheet=[]         
                        ),
                    ),
                ], #className='h-75'
            ),

            html.Br(),


  
            dbc.Alert(
                id='cyto-hover',
                children='Move over a node to display information!'
            )            ,

            # storage of intermediate computational results
            dcc.Store(id='umap_res'),
            dcc.Store(id='wilcox-res')

        ], #style={'height': '100vh'},
    )

 ])
#-----------------------------------------------------------------------



### DYNAMIC CALLBACKS FOR USER INTERACTION

#-----------------------------------------------------------------------
# Callbacks for dependence of dropdown menus for UMAP

@app.callback(
     Output('study', 'options'),
    [Input('tissue', 'value'),
     Input('disease', 'value')]
)
def set_study_options(tissue, disease):
    if len(tissue) == 0:
        tissue = annot.tissue.unique().tolist()
    if len(disease) == 0:
        disease = annot.disease.unique().tolist()
    annot_sub = annot[annot.tissue.isin(tissue) & annot.disease.isin(disease)]
    return [{'label': i, 'value': i} for i in annot_sub.study.unique()]

@app.callback(
     Output('tissue', 'options'),
    [Input('study', 'value'),
     Input('disease', 'value')]
)
def set_tissue_options(study, disease):
    if len(study) == 0:
        study = annot.study.unique().tolist()
    if len(disease) == 0:
        disease = annot.disease.unique().tolist()
    annot_sub = annot[annot.study.isin(study) & annot.disease.isin(disease)]
    return [{'label': i, 'value': i} for i in annot_sub.tissue.unique()]

@app.callback(
     Output('disease', 'options'),
    [Input('study', 'value'),
     Input('tissue', 'value')]
)
def set_disease_options(study, tissue):
    if len(study) == 0:
        study = annot.study.unique().tolist()
    if len(tissue) == 0:
        tissue = annot.tissue.unique().tolist()
    annot_sub = annot[annot.study.isin(study) & annot.tissue.isin(tissue)]
    return [{'label': i, 'value': i} for i in annot_sub.disease.unique()]


# Callback for UMAP computation

@app.callback(
     Output('umap_res', 'data'),
     Input('umap-button', 'n_clicks'),
    [State('study', 'value'),
     State('tissue', 'value'),
     State('disease', 'value')]
)
def compute_UMAP(n_clicks, study, tissue, disease):

    # check if the setting is the default
    default = False
    if len(study) == 0 and len(tissue) == 0 and len(disease) == 0:
        default=True

    if default == True:
        embedding = umap_default
    else:
        # adjust input for empty fields
        if len(study) == 0:
            study = annot.study.unique().tolist()
        if len(tissue) == 0:
            tissue = annot.tissue.unique().tolist()
        if len(disease) == 0:
            disease = annot.disease.unique().tolist()

        # subset data based on user input
        annot_sub = annot[annot.study.isin(study) & annot.tissue.isin(tissue) & annot.disease.isin(disease)]
        samples_GO_sub = annot_sub.index.to_numpy()
        z_sub = z[samples_GO_sub,:]
        top5_GO_sub = top5_GO.iloc[samples_GO_sub,:].reset_index(drop=True)

        # run UMAP
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(z_sub)
        embedding = pd.DataFrame(embedding)
        embedding.columns = ['UMAP 1', 'UMAP 2']

        embedding = pd.concat([embedding, top5_GO_sub, annot_sub.reset_index(drop=True)], axis=1)

    return embedding.to_json(date_format='iso', orient='split')


# Callback for UMAP coloring

@app.callback(
    [Output('UMAP_cluster_1', 'figure'),
     Output('UMAP_cluster_2', 'figure')],
    [Input('umap_res', 'data'),
     Input('color_select_1', 'value'),
     Input('color_select_2', 'value')]
)
def update_scatter_plot(umap_res, color1, color2):

    fig_data = pd.read_json(umap_res, orient='split')

    fig1 = create_scatter_plot(fig_data, color1)
    fig2 = create_scatter_plot(fig_data, color2)

    return fig1, fig2




# Callbacks for dependence of dropdown menus for GO terms

@app.callback(
     Output('study-group1', 'options'),
    [Input('tissue-group1', 'value'),
     Input('disease-group1', 'value')]
)
def set_study_group1_options(tissue, disease):
    if len(tissue) == 0:
        tissue = annot.tissue.unique().tolist()
    if len(disease) == 0:
        disease = annot.disease.unique().tolist()
    annot_sub = annot[annot.tissue.isin(tissue) & annot.disease.isin(disease)]
    return [{'label': i, 'value': i} for i in annot_sub.study.unique()]

@app.callback(
     Output('tissue-group1', 'options'),
    [Input('study-group1', 'value'),
     Input('disease-group1', 'value')]
)
def set_tissue_group1_options(study, disease):
    if len(study) == 0:
        study = annot.study.unique().tolist()
    if len(disease) == 0:
        disease = annot.disease.unique().tolist()
    annot_sub = annot[annot.study.isin(study) & annot.disease.isin(disease)]
    return [{'label': i, 'value': i} for i in annot_sub.tissue.unique()]

@app.callback(
     Output('disease-group1', 'options'),
    [Input('study-group1', 'value'),
     Input('tissue-group1', 'value')]
)
def set_disease_group1_options(study, tissue):
    if len(study) == 0:
        study = annot.study.unique().tolist()
    if len(tissue) == 0:
        tissue = annot.tissue.unique().tolist()
    annot_sub = annot[annot.study.isin(study) & annot.tissue.isin(tissue)]
    return [{'label': i, 'value': i} for i in annot_sub.disease.unique()]

@app.callback(
     Output('study-group2', 'options'),
    [Input('tissue-group2', 'value'),
     Input('disease-group2', 'value')]
)
def set_study_group2_options(tissue, disease):
    if len(tissue) == 0:
        tissue = annot.tissue.unique().tolist()
    if len(disease) == 0:
        disease = annot.disease.unique().tolist()
    annot_sub = annot[annot.tissue.isin(tissue) & annot.disease.isin(disease)]
    return [{'label': i, 'value': i} for i in annot_sub.study.unique()]

@app.callback(
     Output('tissue-group2', 'options'),
    [Input('study-group2', 'value'),
     Input('disease-group2', 'value')]
)
def set_tissue_group2_options(study, disease):
    if len(study) == 0:
        study = annot.study.unique().tolist()
    if len(disease) == 0:
        disease = annot.disease.unique().tolist()
    annot_sub = annot[annot.study.isin(study) & annot.disease.isin(disease)]
    return [{'label': i, 'value': i} for i in annot_sub.tissue.unique()]

@app.callback(
     Output('disease-group2', 'options'),
    [Input('study-group2', 'value'),
     Input('tissue-group2', 'value')]
)
def set_disease_group2_options(study, tissue):
    if len(study) == 0:
        study = annot.study.unique().tolist()
    if len(tissue) == 0:
        tissue = annot.tissue.unique().tolist()
    annot_sub = annot[annot.study.isin(study) & annot.tissue.isin(tissue)]
    return [{'label': i, 'value': i} for i in annot_sub.disease.unique()]




# Callback for performance of Wilcoxon test

@app.callback(
     Output('wilcox-res', 'data'),
     Input('wilcox-button', 'n_clicks'),
    [State('study-group1', 'value'),
     State('tissue-group1', 'value'),
     State('disease-group1', 'value'),
     State('study-group2', 'value'),
     State('tissue-group2', 'value'),
     State('disease-group2', 'value')]
)
def update_stat_table(n_clicks, st1, tis1, dis1, st2, tis2, dis2):

    # check if the setting is the default
    default=False

    if st1 == ['GTEx'] and st2 == ['GTEx'] and tis1 == ['Brain'] and len(tis2) == 0 and len(dis1) == 0 and len(dis2) == 0:
        default = True

    else:
        group1 = None
        group2 = None

        if not (len(st1) == 0 and len(tis1) == 0 and len(dis1) == 0):
            group1 = 'group1'

        if not (len(st2) == 0 and len(tis2) == 0 and len(dis2) == 0):
            group2 = 'group2'
        
    if default == True:
        results = wilcox_default
    else:
        if len(st1) == 0: st1 = annot.study.unique().tolist() 
        if len(st2) == 0: st2 = annot.study.unique().tolist() 
        if len(tis1) > 0:
            if len(tis2) == 0: 
                tis2 = [el for el in annot.tissue.unique().tolist() if el not in tis1]
        else:
            tis1 = annot.tissue.unique().tolist()
            if len(tis2) == 0:
                tis2 = annot.tissue.unique().tolist() 
        if len(dis1) > 0:
            if len(dis2) == 0: 
                dis2 = [el for el in annot.disease.unique().tolist() if el not in dis1]
        else:
            dis1 = annot.disease.unique().tolist()
            if len(dis2) == 0:
                dis2 = annot.disease.unique().tolist() 
            
        # subset two groups to compare based on user input
        annot_group1 = annot[annot.study.isin(st1) & annot.tissue.isin(tis1) & annot.disease.isin(dis1)]
        GO_act_group1 = GO_act[annot_group1.index.to_numpy(),:]

        annot_group2 = annot[annot.study.isin(st2) & annot.tissue.isin(tis2) & annot.disease.isin(dis2)]
        GO_act_group2 = GO_act[annot_group2.index.to_numpy(),:]

        # perform Wilcoxon test
        stats = [ranksums(GO_act_group1[:,i], GO_act_group2[:,i]) for i in range(GO_act.shape[1])]
        stat = [i[0] for i in stats]
        pval = [i[1] for i in stats]
        qvals = fdrcorrection(np.array(pval))

        # create results tables
        results = pd.DataFrame({'GO_id': GO_annot['GO_ID'],
                        'GO_term': GO_annot['GO_term'],
                        'depth': GO_annot['depth'],
                        'genes': GO_annot['genes'],
                        'statistic': np.around(stat, decimals=2),
                        'pval': np.around(pval, decimals=2),
                        'qval': np.around(qvals[1], decimals=2)})

    return results.to_json(date_format='iso', orient='split')


# Callbacks for cytoscape graph generation

@app.callback(
    [Output('go-graph1', 'elements'),
     Output('go-graph1', 'stylesheet')],
    [Input('wilcox-res', 'data'),
    Input('cyto-slider1', 'value')]
)
def draw_graph1(results, values):

    data = pd.read_json(results, orient='split')

    group1 = data[data.statistic > 0].sort_values(['qval', 'statistic'], ascending=[True, False]).iloc[values[0]:values[1],:]
    elements1, stylesheet1 = get_cytoscape_components(group1, wsem_sim)

    return elements1, stylesheet1

@app.callback(
    [Output('go-graph2', 'elements'),
     Output('go-graph2', 'stylesheet')],
    [Input('wilcox-res', 'data'),
     Input('cyto-slider2', 'value')]
)
def draw_graph2(results, values):

    data = pd.read_json(results, orient='split')

    group2 = data[data.statistic < 0].sort_values(['qval', 'statistic']).iloc[values[0]:values[1],:]
    elements2, stylesheet2 = get_cytoscape_components(group2, wsem_sim)

    return elements2, stylesheet2



# Callback for hovering over cytoscape graphs

@app.callback(
     Output('cyto-hover', 'children'),
    [Input('go-graph1', 'mouseoverNodeData'),
     Input('go-graph2', 'mouseoverNodeData')]
)
def display_info(data1, data2):

    data = None
    ctx = dash.callback_context
    
    if ctx.triggered[0]['prop_id'] == '.':
        contents = 'Move over a node to display its information!'

    if ctx.triggered[0]['prop_id'] == 'go-graph1.mouseoverNodeData':
        data = data1
    if ctx.triggered[0]['prop_id'] == 'go-graph2.mouseoverNodeData':
        data = data2

    if data is None:
        contents = 'Move over a node to display its information!'
    else:
        contents = []
        contents.extend([
            html.H5(data['id'] + ' | ' + data['label']),
            html.P('Cluster: ' + data['rep_label_hover'])
        ])
        
    return contents
#-----------------------------------------------------------------------      





### RUNNING THE APP

#-----------------------------------------------------------------------
# main body to run the app
if __name__ == '__main__':
    app.run_server(debug=True)
#-----------------------------------------------------------------------