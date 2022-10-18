###========================================================================###
##                        OntoVAE Model Explorer                            ##
###========================================================================###



### SETTING UP ENVIRONMENT ###

#-----------------------------------------------------------------------
# import libraries
import pandas as pd
import numpy as np
import json
import itertools
import requests
from io import BytesIO

import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
import dash_cytoscape as cyto
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.io as pio

from igraph import Graph

pio.templates.default = "plotly_dark"

W3 = "https://www.w3schools.com/w3css/4/w3.css"
FA = "https://use.fontawesome.com/releases/v5.12.1/css/all.css"
#-----------------------------------------------------------------------



### IMPORTING NECESSARY FILES ###

#-----------------------------------------------------------------------

# function to load pandas dataframe from url

def load_pandas(url, sep):
    response = requests.get(url)
    return pd.read_csv(BytesIO(response.content), sep=sep)

# function to load numpy arrays from url
def load_numpy(url):
    response = requests.get(url)
    return np.load(BytesIO(response.content), allow_pickle=True)

# function to load json from url
def load_json(url):
    response = requests.get(url)
    return json.load(BytesIO(response.content))

# load GTEx annotation
sample_annot = load_pandas('https://github.com/daria-dc/ovae-app_data/raw/main/recount3_GTEx_annot.csv', sep=",")

# load annotation of ontology terms
annot = load_pandas('https://github.com/daria-dc/ovae-app_data/raw/main/GO_ensembl_trimmed_annot.csv', sep=";")
annot['Term'] = annot[['ID', 'Name']].agg(' | '.join, axis=1)
roots = annot[annot.depth == 0].ID.tolist()

# load ontology graph
onto_graph = load_json('https://github.com/daria-dc/ovae-app_data/raw/main/GO_ensembl_trimmed_graph.json')

# load wang semantic similarities
wsem_sims = load_numpy('https://github.com/daria-dc/ovae-app_data/raw/main/GO_ensembl_trimmed_wsem_sim.npy')

# load UMAP results
umap_res = load_pandas('https://github.com/daria-dc/ovae-app_data/raw/main/recount3_GTEx_UMAP_results.csv', ";")

# load Wilcoxon res
wilcox_res = load_pandas('https://github.com/daria-dc/ovae-app_data/raw/main/recount3_GTEx_Wilcoxon_results.csv', ";")


#-----------------------------------------------------------------------



### DEFINING HELPER FUNCTIONS

#-----------------------------------------------------------------------
# function to create scatter plot from UMAP or pathway activities

def create_scatter_plot(data, color, x, y):
    '''
    Input
    data: data from the precomputed UMAP or from 2 pathway activities
    color: the variable by which user wishes to color the plot ('study', 'tissue')
    Output
    fig: the UMAP scatter plot colored by color or pathway activity scatter plot
    '''
    fig = px.scatter(
        data, 
        x=x, 
        y=y,
        color=data[color],
        color_discrete_sequence=px.colors.qualitative.Dark24, 
        labels={'color': color}
    )   

    fig.update_traces(marker=dict(size=2.5))
    fig.update_layout({'plot_bgcolor': 'black'})

    return fig


# functions to retrieve common ancestors from list of ontology IDs

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

def get_cytoscape_components(group):
    '''
    Input 
    group: the sorted results of the Wilcoxon test for one group
    Output
    nodes + edges: the elements for drawing the graph
    stylesheet: the stylesheet for the group
    '''
    # load onto graph
    # with open('data/GO_ensembl_trimmed_graph.json', 'r') as jfile:
    #     onto_graph = json.load(jfile)

    # # read in Wang semantic similarities
    # group_sims = np.load('data/GO_ensembl_trimmed_wsem_sim.npy', mmap_mode='r')

    # filter and sort the Wang sem sims to match the sorted terms of the group
    group_sims = wsem_sims[group.ind.to_numpy(),:]
    group_sims = group_sims[:,group.ind.to_numpy()]

    # apply a threshold and set similarity values below to 0
    group_sims[group_sims < 0.5] = 0

    # create the graph and retrieve coordinates
    graph = Graph.Weighted_Adjacency(group_sims, mode='undirected', loops=False)
    layout = graph.layout()
    group['x'], group['y'] = np.array(layout.coords).T

    # retrieve edge information
    edge_df = graph.get_edge_dataframe()
    es = [(group.id.iloc[edge_df.source[i]], group.id.iloc[edge_df.target[i]], edge_df.weight[i]) for i in range(edge_df.shape[0])]

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
    comm_members = {i: group[group.community == i].id.tolist() for i in group.community.unique()}
    comm_members = {k:v for k,v in comm_members.items() if len(v) > 1}

    # get community representatives (members with most genes)
    representatives = []
    for vals in comm_members.values():
        representatives.append(group[group.id.isin(vals)].sort_values('genes', ascending=False).iloc[0,:].id)
    group['representative'] = np.where(group['id'].isin(representatives), True, False)

    # get their common ancestors
    comm_ancestors = {k: get_comm_ancestors(leaves, roots, onto_graph) for k,leaves in comm_members.items()}

    # get representative labels
    rep_labels = []
    for k, v in comm_ancestors.items():
        if len(v) == 1:
            rep_labels.append(annot[annot.ID.isin(v)].Name.iloc[0])
        elif len(v) == 0:
            rep_labels.append(annot[annot.ID.isin(comm_members[k])].sort_values('genes', ascending=False).iloc[0,:].Name)
        else:
            rep_labels.append(annot[annot.ID.isin(v)].sort_values(['depth', 'genes'], ascending=[False, False]).iloc[0,:].Name)
    
    rep_dict = dict(zip(representatives, rep_labels))
    group['rep_label'] = group['id']
    group['rep_label'] = group['rep_label'].map(rep_dict)

    # get rep labels for hover
    rep_dict2 = dict(zip(list(comm_ancestors.keys()), rep_labels))
    group['rep_label_hover'] = group['community']
    group['rep_label_hover'] = group['rep_label_hover'].map(rep_dict2).fillna('None')

    # create graph nodes and edges for cytoscape
    nodes = [{'data': 
                {'id': group.id.iloc[i], 
                'label': group.term.iloc[i],
                'genes': np.log(group.genes.iloc[i] + 2)*10,
                'representative': group.representative.iloc[i],
                'rep_label': group.rep_label.iloc[i],
                'rep_label_hover': group.rep_label_hover.iloc[i]},
             'position': {'x': group.x.iloc[i]*50, 'y': group.y.iloc[i]*50},
             'classes': str(group.community.iloc[i]),
             'grabbable': True,
             'selectable': True} for i in range(group.shape[0])]

    edges = [{'data': {'source': e[0], 'target': e[1], 'weight': e[2]*5},
          'classes': str( group[group.id == e[0]].community.iloc[0])} for e in es]

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
                                    href="https://github.com/daria-dc/OntoVAE-Model-Explorer",
                                )
                            ),
                        ],
                        brand="OntoVAE Model Explorer",
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
                        dcc.Markdown('Inspect UMAP clustering of samples in the latent space of the model.'),
                            width={'size': 6},
                    ), # dbc.Col
                    
                ],
            ), # dbc.Row

            # Items that allow to pick by which variable UMAP should be colored
            dbc.Row(
                [   
                    dbc.Col(
                        html.P('Color by'),
                        width = {'size': 1, 'offset': 0}
                    ), # dbc.Col

                    dbc.Col(
                        dcc.RadioItems(id = 'color_select_1',
                            options = [{'label': 'tissue', 'value': 'tissue'},
                                    {'label': 'study', 'value': 'study'}],
                            value = 'tissue',
                            inputStyle={'margin-right': '3px', 'margin-left': '10px'}
                        ),
                        #width = {'size': 4, 'offset': 1}
                    ), #dbc.Col

                    dbc.Col(
                        html.P('Color by'),
                        width = {'size': 1, 'offset': 0}
                    ), # dbc.Col

                    dbc.Col(
                        dcc.RadioItems(id = 'color_select_2',
                            options = [{'label': 'tissue', 'value': 'tissue'},
                                    {'label': 'study', 'value': 'study'}],
                            value = 'study',
                            inputStyle={'margin-right': '3px', 'margin-left': '10px'}
                        ), # dcc.RadioItems
                        #width = {'size': 4, 'offset': 7}
                    ), # dbc.Col
                ]
            ), # dbc.Row

            # UMAP graph of the latent space colored by two different variables
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id = 'UMAP_cluster_1', figure = {}),
                        width = {'size': 6},
                        #style={'backgroundColor': 'black'}
                        ), # dbc.Col

                    dbc.Col(dcc.Graph(id = 'UMAP_cluster_2', figure = {}),
                        width = {'size': 6},
                        #style={'backgroundColor': 'black'}
                        ), # dbc.Col
                ], 
            ), # dbc.Row

            html.Br(),

            html.Hr(),

            dbc.Row(dbc.Col(dcc.Markdown('Select for which GO terms you want to display the pathway activities.'),
                            width={'size': 12},
                            ), # dbc.Col
                    ), # dbc.Row
            
            # Dropdown menus to select GO terms for which to display the activities
            dbc.Row(
                [
                    dbc.Col(dcc.Dropdown(id = 'goterm1',
                    options = [{'label': i, 'value': i} for i in annot.Name.tolist()],
                    placeholder="Select a GO term...",
                    multi=False,
                    value='digestive system process'
                    ),
                    width = {'size': 6}
                    ), # dbc.Col

                    dbc.Col(dcc.Dropdown(id = 'goterm3',
                    options = [{'label': i, 'value': i} for i in annot.Name.tolist()],
                    placeholder="Select a GO term...",
                    multi=False,
                    value='aortic valve morphogenesis'
                    ),
                    width = {'size': 6}
                    ), # dbc.Col
                ]
            ), # dbc.Row

            dbc.Row(
                [
                    dbc.Col(dcc.Dropdown(id = 'goterm2',
                    options = [{'label': i, 'value': i} for i in annot.Name.tolist()],
                    placeholder="Select a GO term...",
                    multi=False,
                    value='glutamate receptor signaling pathway'
                    ),
                    width = {'size': 6}
                    ), # dbc.Col

                    dbc.Col(dcc.Dropdown(id = 'goterm4',
                    options = [{'label': i, 'value': i} for i in annot.Name.tolist()],
                    placeholder="Select a GO term...",
                    multi=False,
                    value='axon ensheathment'
                    ),
                    width = {'size': 6}
                    ), # dbc.Col
                ]
            ), # dbc.Row

            html.Br(),

            # Scatter plots of pathway activities
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id = 'scatter1', figure = {}),
                        width = {'size': 6},
                        #style={'backgroundColor': 'black'}
                        ), # dbc.Col

                    dbc.Col(dcc.Graph(id = 'scatter2', figure = {}),
                        width = {'size': 6},
                        #style={'backgroundColor': 'black'}
                        ), # dbc.Col
                ], 
            ), # dbc.Row


            html.Br(),

            html.Hr(),

            dbc.Row(dbc.Col(dcc.Markdown('Select for which tissues you want to display the top terms.'),
                            width={'size': 12},
                            ), # dbc.Col
                    ), # dbc.Row

            # Dropdown menus to select groups for which to display ontology networks
            dbc.Row(
                [
                    dbc.Col(dcc.Dropdown(id = 'study-group1',
                            options = [{'label': i, 'value': i} for i in sample_annot.study.unique()],
                            placeholder="Select one or more studies...",
                            multi = True,
                            value = ['GTEx']),
                            width = {'size': 6}
                            ),

                    dbc.Col(dcc.Dropdown(id = 'study-group2',
                             options = [{'label': i, 'value': i} for i in sample_annot.study.unique()],
                             placeholder="Select one or more studies...",
                             multi = True,
                             value = ['GTEx']),
                             width = {'size': 6}
                             ),
                ]
            ), # dbc.Row

            dbc.Row(
                [
                    dbc.Col(dcc.Dropdown(id = 'tissue-group1',
                            options = [{'label': i, 'value': i} for i in sample_annot.tissue.unique()],
                            placeholder="Select a tissue...",
                            multi = False,
                            value = 'Blood'),
                            width = {'size': 6}
                            ),

                    dbc.Col(dcc.Dropdown(id = 'tissue-group2',
                             options = [{'label': i, 'value': i} for i in sample_annot.tissue.unique()],
                             placeholder="Select a tissue...",
                             multi = False,
                             value = 'Liver'),
                             width = {'size': 6}
                             ),
                ]
            ), # dbc.Row

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

            # download buttons for the cytopscape graphs


            html.Br(),


  
            dbc.Alert(
                id='cyto-hover',
                children='Move over a node to display information!'
            )            ,

            # storage of intermediate computational results
            dcc.Store(id='umap_res')

        ], #style={'height': '100vh'},
    )

 ])
#-----------------------------------------------------------------------



### DYNAMIC CALLBACKS FOR USER INTERACTION

#-----------------------------------------------------------------------

# Callback for UMAP coloring

@app.callback(
    [Output('UMAP_cluster_1', 'figure'),
     Output('UMAP_cluster_2', 'figure')],
    [Input('color_select_1', 'value'),
     Input('color_select_2', 'value')]
)
def update_umap_scatter_plot(color1, color2):

    fig1 = create_scatter_plot(umap_res, color1, 'UMAP 1', 'UMAP 2')
    fig2 = create_scatter_plot(umap_res, color2, 'UMAP 1', 'UMAP 2')

    return fig1, fig2


# Callback to make scatterplots for pathway activities

@app.callback(
    Output('scatter1', 'figure'),
    [Input('goterm1', 'value'),
     Input('goterm2', 'value')]
)
def update_scatter_plot1(term1, term2):

    act1 = load_numpy('https://github.com/daria-dc/ovae-app_data/raw/main/pathway_activities/recount3_GTEx_pathway_activities_' + str(annot.Name.tolist().index(term1)) + '.npy')
    act2 = load_numpy('https://github.com/daria-dc/ovae-app_data/raw/main/pathway_activities/recount3_GTEx_pathway_activities_' + str(annot.Name.tolist().index(term2)) + '.npy')
    act = pd.concat([pd.DataFrame(np.vstack((act1,act2)).T), sample_annot], axis=1)
    act.columns = [term1, term2] + sample_annot.columns.tolist()

    fig = create_scatter_plot(act, 'tissue', term1, term2)

    return fig

@app.callback(
     Output('scatter2', 'figure'),
    [Input('goterm3', 'value'),
     Input('goterm4', 'value')]
)
def update_scatter_plot2(term3, term4):

    act1 = load_numpy('https://github.com/daria-dc/ovae-app_data/raw/main/pathway_activities/recount3_GTEx_pathway_activities_' + str(annot.Name.tolist().index(term3)) + '.npy')
    act2 = load_numpy('https://github.com/daria-dc/ovae-app_data/raw/main/pathway_activities/recount3_GTEx_pathway_activities_' + str(annot.Name.tolist().index(term4)) + '.npy')
    act = pd.concat([pd.DataFrame(np.vstack((act1,act2)).T), sample_annot], axis=1)
    act.columns = [term3, term4] + sample_annot.columns.tolist()

    fig = create_scatter_plot(act, 'tissue', term3, term4)

    return fig

# Callbacks for cytoscape graph generation

@app.callback(
    [Output('go-graph1', 'elements'),
     Output('go-graph1', 'stylesheet')],
    [Input('tissue-group1', 'value'),
    Input('cyto-slider1', 'value')]
)
def draw_graph1(tissue, values):

    data = wilcox_res[wilcox_res.tissue == tissue]
    data = data.sort_values(['rank', 'med_stat'], ascending = (True, False)).iloc[values[0]:values[1],:]

    elements1, stylesheet1 = get_cytoscape_components(data)

    return elements1, stylesheet1

@app.callback(
    [Output('go-graph2', 'elements'),
     Output('go-graph2', 'stylesheet')],
    [Input('tissue-group2', 'value'),
     Input('cyto-slider2', 'value')]
)
def draw_graph2(tissue, values):

    data = wilcox_res[wilcox_res.tissue == tissue]
    data = data.sort_values(['rank', 'med_stat'], ascending = (True, False)).iloc[values[0]:values[1],:]

    elements2, stylesheet2 = get_cytoscape_components(data)

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
    app.run_server(debug=False, host='0.0.0.0', port=8080)
#-----------------------------------------------------------------------
