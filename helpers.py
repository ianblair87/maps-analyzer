import os
import json
import numpy as np
from triangle import get_triangle_impl
from course_layer import detect_course
from runnable_layer import get_runnable_layer
from circles import get_circles_impl
from quadtree import build_quadtree, build_graph_from_quadtree, quadtree_dist_matrix, get_best_permutation
from route_solver import get_best_routes
import pickle
import streamlit as st

def cache(func):
    def wrapper(*args):
        session_id = args[0]
        cache_file = f"sessions/{session_id}/{func.__name__}_cache.json"
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as file:
                try:
                    result = json.load(file)
                    print(f"Cache hit for {func.__name__}")
                    return result
                except json.JSONDecodeError:
                    pass
        result = func(*args)
        with open(cache_file, 'w') as file:
            json.dump(result, file, indent=4)
            print(f"Result cached for {func.__name__}")

        return result
    return wrapper

def need_data(call_list):
    def wrapper(func):
        def g(*args):
            for f in call_list:
                f(*args)
            return func(*args)
        g.__name__ = func.__name__ + '_wrapper'
        return g
    return wrapper


@st.cache_data
def get_separated_layers(session_id):
    # Get Ian's CNN and apply to an image
    course = detect_course(session_id)
    np.save(open(f'sessions/{session_id}/course_layer.npy', 'wb'), course)
    return {
        'course_layer_picture': f'{session_id}/course_layer.jpg',
        'course_layer_data': f'{session_id}/course_layer.npy'
    }
    

@st.cache_data
def get_runnable_mask(session_id):
    # Apply U-Net to either layers or an image
    # based on u-net.ipynb
    runnable_layer = get_runnable_layer(session_id)
    np.save(f'sessions/{session_id}/runnable_layer.npy', runnable_layer)
    return  {
        'runnable_layer': f'sessions/{session_id}/runnable_layer.npy'
    }

@st.cache_data
# @need_data([get_separated_layers])
def get_triangle(session_id, erode, dilate):
    return get_triangle_impl(session_id, erode, dilate)

@st.cache_data
# @need_data([get_separated_layers])
def get_circles(session_id, r):
    # get_circles based on track_analysis.ipynb
    return get_circles_impl(session_id, r)

@st.cache_data
# @need_data([get_separated_layers])
def build_course_quadtree_graph(session_id):
    course = np.load(f'sessions/{session_id}/course_layer.npy')
    graph = {}
    tree = build_quadtree(course)
    pickle.dump(tree, open(f'sessions/{session_id}/quadtree.pkl', 'wb'))
    build_graph_from_quadtree(tree.root, graph)
    return graph


@st.cache_data
# @need_data([build_course_quadtree_graph])
def build_course_distance_matrix(session_id, points):
    course = np.load(f'sessions/{session_id}/course_layer.npy')
    graph = build_course_quadtree_graph(session_id)
    tree = pickle.load(open(f'sessions/{session_id}/quadtree.pkl', 'rb'))
    return quadtree_dist_matrix(course, points, tree, graph)

@st.cache_data
# @need_data([build_course_distance_matrix])
def get_circuit_threshold(session_id, threshold):
    # get dist matrix
    # get all edges < threshold
    pass

@st.cache_data
# @need_data([build_course_distance_matrix])
def get_circuit_permutation(session_id, points):
    course = np.load(f'sessions/{session_id}/course_layer.npy')
    dist = build_course_distance_matrix(session_id, points)
    return get_best_permutation(points, dist, course)

@st.cache_data
# @need_data([get_runnable_mask])
def build_runnable_quadtree_graph(session_id):
    runnable = np.load(f'sessions/{session_id}/runnable_layer.npy')
    graph = {}
    tree = build_quadtree(runnable)
    pickle.dump(tree, open(f'sessions/{session_id}/runnable_quadtree.pkl', 'wb'))
    build_graph_from_quadtree(tree.root, graph)
    return graph

@st.cache_data
# @need_data([build_runnable_quadtree_graph])
def get_routes(session_id, legs):
    graph = build_runnable_quadtree_graph(session_id)
    tree = pickle.load(open(f'sessions/{session_id}/runnable_quadtree.pkl', 'rb'))
    runnable = np.load(f'sessions/{session_id}/runnable_layer.npy')
    res = get_best_routes(runnable, tree, graph, legs)
    np.save(f'sessions/{session_id}/routes_map.npy', res)
    return {
        'routes_map': f'sessions/{session_id}/routes_map.npy'
    }
