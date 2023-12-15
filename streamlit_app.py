import streamlit as st
import cv2
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
from helpers import *
import uuid
import os
import skimage.measure

st.set_page_config(layout="wide")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    session_id = st.session_state.get('session_id', str(uuid.uuid4()))
    session_id = st.text_area('session_id', value=session_id)
    st.session_state['session_id'] = session_id
    path = f'sessions/{session_id}'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)
    with open(f'sessions/{session_id}/image_orig.jpg', 'wb') as f:
        f.write(uploaded_file.getvalue())


    # Draw an image
    img = cv2.imread(f'sessions/{session_id}/image_orig.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    while img.shape[0] > 1500 or img.shape[1] > 1500:
        img = skimage.measure.block_reduce(img, (2,2,1), np.mean)
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = plt.figure()
        plt.title("Original image")
        img = np.uint8(img)
        plt.imshow(img)
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html,height=600)


    img_for_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"sessions/{session_id}/image.jpg", img_for_save)

    
    # get course layer
    get_separated_layers(session_id)
    course = np.load(f'sessions/{session_id}/course_layer.npy')
    with col2:
        fig = plt.figure()
        plt.title("Course layer")
        plt.imshow(course, cmap='gray')
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html,height=600)



    get_runnable_mask(session_id)
    runnable_mask = np.load(f'sessions/{session_id}/runnable_layer.npy')
    with col3:
        fig = plt.figure()
        plt.title("Runnable area")
        plt.imshow(runnable_mask * 255, cmap='gray')
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html,height=600)
    
    # detect triangle
    col1, col2 = st.columns([1, 3])
    with col1:
        triangle_erode = st.slider(f'{session_id}: triangle_erode', 0, 20, 3)
        triangle_dilate = st.slider(f'{session_id}: triangle_dilate', 0, 20, 5)
        triangles = get_triangle(session_id, triangle_erode, triangle_dilate)
        triangles = json.loads(st.text_area(f'{session_id}: triangle', json.dumps(triangles)))
    with col2:
        triangle_plot = np.zeros_like(course)
        for triangle in triangles:
            contours = np.array([[triangle['a'], triangle['b'], triangle['c']]], dtype=np.int32)
            triangle_plot = cv2.drawContours(triangle_plot, contours, -1, 255, 3)
        fig = plt.figure() 
        plt.title("Triangles detected")
        plt.imshow(triangle_plot, cmap='gray')
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600)

    col1, col2 = st.columns([1, 3])
    with col1:
        circle_radius = st.slider(f'{session_id}: circle radius', 1, 59, 9, step=2)
        circle_x = st.slider(f'{session_id}: circle X', circle_radius, img.shape[0]-circle_radius, circle_radius + (img.shape[0] - 2 * circle_radius) // 2)
        circle_y = st.slider(f'{session_id}: circle Y', circle_radius, img.shape[1]-circle_radius, circle_radius + (img.shape[1] - 2 * circle_radius) // 2)
    with col2:
        fig = plt.figure()
        plt.title("Original image")
        img = np.uint8(img)
        plt.imshow(img)
        circle = plt.Circle((circle_x, circle_y), circle_radius, color='r')
        plt.gca().add_patch(circle)
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600)
    

    # detect circles
    col1, col2 = st.columns([1, 3])
    generate_circles = None
    with col1:
        candidates_threshold = st.slider(f'{session_id}: candidates threshold', 0, 10 * circle_radius ** 2, int(0.8 * circle_radius ** 2))
        match_threshold = st.slider(f'{session_id}: match threshold', 0, 10 * circle_radius ** 2, int(0.3 * circle_radius ** 2))
        generate_circles = st.checkbox('generate circles (slow)')
        if generate_circles:
            circles = get_circles(session_id, circle_radius, candidates_threshold, match_threshold)    
            print(circles)
            if f'{session_id}: circles' in st.session_state:
                del st.session_state[f'{session_id}: circles']
        else:
            circles = {'circles': []}
        circles = json.loads(st.text_area(f'{session_id}: circles', json.dumps(circles)))
    with col2:
        circles = circles['circles']
        circles_plot = np.zeros_like(course)
        for circle in circles:
            circles_plot = cv2.circle(circles_plot, (circle['y'], circle['x']), circle['r'], 255, 3)
        fig = plt.figure() 
        plt.title("Circles detected")
        light_course = np.float32(course / 2)
        plt.imshow(light_course + circles_plot, cmap='gray')
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600)


    # find configuration
    points = []
    for triangle in triangles:
        points.append(np.int32((np.array(triangle['a']) + np.array(triangle['b']) + np.array(triangle['c'])) / 3)[0])
    for circle in circles:
        points.append(np.array([circle['x'], circle['y']]))
    build_course_quadtree_graph(session_id)
    build_course_distance_matrix(session_id, points)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        permutation = json.loads(st.text_area(f'{session_id}: permutation', json.dumps(get_circuit_permutation(session_id, points))))
    with col2:
        fig = plt.figure() 
        configuration = np.zeros((img.shape[0],img.shape[1],1), np.uint8)
        for circle in circles:
            cv2.circle(configuration, (circle['y'],circle['x']),12,255,10)
        for triangle in triangles:
            contours = np.array([[triangle['a'], triangle['b'], triangle['c']]], dtype=np.int32)
            triangle_plot = cv2.drawContours(configuration, contours, -1, 255, 3)
        for i in range(len(permutation) - 1):
            cv2.line(configuration, (points[permutation[i]][1], points[permutation[i]][0]), ((points[permutation[(i+1)%len(permutation)]][1], points[permutation[(i+1)%len(permutation)]][0])), 255, 10)

        plt.imshow(configuration + light_course.reshape((img.shape[0], img.shape[1], 1)), cmap='gray')
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600)
    
    build_runnable_quadtree_graph(session_id)
    legs = []
    for i in range(len(permutation)-1):
        legs.append((points[permutation[i]], points[permutation[(i+1)]]))
    print(legs)
    get_routes(session_id, legs)
    routes = np.load(f'sessions/{session_id}/routes_map.npy')
    map_with_routes = (img / 255 * 0.8 + routes * 0.2)
    print(img.max())
    print(routes.max())
    print(img.shape)
    print(routes.shape)
    fig = plt.figure() 
    plt.imshow(map_with_routes)
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)
    
