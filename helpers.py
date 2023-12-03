import os
import json

def cache(func):
    def wrapper(session_id):
        cache_file = f"{func.__name__}_cache.json"
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as file:
                try:
                    result = json.load(file)
                    print(f"Cache hit for {func.__name__}")
                    return result
                except json.JSONDecodeError:
                    pass
        result = func(session_id)
        with open(cache_file, 'w') as file:
            json.dump(result, file, indent=4)
            print(f"Result cached for {func.__name__}")

        return result
    return wrapper

def need_data(func, call_list):
    def wrapper(session_id):
        for f in call_list:
            f(session_id)
        return func(session_id)
    return wrapper


@cache
def get_separated_layers(session_id):
    # Get Ian's CNN and apply to an image
    pass

@cache
def get_runnable_mask(session_id):
    # Apply U-Net to either layers or an image
    # based on u-net.ipynb
    pass

@cache
@need_data([get_separated_layers])
def get_triangle(session_id):
    # get triangle based on triangle.ipynb
    pass

@cache
@need_data([get_separated_layers])
def get_circles(session_id):
    # get_circles based on track_analysis.ipynb
    pass

@cache
@need_data([get_triangle, get_circles])
def get_configuration(session_id):
    # get_configuration based on track_analysis.ipynb
    pass

@cache
@need_data([get_runnable_mask])
def get_route(session_id):
    # get route based on find_path.ipynb
    pass



'''
needs to be done:

build_quadtree


transfer
'''

