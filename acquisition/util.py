import pandas as pd
import numpy as np
import itertools
from pyproj import Transformer
import folium

def save_df(df:pd.DataFrame, dst, **options):
    df.to_csv(dst,**options)

def change_coordinate_system(coordinate, reverse=True, crs_from="epsg:4326", crs_to="epsg:5179"):
    transformer = Transformer.from_crs(crs_from, crs_to)
    out = list(transformer.transform(*coordinate))
    if reverse:
        out.reverse()
    return out

def cartesian_product(*args):
    out = itertools.product(*args)
    return out

def specify_file(rfile, wfile, tokens, words):
    fdr = open(rfile, "r")
    content = fdr.read()

    for token, word in zip(tokens, words):
        content = content.replace(token, str(word))
    
    fdr.close()
    fdw = open(wfile, "w")
    fdw.write(content)
    fdw.close()

def marking_on_map(loc,html_name="hello.html"):
    map = folium.Map(location=loc.loc[0].values, zoom_start=6)
    for i in range(loc.shape[0]):
        folium.Marker([loc.loc[i][0],loc.loc[i][1]], popup=loc.index[i]).add_to(map)
    map.save(html_name)