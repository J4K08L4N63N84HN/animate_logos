""" These functions add ids into SVG file paths.
"""

import os
from pathlib import Path
from xml.dom import minidom


def insert_ids_in_folder(old_folder, new_folder="../../data/interim/logos_preprocessed"):
    """ Add the attribute "animation_id" to all SVGs in a given folder.

    Args:
        old_folder (str): Path of folder containing all SVGs without animation ID.
        new_folder (str): Target directory where SVGs with animation ID are saved.

    """
    for file in os.listdir(old_folder):
        if file.endswith(".svg"):
            insert_id(old_folder + "/" + file, new_folder)


def insert_id(svg_file, new_folder="data/interim/logos_id_added"):
    """ Add the attribute "animation_id" to all elements of a given SVG.

       Args:
           new_folder (str): Folder where the new files should be saved.
           svg_file (str): Path of SVG file.

    """
    Path(new_folder).mkdir(parents=True, exist_ok=True)
    filename = svg_file.replace('.svg', '').split("/")[-1]
    doc = minidom.parse(svg_file)
    # Store all elements in list
    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
        'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
        'rect') + doc.getElementsByTagName('text')
    for i in range(len(elements)):
        elements[i].setAttribute('animation_id', str(i))
    # write svg
    textfile = open(new_folder + '/' + filename + '.svg', 'wb')
    textfile.write(doc.toprettyxml(encoding="iso-8859-1"))  # needed to handle "Umlaute"
    textfile.close()
    doc.unlink()
