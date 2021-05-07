"""This module creates a Flask app to build a website where a user can upload an SVG logo and receive two animated
versions of the logo. One animation is created using the genetic algorithm approach, and the other animation is
created using the entmoot optimiser approach.
"""

import os

from flask import Flask, flash, request, redirect, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename

from src.pipeline import Logo

UPLOAD_FOLDER = "data/raw/uploads"
ALLOWED_EXTENSIONS = {"svg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = ALLOWED_EXTENSIONS
app.config["SECRET_KEY"] = "X38DJ0MEJKFNWEDK2342343249"


def allowed_file(filename):
    """ The functions checks if the uploaded file is an SVG.

    Args:
        filename (str): Name of an uploaded file

    Returns:
        (boolean): True if the uploaded file is an SVG
    """
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Not allowed file format")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            try:
                logo = Logo(data_dir=os.path.join(app.config["UPLOAD_FOLDER"], filename))
                logo.animate()
                animation_filename_genetic = filename.replace(".svg", "_animated_ga.svg")
                animation_filename_entmoot = filename.replace(".svg", "_animated_entmoot.svg")
                return render_template("show.html", filename=filename,
                                       animation_filename_genetic=animation_filename_genetic,
                                       animation_filename_entmoot=animation_filename_entmoot)
            except:
                flash("SVG file could not be processed")
                return redirect(request.url)
    return render_template("index.html")


@app.route("/data/raw/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"],
                               filename)


@app.route("/data/raw/uploads/<animation_filename_genetic>")
def animated_svg_genetic(animation_filename_genetic):
    return send_from_directory(app.config["UPLOAD_FOLDER"],
                               animation_filename_genetic)


@app.route("/data/raw/uploads/<animation_filename_entmoot>")
def animated_svg_entmoot(animation_filename_entmoot):
    return send_from_directory(app.config["UPLOAD_FOLDER"],
                               animation_filename_entmoot)


@app.route("/redirect_button")
def redirect_button():
    return redirect(url_for("index", _anchor="upload_svg_logo"))


if __name__ == "__index__":
    app.debug = True
    app.run()
