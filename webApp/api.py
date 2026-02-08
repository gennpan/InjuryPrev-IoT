from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import pandas as pd
import os
import shutil


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Feature di interesse che mostro nella dash
FEATURES = [
    "player_id",
    "date",
    "speed_mean", "speed_max", "speed_std",
    "acc_norm_mean", "acc_norm_max", "acc_norm_std",
    "gyro_norm_mean", "gyro_norm_max",
    "n_samples"
]

def load_team_data(team_id, csv_file, limit=10000):
    team_dir = os.path.join(DATA_DIR, str(team_id))
    csv_path = os.path.join(team_dir, csv_file)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV non trovato: {csv_file} per team_id={team_id}")

    df = pd.read_csv(csv_path)

    return df


@app.route("/")
def home_page():
    return render_template("dashboard.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/data-view")
def data_view():
    return render_template("data_view.html")

@app.route("/api/team/<team_id>/files")
def api_team_files(team_id):

    team_dir = os.path.join(DATA_DIR, team_id)

    if not os.path.exists(team_dir):
        return jsonify([])

    files = [
        f for f in os.listdir(team_dir)
        if f.endswith(".csv")
    ]

    return jsonify(files)

@app.route("/api/teams")
def api_teams():

    if not os.path.exists(DATA_DIR):
        return jsonify([])

    teams = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    return jsonify(sorted(teams))

@app.route("/api/upload", methods=["POST"])
def upload_csv():

    team_name = request.form.get("team_name")
    file = request.files.get("csv_file")

    if not team_name or not file:
        return jsonify({"error": "Nome team o file mancante"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Sono accettati solo file CSV"}), 400

    team_dir = os.path.join(DATA_DIR, team_name)

    # crea cartella se non esiste
    os.makedirs(team_dir, exist_ok=True)

    save_path = os.path.join(team_dir, file.filename)

    file.save(save_path)

    return jsonify({
        "message": f"Team '{team_name}' creato e file caricato correttamente."
    })

@app.route("/api/team/<team_id>")
def api_team_features(team_id):
    csv_file = request.args.get("file")
    if not csv_file:
        return jsonify({"error": "Devi specificare il parametro 'file'"}), 400

    try:
        df = load_team_data(team_id, csv_file, limit=10000)
        return jsonify(df.to_dict(orient="records"))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

@app.route("/api/team/<team_id>/file/<filename>", methods=["DELETE"])
def delete_csv(team_id, filename):

    team_id = secure_filename(team_id)
    filename = secure_filename(filename)

    csv_path = os.path.join(DATA_DIR, team_id, filename)

    if not os.path.exists(csv_path):
        return jsonify({"error": "File non trovato"}), 404

    os.remove(csv_path)

    return jsonify({"message": "File eliminato"}), 200


@app.route("/api/team/<team_id>", methods=["DELETE"])
def delete_team(team_id):

    team_id = secure_filename(team_id)
    team_path = os.path.join(DATA_DIR, team_id)

    if not os.path.exists(team_path):
        return jsonify({"error": "Team non trovato"}), 404

    shutil.rmtree(team_path)

    return jsonify({"message": "Team eliminato"}), 200



if __name__ == "__main__":
    app.run(debug=True)
