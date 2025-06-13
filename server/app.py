from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import glob
import numpy as np
import pandas as pd
from nn import NeuralNetwork, load_data, save_model_json, load_model_json, detect_problem_type, preprocess_data
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        nn_files = glob.glob("*.nn")
        for file in nn_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error removing file {file}: {e}")
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({'error': 'Invalid file, please upload a CSV'}), 400

        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file.save(temp_path)
        df = pd.read_csv(temp_path)
        print("Parsed columns:", df.columns.tolist())
        print("head", df.head())

        hidden_layers = request.form.get('hidden_layers', '64 32')
        learning_rate = float(request.form.get('learning_rate', 0.01))
        epochs = int(request.form.get('epochs', 1000))
        loss_function = request.form.get('loss_function', 'mse')

        try:
            X, Y, input_cols, output_cols, encoders, scalers, problem_type, suggested_loss, suggested_activation, input_types, output_types = load_data(temp_path)
        except Exception as e:
            os.remove(temp_path)
            return jsonify({'error': f'Error loading data: {str(e)}'}), 400

        if problem_type == "unsupervised":
            Y = X.copy()
            suggested_loss = "mse"
            suggested_activation = "sigmoid"

        hidden_layers_array = [int(x) for x in hidden_layers.split() if x.strip().isdigit()]
        if not hidden_layers_array:
            hidden_layers_array = [64, 32]

        num_layers = len(hidden_layers_array) + 1
        activations = ["relu"] * (num_layers - 1) + [suggested_activation]

        model = NeuralNetwork(
            input_size=X.shape[1],
            output_size=Y.shape[1] if Y is not None else X.shape[1],
            hidden_layers=hidden_layers_array,
            activations=activations,
            loss=loss_function,
            learning_rate=learning_rate
        )

        model.train(X, Y, epochs,validation_split=0.2)

        metadata = {
            "problem_type": problem_type,
            "input_cols": input_cols,
            "output_cols": output_cols,
            "input_types": input_types,
            "output_types": output_types,
            "encoders": encoders,
            "scalers": scalers
        }

        model_filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nn"
        save_model_json(model, model_filename, metadata)

        with open(model_filename, 'r') as f:
            model_json = json.load(f)

        os.remove(temp_path)

        return jsonify(model_json), 200

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'model' not in data or 'inputs' not in data:
            return jsonify({'error': 'Model JSON and inputs required'}), 400

        model_json = data['model']
        inputs = data['inputs']
        print(f"Received model JSON: {model_json['training_config']}")
        # Validate inputs
        if not isinstance(inputs, list) or len(inputs) != model_json['architecture']['input_size']:
            return jsonify({'error': f'Expected {model_json["architecture"]["input_size"]} inputs'}), 400

        # Load model
        model, metadata = load_model_json(model_json)
        print(f"Received model JSON: {model.loss_fn}")
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 500

        # Preprocess inputs using metadata
        encoders = metadata.get('encoders', {})
        scalers = metadata.get('input_scalers', {})
        input_cols = metadata.get('input_cols', [])
        input_types = metadata.get("input_types", {})
        # Create DataFrame for input
        if len(inputs) != model.input_size:
                return jsonify({"Error": f"Expected {model.input_size} values, got {len(inputs)}"}),400
                
            
        # Preprocess input
        processed_inputs = []
        for i, (value, col) in enumerate(zip(inputs, input_cols)):
            col_type = input_types.get(col, 'numeric')
       
            if col_type == 'categorical':
                encoder_key = f'input_{col}'
                if encoder_key in encoders:
                    try:
                        mapping = encoders[encoder_key]['mapping']
                        if value not in mapping:
                            print(f"Error: '{value}' not in valid categories for {col}: {list(mapping.keys())}")
                            break
                        processed_value = mapping[value]
                    except Exception as e:
                        print(f"Error encoding {col}: {e}")
                        break
                else:
                    print(f"Error: No encoder found for categorical column {col}")
                    break
            else:
                try:
                    processed_value = float(value)
                    # Apply scaling
                    scaler_key = f'input_{col}'
                    if scaler_key in scalers:
                        mean = scalers[scaler_key]['mean']
                        std = scalers[scaler_key]['std']
                        processed_value = (processed_value - mean) / std if std != 0 else processed_value
              
                except ValueError:
                    print(f"Error: Expected numeric value for {col}, got '{value}'")
                    break
            processed_inputs.append(processed_value)
        else:  # Only proceed if no errors in loop
            input_data = np.array(processed_inputs).reshape(1, -1)
            if np.any(np.isnan(input_data)):
                return jsonify({"Error": "NaN values in processed input"}), 400
      
            prediction = model.predict(input_data).tolist()
            print(f"Prediction: {prediction}")
        return jsonify({'prediction': prediction}), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)