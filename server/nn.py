
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import warnings
import os
import sys
from datetime import datetime
import glob
from typing import Union, Tuple, Optional
from tqdm import tqdm

warnings.filterwarnings('ignore')

# CLI Art and Formatting
def print_banner():
    banner = """
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║               ██████╗ ██╗   ██╗██╗██╗     ██████╗                  ║
║               ██╔══██╗██║   ██║██║██║     ██╔══██╗                 ║
║               ██████╔╝██║   ██║██║██║     ██║  ██║                 ║
║               ██╔══██╗██║   ██║██║██║     ██║  ██║                 ║
║               ██████╔╝╚██████╔╝██║███████╗██████╔╝                 ║
║               ╚═════╝  ╚═════╝ ╚═╝╚══════╝╚═════╝                  ║
║                                                                    ║
║                              █████╗                                ║
║                             ██╔══██╗                               ║
║                 ███████╗    ███████║    ███████╗                   ║
║                 ╚══════╝    ██╔══██║    ╚══════╝                   ║
║                             ██║  ██║                               ║
║                             ╚═╝  ╚═╝                               ║
║                                                                    ║
║             ██████╗ ██████╗  █████╗ ██╗███╗   ██╗                  ║
║             ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║                  ║
║             ██████╔╝██████╔╝███████║██║██╔██╗ ██║                  ║
║             ██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║                  ║
║             ██████╔╝██║  ██║██║  ██║██║██║ ╚████║                  ║
║             ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝                  ║
║                                                                    ║
║                     Neural Network CLI Tool                        ║
║                        v1.0 - Interactive                          ║
╚════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_section(title):
    print(f"{'='*60}")
    print(f"  {title}")
    print('='*60)

def print_subsection(title):
    print(f"--- {title} ---")

### Activation Functions and Derivatives ###
def relu(x): 
    return np.maximum(0, x)

def relu_deriv(x): 
    return (x > 0).astype(float)

def sigmoid(x): 
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x): 
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x): 
    return np.tanh(x)

def tanh_deriv(x): 
    return 1 - np.tanh(x)**2

def softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(x_shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

def identity(x): 
    return x

def identity_deriv(x): 
    return np.ones_like(x)

ACTIVATIONS = {
    "relu": (relu, relu_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
    "identity": (identity, identity_deriv),
    "softmax": (softmax, None)
}

### Loss Functions ###
def mse(y_true, y_pred): 
    return np.mean((y_true - y_pred)**2)

def mse_deriv(y_true, y_pred): 
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_deriv(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

LOSSES = {
    "mse": (mse, mse_deriv),
    "cross_entropy": (cross_entropy, cross_entropy_deriv)
}

### Neural Network Classes ###
class Layer:
    def __init__(self, input_size, output_size, activation, learning_rate=0.01):
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.zeros((1, output_size))
        self.activation_name = activation
        self.activation, self.activation_deriv = ACTIVATIONS[activation]
        self.learning_rate = learning_rate

    def forward(self, x):
        self.input = x
        self.z = x @ self.weights + self.biases
        self.output = self.activation(self.z)
        return self.output

    def backward(self, grad_output):
        if self.activation_deriv:
            grad_activation = self.activation_deriv(self.z) * grad_output
        else:
            grad_activation = grad_output
        
        grad_weights = self.input.T @ grad_activation
        grad_biases = np.sum(grad_activation, axis=0, keepdims=True)
        grad_input = grad_activation @ self.weights.T
        
        grad_weights = np.clip(grad_weights, -1, 1)
        grad_biases = np.clip(grad_biases, -1, 1)
        
        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases
        return grad_input

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers, activations, loss, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_names = activations
        self.loss_name = loss
        self.loss_fn, self.loss_deriv = LOSSES[loss]
        self.learning_rate = learning_rate

        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            act = activations[i] if i < len(activations) else "identity"
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], act, learning_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    ### Neural Network Training ###
    def train(self, X, Y, epochs, validation_split=0.2):
        # Custom train-validation split
        if validation_split > 0:
            n_samples = X.shape[0]
            indices = np.arange(n_samples)
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(indices)
            split_idx = int(n_samples * (1 - validation_split))
            
            train_idx = indices[:split_idx]
            val_idx = indices[split_idx:]
            
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_val = X[val_idx]
            Y_val = Y[val_idx]
        else:
            X_train, Y_train = X, Y
            X_val, Y_val = None, None
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 50
        
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            # Forward pass on training data
            out = self.forward(X_train)
            
            # Check for NaN/inf in output
            if np.any(np.isnan(out)) or np.any(np.isinf(out)):
                print(f" Warning: NaN/inf detected in output at epoch {epoch}")
                break
            
            # Compute loss and gradient
            loss = self.loss_fn(Y_train, out)
            
            # Check for NaN/inf in loss
            if np.isnan(loss) or np.isinf(loss):
                print(f" Warning: NaN/inf loss at epoch {epoch}")
                break
            
            grad = self.loss_deriv(Y_train, out)
            self.backward(grad)
            
            # Validation step
            if X_val is not None:
                val_out = self.forward(X_val)
                val_loss = self.loss_fn(Y_val, val_out)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 100 == 0:
                    print(f" Epoch {epoch:4d} | Train Loss: {loss:.6f} | Val Loss: {val_loss:.6f}")
                
                if patience_counter >= patience:
                    print(f" Early stopping at epoch {epoch}")
                    break
            else:
                if epoch % 100 == 0:
                    print(f" Epoch {epoch:4d} | Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)

    def get_architecture_summary(self):
        layers_info = []
        for i, layer in enumerate(self.layers):
            layer_info = {
                'layer': i + 1,
                'input_size': layer.weights.shape[0],
                'output_size': layer.weights.shape[1],
                'activation': layer.activation_name,
                'parameters': layer.weights.size + layer.biases.size
            }
            layers_info.append(layer_info)
        
        total_params = sum(info['parameters'] for info in layers_info)
        return layers_info, total_params

### Data Utilities ###
def detect_problem_type(y_data, output_cols):
    if len(output_cols) == 0:
        return "unsupervised", None, None
    
    is_numeric = all(pd.api.types.is_numeric_dtype(y_data[col]) for col in output_cols)
    
    if is_numeric:
        unique_counts = [y_data[col].nunique() for col in output_cols]
        max_unique = max(unique_counts)
        
        if max_unique <= 10 and all(y_data[col].dtype == 'int64' for col in output_cols if y_data[col].dtype != 'float64'):
            if max_unique == 2:
                return "binary_classification", "cross_entropy", "sigmoid"
            else:
                return "multiclass_classification", "cross_entropy", "softmax"
        else:
            return "regression", "mse", "identity"
    else:
        unique_counts = [y_data[col].nunique() for col in output_cols]
        max_unique = max(unique_counts)
        
        if max_unique == 2:
            return "binary_classification", "cross_entropy", "sigmoid"
        else:
            return "multiclass_classification", "cross_entropy", "softmax"


def identify_columns(df):
    print_subsection("Column Identification")
    
    input_cols = [c for c in df.columns if c.lower().startswith(('x', 'feature', 'input'))]
    output_cols = [c for c in df.columns if c.lower().startswith(('y', 'target', 'output', 'label', 'class'))]
    
    if not input_cols and not output_cols:
        print(" No explicit column patterns found, using heuristics...")
        last_col = df.columns[-1]
        if df[last_col].dtype == 'object' or df[last_col].nunique() <= 20:
            output_cols = [last_col]
            input_cols = list(df.columns[:-1])
        else:
            input_cols = list(df.columns)
            output_cols = []
    
    elif not output_cols:
        output_cols = []
        if not input_cols:
            input_cols = list(df.columns)
    
    elif not input_cols:
        input_cols = [c for c in df.columns if c not in output_cols]
    
    print(f" Input columns ({len(input_cols)}): {input_cols}")
    print(f" Output columns ({len(output_cols)}): {output_cols}")
    
    return input_cols, output_cols

def clean_data(df):
    print_subsection("Data Cleaning")
    original_shape = df.shape
    print(f" Original shape: {original_shape}")
    
    # Remove entirely empty rows/columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    print(f" After removing empty rows/columns: {df.shape}")
    
    # Handle missing values
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f" Column '{col}' has {missing_count} missing values")
            
            if missing_count / len(df) > 0.5:
                print(f" Dropping column '{col}' (>50% missing)")
                df = df.drop(col, axis=1)
                continue
            
            # Check if column is categorical (object or low cardinality)
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
                # Numeric column
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f" Filled numeric column with median: {median_val}")
            except (ValueError, TypeError):
                # Categorical column
                mode_val = df[col].mode()
                if not mode_val.empty:
                    fill_value = mode_val[0]
                else:
                    fill_value = 'Unknown'
                df[col] = df[col].fillna(fill_value)
                print(f" Filled categorical column with mode: {fill_value}")
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f" Removed {duplicates} duplicate rows")
    
    # Final check for NaN
    if df.isnull().any().any():
        raise ValueError("NaN values remain after cleaning")
    
    return df

def preprocess_data(df, input_cols, output_cols):
    print_subsection("Data Preprocessing")
    
    if not input_cols:
        raise ValueError("No input columns identified. Please check your data or specify columns manually.")
    
    X_raw = df[input_cols].copy()
    Y_raw = df[output_cols].copy() if output_cols else pd.DataFrame()
    
    encoders = {}
    scalers = {}
    input_types = {}
    output_types = {}
    
    print("Processing input features...")
    for col in X_raw.columns:
        if X_raw[col].isnull().any():
            raise ValueError(f"NaN values found in column {col} after cleaning")
        
        try:
            X_raw[col] = pd.to_numeric(X_raw[col], errors='raise')
            print(f"  Numeric: {col}")
            input_types[col] = 'numeric'
            # Custom scaling: compute mean and std
            mean = X_raw[col].mean()
            std = X_raw[col].std()
            if std == 0:
                std = 1.0  # Avoid division by zero
            X_raw[col] = (X_raw[col] - mean) / std
            scalers[f'input_{col}'] = {'mean': mean, 'std': std}
        except (ValueError, TypeError):
            print(f" Encoding categorical: {col} -> {X_raw[col].unique()[:5]}...")
            input_types[col] = 'categorical'
            # Custom label encoding: map categories to integers
            categories = X_raw[col].unique().tolist()
            cat_to_int = {cat: idx for idx, cat in enumerate(categories)}
            X_raw[col] = X_raw[col].map(cat_to_int)
            encoders[f'input_{col}'] = {'classes': categories, 'mapping': cat_to_int}
    
    X = X_raw.values
    if np.any(np.isnan(X)):
        raise ValueError("NaN values introduced during input processing")
    
    Y = None
    if len(output_cols) > 0:
        print("Processing output features...")
        for col in Y_raw.columns:
            if Y_raw[col].isnull().any():
                raise ValueError(f"NaN values found in output column {col} after cleaning")
            
            try:
                Y_raw[col] = pd.to_numeric(Y_raw[col], errors='raise')
                print(f"  Numeric target: {col}")
                output_types[col] = 'numeric'
            except (ValueError, TypeError):
                print(f"  🏷️  Encoding target: {col} -> {Y_raw[col].unique()}")
                output_types[col] = 'categorical'
                # Custom label encoding
                categories = Y_raw[col].unique().tolist()
                cat_to_int = {cat: idx for idx, cat in enumerate(categories)}
                Y_raw[col] = Y_raw[col].map(cat_to_int)
                encoders[f'target_{col}'] = {'classes': categories, 'mapping': cat_to_int}
        
        Y = Y_raw.values
        
        if len(output_cols) == 1:
            unique_count = len(np.unique(Y))
            if unique_count <= 20 and output_types[output_cols[0]] == 'categorical':
                # Custom one-hot encoding
                categories = encoders[f'target_{output_cols[0]}']['classes']
                n_classes = len(categories)
                Y_onehot = np.zeros((Y.shape[0], n_classes))
                for i, val in enumerate(Y.flatten()):
                    Y_onehot[i, val] = 1
                Y = Y_onehot
                encoders[f'target_{output_cols[0]}']['n_classes'] = n_classes
                print(f"  One-hot encoded to shape: {Y.shape}")
    
    print(f"\nFinal preprocessing results:")
    print(f"  X shape: {X.shape}, range: [{np.nanmin(X):.3f}, {np.nanmax(X):.3f}]")
    if Y is not None:
        print(f"  Y shape: {Y.shape}, range: [{np.nanmin(Y):.3f}, {np.nanmax(Y):.3f}]")
    
    return X, Y, encoders, scalers, input_types, output_types

def load_data(path):
    print(f" Loading data from: {path}")
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")
    
    print(f" Raw data shape: {df.shape}")
    print(" Column types:")
    print(df.dtypes)
    
    df = clean_data(df)
    
    input_cols, output_cols = identify_columns(df)
    
    problem_type, suggested_loss, suggested_activation = detect_problem_type(df, output_cols)
    print_subsection("Problem Analysis")
    print(f" Detected problem type: {problem_type}")
    print(f" Suggested loss function: {suggested_loss}")
    print(f" Suggested final activation: {suggested_activation}")
    
    X, Y, encoders, scalers, input_types, output_types = preprocess_data(df, input_cols, output_cols)
    
    return X, Y, input_cols, output_cols, encoders, scalers, problem_type, suggested_loss, suggested_activation,  input_types, output_types


### Save/Load Model ###
def save_model_json(model, path, metadata=None):
    if os.path.exists(path):
        overwrite = input(f"⚠️ File {path} already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Save cancelled.")
            return False
    
    metadata = metadata or {}
    # Store encoder and scaler details in metadata
    if 'encoders' in metadata:
        metadata['input_encoders'] = {
            key: {
                'classes': metadata['encoders'][key]['classes'],
                'mapping': metadata['encoders'][key]['mapping'],
                'n_classes': metadata['encoders'][key].get('n_classes', None)
            } for key in metadata['encoders'] if key.startswith('input_')
        }
        metadata['target_encoders'] = {
            key: {
                'classes': metadata['encoders'][key]['classes'],
                'mapping': metadata['encoders'][key]['mapping'],
                'n_classes': metadata['encoders'][key].get('n_classes', None)
            } for key in metadata['encoders'] if key.startswith('target_')
        }
        del metadata['encoders']
    
    if 'scalers' in metadata:
        metadata['input_scalers'] = {
            key: {
                'mean': metadata['scalers'][key]['mean'],
                'std': metadata['scalers'][key]['std']
            } for key in metadata['scalers'] if key.startswith('input_')
        }
        del metadata['scalers']
    
    model_data = {
        "model_info": {
            "created_at": datetime.now().isoformat(),
            "version": "2.0"
        },
        "architecture": {
            "input_size": model.input_size,
            "output_size": model.output_size,
            "hidden_layers": [layer.weights.shape[1] for layer in model.layers[:-1]],
        },
        "training_config": {
            "activation_names": model.activation_names,
            "loss_function": model.loss_name,
            "learning_rate": model.learning_rate,
        },
        "layers": [
            {
                "weights": layer.weights.tolist(), 
                "biases": layer.biases.tolist(), 
                "activation": layer.activation_name,
                "input_size": layer.weights.shape[0],
                "output_size": layer.weights.shape[1]
            }
            for layer in model.layers
        ],
        "metadata": metadata
    }
    
    try:
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"Model saved to {path}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model_json(path_or_data: Union[str, dict]) -> Tuple[Optional['NeuralNetwork'], Optional[dict]]:
    try:
        if isinstance(path_or_data, str):
            with open(path_or_data, 'r') as f:
                data = json.load(f)
        elif isinstance(path_or_data, dict):
            data = path_or_data
        else:
            print(f"Error loading model: Expected str or dict, got {type(path_or_data)}")
            return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    if "architecture" in data:
        input_size = data["architecture"]["input_size"]
        output_size = data["architecture"]["output_size"]
        hidden_layers = data["architecture"].get("hidden_layers", [])
        activations = data["training_config"]["activation_names"]
        loss_fn = data["training_config"].get("loss_function", "mse")
        learning_rate = data["training_config"].get("learning_rate", 0.01)
    else:
        input_size = data["input_size"]
        output_size = data["output_size"]
        activations = data["activation_names"]
        loss_fn = "mse"
        learning_rate = data.get("learning_rate", 0.01)
        hidden_layers = []
    
    model = NeuralNetwork(input_size, output_size, hidden_layers, 
                         activations, loss_fn, learning_rate)
    model.layers = []
    
    for layer_data in data["layers"]:
        layer = Layer(layer_data["input_size"], layer_data["output_size"], 
                     layer_data["activation"], learning_rate)
        layer.weights = np.array(layer_data["weights"])
        layer.biases = np.array(layer_data["biases"])
        model.layers.append(layer)
    
    # Reconstruct encoders and scalers
    encoders = {}
    scalers = {}
    metadata = data.get("metadata", {})
    for key, enc_data in metadata.get("input_encoders", {}).items():
        encoders[key] = {
            'classes': enc_data['classes'],
            'mapping': enc_data['mapping'],
            'n_classes': enc_data.get('n_classes', None)
        }
    for key, enc_data in metadata.get("target_encoders", {}).items():
        encoders[key] = {
            'classes': enc_data['classes'],
            'mapping': enc_data['mapping'],
            'n_classes': enc_data.get('n_classes', None)
        }
    for key, scaler_data in metadata.get("input_scalers", {}).items():
        scalers[key] = {
            'mean': scaler_data['mean'],
            'std': scaler_data['std']
        }
    
    print(f"Model loaded successfully")
    return model, metadata

### Interactive Play Mode ###
def interactive_play_mode(model, metadata, scalers=None, encoders=None):
    print_section("INTERACTIVE PLAY MODE")
    
    layers_info, total_params = model.get_architecture_summary()
    print(f"Model Architecture:")
    for info in layers_info:
        print(f"   Layer {info['layer']}: {info['input_size']} → {info['output_size']} ({info['activation']})")
    print(f"Total Parameters: {total_params:,}")
    
    problem_type = metadata.get("problem_type", "unknown")
    input_cols = metadata.get("input_cols", [])
    input_types = metadata.get("input_types", {})
    output_types = metadata.get("output_types", {})
    output_cols = metadata.get("output_cols", [])
    print(f"Problem Type: {problem_type}")
    print(f"Expected Input Size: {model.input_size}")
    if input_cols:
        print(f"Input Columns: {input_cols}")
        print(f"Input Types:")
        for col in input_cols:
            if col in input_types and input_types[col] == 'categorical':
                valid_values = metadata.get("input_encoders", {}).get(f"input_{col}", {}).get("classes", [])
                print(f"   {col}: categorical {valid_values}")
            else:
                print(f"   {col}: numeric")
    print(f"Output Size: {model.output_size}")
    if output_types:
        print(f"Output Types:")
        for col in output_types:
            print(f"   {col}: {output_types[col]}")
    
    print(f"\n{'='*50}")
    print(" Enter input values (comma-separated) or 'quit' to exit")
    if input_cols:
        example = ','.join(['0.0' if input_types.get(col, 'numeric') == 'numeric' else metadata.get('input_encoders', {}).get(f'input_{col}', {}).get('classes', ['value'])[0] for col in input_cols])
        print(f" Example: {example}")
    print('='*50)
    
    while True:
        try:
            user_input = input(f"\nInput ({model.input_size} values): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                save_choice = input("\nWould you like to save the model? (y/n): ").strip().lower()
                if save_choice == 'y':
                    filename = input("Enter filename for model (e.g., model.json): ").strip()
                    if not filename.endswith('.json'):
                        filename += '.json'
                    save_model_json(model, filename, metadata)
                break
            
            # Split input by commas and strip whitespace
            input_values = [val.strip() for val in user_input.split(',')]
            if len(input_values) != model.input_size:
                print(f"Error: Expected {model.input_size} values, got {len(input_values)}")
                continue
            
            # Preprocess input
            processed_inputs = []
            for i, (value, col) in enumerate(zip(input_values, input_cols)):
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
                    print(f"Error: NaN values in processed input")
                    continue
                print(f"Processed input: {input_data}")
                prediction = model.predict(input_data)
                
                print(f"Raw prediction: {prediction.flatten()}")
                
                if problem_type == "binary_classification" or problem_type == "multiclass_classification":
                    pred_class = np.argmax(prediction, axis=1)[0]
                    confidence = prediction[0][pred_class]
                    if output_cols and encoders and f'target_{output_cols[0]}' in encoders:
                        classes = encoders[f'target_{output_cols[0]}']['classes']
                        class_label = classes[pred_class]
                        print(f" class: {class_label} ({pred_class})")
                    else:
                        print(f" class: {pred_class}")
                        print(f"Warning: No target encoder found, displaying numeric class")
                    print(f"Confidence: {confidence:.4f}")
                    print(f"All probabilities: {prediction[0]}")
                
                elif problem_type == "regression":
                    print(f"Predicted value: {prediction[0]}")
                
                else:
                    print(f"Model output: {prediction[0]}")
                
        except KeyboardInterrupt:
            print(f"Goodbye!")
            save_choice = input("Would you like to save the model? (y/n): ").strip().lower()
            if save_choice == 'y':
                filename = input("Enter filename for model (e.g., model.json): ").strip()
                if not filename.endswith('.json'):
                    filename += '.json'
                save_model_json(model, filename, metadata)
            break
        except Exception as e:
            print(f"Error during prediction: {e}")


def predict_from_csv(model, metadata, scalers, encoders, csv_path):
    print_section("BATCH PREDICTION FROM CSV")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    input_cols = metadata.get("input_cols", [])
    input_types = metadata.get("input_types", {})
    output_cols = metadata.get("output_cols", [])[:1]  # Assume single output
    problem_type = metadata.get("problem_type", "unknown")
    
    # Validate input columns
    missing_cols = [col for col in input_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # Filter to required columns
    df = df[input_cols].copy()
    
    # Preprocess inputs
    print("Preprocessing inputs...")
    for col in input_cols:
        if df[col].isnull().any():
            print(f"❓ Warning: NaN values in column {col}. Filling with default values.")
            if input_types.get(col) == 'categorical':
                # Use first category or 'Unknown'
                default = encoders.get(f'input_{col}', {}).get('classes', ['Unknown'])[0]
                df[col] = df[col].fillna(default)
            else:
                # Use mean from scaler or 0.0
                default = scalers.get(f'input_{col}', {}).get('mean', 0.0)
                df[col] = df[col].fillna(default)
        
        if input_types.get(col) == 'categorical':
            encoder_key = f'input_{col}'
            if encoder_key in encoders:
                mapping = encoders[encoder_key]['mapping']
                valid_categories = list(mapping.keys())
                # Check for invalid categories
                invalid_values = df[col][~df[col].isin(valid_categories)].unique()
                if len(invalid_values) > 0:
                    print(f"Error: Invalid categories in {col}: {invalid_values}")
                    return
                df[col] = df[col].map(mapping)
            else:
                print(f"Error: No encoder found for categorical column {col}")
                return
        else:
            try:
                df[col] = df[col].astype(float)
                # Apply scaling
                scaler_key = f'input_{col}'
                if scaler_key in scalers:
                    mean = scalers[scaler_key]['mean']
                    std = scalers[scaler_key]['std']
                    df[col] = (df[col] - mean) / std if std != 0 else df[col]
            except ValueError:
                print(f"Error: Non-numeric values in {col}")
                return
    
    X = df.values
    if np.any(np.isnan(X)):
        print(f"Error: NaN values in processed input")
        return
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(X)
    
    # Prepare output DataFrame
    output_df = df.copy()
    
    if problem_type == "binary_classification" or problem_type == "multiclass_classification":
        pred_classes = np.argmax(predictions, axis=1)
        confidences = predictions[np.arange(len(predictions)), pred_classes]
        
        if output_cols and encoders and f'target_{output_cols[0]}' in encoders:
            classes = encoders[f'target_{output_cols[0]}']['classes']
            pred_labels = [classes[idx] for idx in pred_classes]
        else:
            pred_labels = pred_classes
            print(f"Warning: No target encoder found, using numeric classes")
        
        output_df[f'predicted_{output_cols[0]}'] = pred_labels
        output_df[f'confidence_{output_cols[0]}'] = confidences
    else:
        output_df[f'predicted_{output_cols[0]}'] = predictions.flatten()
    
    # Save predictions to CSV
    output_path = csv_path.rsplit('.', 1)[0] + '_predictions.csv'
    try:
        output_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
    
    # Display sample
    print(f"\nSample predictions (first 5 rows):")
    print(output_df.head().to_string())


### Utility Functions ###
def list_saved_models():
    print_section("SAVED MODELS")
    model_files = glob.glob("*.json")
    
    if not model_files:
        print(" No saved models found.")
        return
    
    for i, file in enumerate(model_files, 1):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            created_at = data.get("model_info", {}).get("created_at", "Unknown")
            architecture = data.get("architecture", {})
            input_size = architecture.get("input_size", "Unknown")
            output_size = architecture.get("output_size", "Unknown")
            hidden_layers = architecture.get("hidden_layers", [])
            problem_type = data.get("metadata", {}).get("problem_type", "Unknown")
            
            print(f" {i}. {file}")
            print(f"    Created: {created_at}")
            print(f"    Architecture: {input_size} → {hidden_layers} → {output_size}")
            print(f"    Problem Type: {problem_type}")
            print()
        except Exception as e:
            print(f" {i}. {file} (Error loading: {e})")
            print()

def help_menu():
    print_section("HELP")
    print(" Welcome to the Universal ML Training Tool v2.0")
    print(" Main Menu Options:")
    print(" 1. Train a new model: Load a CSV file, configure, and train a neural network.")
    print(" 2. Load existing model: Load a saved model and use it for predictions.")
    print(" 3. List saved models: View all saved model files and their details.")
    print(" 4. Exit: Quit the application.")
    print(" Training a Model:")
    print(" - Provide a CSV file with input and optional target columns.")
    print(" - Configure hidden layers, learning rate, epochs, and loss function.")
    print(" - After training, choose to play with the model or save it.")
    print(" Playing with a Model:")
    print(" - Enter input values to get predictions.")
    print(" - Use 'quit' to exit play mode and optionally save the model.")
    print(" Saving/Loading Models:")
    print(" - Models are saved as JSON files with architecture and weights.")
    print(" - Load saved models to make predictions or view details.")
    print(" Press Enter to return to the main menu.")

def main_menu():
    print_banner()
    
    while True:
        print_section("MAIN MENU")
        print(" 1. Train a new model")
        print(" 2. Load existing model and play")
        print(" 3. List saved models")
        print(" 4. Help")
        print(" 5. Exit")
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == '1':
            train_model_interactive()
        elif choice == '2':
            load_and_play_interactive()
        elif choice == '3':
            list_saved_models()
        elif choice == '4':
            help_menu()
        elif choice == '5':
            print(" Goodbye!")
            sys.exit(0)
        else:
            print(" Invalid choice. Please select 1-5.")


### Training Function ###
def train_model_interactive():
    print_section("🏋️ MODEL TRAINING")
    
    train_file = input("Enter path to training CSV file: ").strip()
    if not os.path.exists(train_file):
        print(f"File not found: {train_file}")
        return
    
    try:
        df = pd.read_csv(train_file)
        df = clean_data(df)
        input_cols = [col for col in df.columns if col.startswith('x')]
        output_cols = [col for col in df.columns if col.startswith('y')]
        
        X, Y, encoders, scalers, input_types, output_types = preprocess_data(df, input_cols, output_cols)
        problem_type, suggested_loss, suggested_activation = detect_problem_type(df[output_cols], output_cols)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if problem_type == "unsupervised":
        print("Unsupervised learning detected - creating autoencoder")
        Y = X.copy()
        suggested_loss = "mse"
        suggested_activation = "sigmoid"
    
    print_section("MODEL CONFIGURATION")
    
    hidden_input = input(f"Hidden layers (default: 64 32): ").strip()
    if hidden_input:
        try:
            hidden_layers = [int(x) for x in hidden_input.split()]
        except ValueError:
            print("Invalid input, using default: [64, 32]")
            hidden_layers = [64, 32]
    else:
        hidden_layers = [64, 32]
    
    lr_input = input(f"Learning rate (default: 0.01): ").strip()
    learning_rate = float(lr_input) if lr_input else 0.01
    
    epochs_input = input(f"Epochs (default: 1000): ").strip()
    epochs = int(epochs_input) if epochs_input else 1000
    
    loss_input = input(f"Loss function (default: {suggested_loss}): ").strip()
    loss_fn = loss_input if loss_input and loss_input in LOSSES else suggested_loss
    
    num_layers = len(hidden_layers) + 1
    activations = ["relu"] * (num_layers - 1) + [suggested_activation]
    
    print(f"\nConfiguration Summary:")
    print(f"  Architecture: {X.shape[1]} → {hidden_layers} → {Y.shape[1] if Y is not None else X.shape[1]}")
    print(f"  Activations: {activations}")
    print(f"  Loss function: {loss_fn}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    
    model = NeuralNetwork(X.shape[1], Y.shape[1] if Y is not None else X.shape[1], 
                         hidden_layers, activations, loss_fn, learning_rate)
    
    print_section("TRAINING")
    model.train(X, Y, epochs)
    
    metadata = {
        "problem_type": problem_type,
        "input_cols": input_cols,
        "output_cols": output_cols,
        "input_types": input_types,
        "output_types": output_types,
        "encoders": encoders,
        "scalers": scalers
    }
    
    print_section("TRAINING COMPLETE")
    while True:
        print("\n1. Play with model (single input)")
        print("2. Predict from CSV (batch prediction)")
        print("3. Save model")
        print("4. Return to main menu")
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            interactive_play_mode(model, metadata, scalers, encoders)
        elif choice == '2':
            csv_path = input("Enter path to input CSV file: ").strip()
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
            else:
                predict_from_csv(model, metadata, scalers, encoders, csv_path)
        elif choice == '3':
            filename = input("Enter filename for model (e.g., model.json): ").strip()
            if not filename.endswith('.json'):
                filename += '.json'
            save_model_json(model, filename, metadata)
        elif choice == '4':
            save_choice = input("\nWould you like to save the model before returning? (y/n): ").strip().lower()
            if save_choice == 'y':
                filename = input("Enter filename for model (e.g., model.json): ").strip()
                if not filename.endswith('.json'):
                    filename += '.json'
                save_model_json(model, filename, metadata)
            break
        else:
            print("Invalid choice. Please select 1-4.")

def load_and_play_interactive():
    print_section("LOAD AND PLAY")
    filename = input("Enter model filename (e.g., model.json): ").strip()
    if not os.path.exists(filename):
        print(f" File not found: {filename}")
        return
    
    model, metadata = load_model_json(filename)
    if model is None:
        return
    
    scalers = {}
    encoders = {}
    while True:
        print("\n1. Play with model (single input)")
        print("2. Predict from CSV (batch prediction)")
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            interactive_play_mode(model, metadata, scalers, encoders)
        elif choice == '2':
            csv_path = input("Enter path to input CSV file: ").strip()
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
            else:
                predict_from_csv(model, metadata, scalers, encoders, csv_path)

        else:
            print("Invalid choice. Please select 1-2.")
 
if __name__ == "__main__":
    main_menu()