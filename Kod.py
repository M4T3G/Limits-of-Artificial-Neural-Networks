import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random
import os
import pickle
from datetime import datetime

# Grafikleri kaydetmek için klasör oluştur
if not os.path.exists('model_plots'):
    os.makedirs('model_plots')

# Dataset kaydetme ve yükleme fonksiyonları
def save_dataset(X_train, X_test, y_train, y_test, filename):
    """Dataseti diske kaydeder"""
    with open(filename, 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)

def load_dataset(filename):
    """Diskten dataset yükler"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_dataset(problem_type, num_samples=1000, force_generate=False):
    """Dataseti yükler veya oluşturur"""
    filename = f'dataset_{problem_type}.pkl'
    
    if not force_generate and os.path.exists(filename):
        print(f"Loading existing dataset for problem {problem_type}")
        return load_dataset(filename)
    else:
        print(f"Generating new dataset for problem {problem_type}")
        X_train, X_test, y_train, y_test = create_dataset(problem_type, num_samples)
        save_dataset(X_train, X_test, y_train, y_test, filename)
        return X_train, X_test, y_train, y_test

## 1. Veri Kümesi Oluşturma Fonksiyonları 

def generate_two_points(size=25):
    """A) İki nokta arası mesafe için veri oluşturur"""
    matrix = np.zeros((size, size))
    x1, y1 = random.randint(0, size-1), random.randint(0, size-1)
    x2, y2 = random.randint(0, size-1), random.randint(0, size-1)
    matrix[x1, y1] = 1
    matrix[x2, y2] = 1
    dist = distance.euclidean((x1, y1), (x2, y2))
    return matrix, dist

def generate_n_points(size=25, min_points=3, max_points=10):
    """B ve C) Nokta kümesi oluşturur"""
    matrix = np.zeros((size, size))
    num_points = random.randint(min_points, max_points)
    points = []
    for _ in range(num_points):
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        matrix[x, y] = 1
        points.append((x, y))
    
    # En yakın ve en uzak çiftleri bul
    min_dist = float('inf')
    max_dist = 0
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = distance.euclidean(points[i], points[j])
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
    return matrix, min_dist, max_dist, num_points

def generate_squares(size=25, min_squares=1, max_squares=10):
    """E) Kareler oluşturur"""
    matrix = np.zeros((size, size))
    num_squares = random.randint(min_squares, max_squares)
    
    for _ in range(num_squares):
        side = random.randint(2, 6)  # Kare kenar uzunluğu
        x = random.randint(0, size - side)
        y = random.randint(0, size - side)
        
        # Kareyi çiz
        matrix[x:x+side, y:y+side] = 1
    
    return matrix, num_squares

def create_dataset(problem_type, num_samples=1000):
    """Problem tipine göre veri kümesi oluşturur"""
    X = []
    y = []
    
    for _ in range(num_samples):
        if problem_type == 'A':
            matrix, dist = generate_two_points()
            X.append(matrix)
            y.append(dist)
        elif problem_type == 'B':
            matrix, min_dist, _, _ = generate_n_points()
            X.append(matrix)
            y.append(min_dist)
        elif problem_type == 'C':
            matrix, _, max_dist, _ = generate_n_points()
            X.append(matrix)
            y.append(max_dist)
        elif problem_type == 'D':
            matrix, _, _, num_points = generate_n_points(min_points=1)
            X.append(matrix)
            y.append(num_points)
        elif problem_type == 'E':
            matrix, num_squares = generate_squares()
            X.append(matrix)
            y.append(num_squares)
    
    X = np.array(X).reshape(-1, 25, 25, 1)
    y = np.array(y)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

## 2. Model Tanımlama Fonksiyonları 

def create_model_A():
    """Problem A için CNN modeli"""
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(25, 25, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    
    optimizer = optimizers.Adam(learning_rate=0.0001)
    loss = 'mse'
    metrics = ['mae']
    
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model, optimizer, loss, metrics

def create_model_B():
    """Problem B için Transformer modeli"""
    inputs = layers.Input(shape=(25, 25, 1))
    
    # CNN özellik çıkarıcı
    x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.GlobalMaxPooling2D()(x)
    
    # Transformer
    x = layers.Reshape((-1, 256))(x)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Çıktı katmanı
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs, outputs)
    
    optimizer = 'adam'
    loss = 'mse'
    metrics = ['mae']
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model, optimizer, loss, metrics

def create_model_C():
    """Problem C için MLP modeli"""
    model = models.Sequential([
        layers.Flatten(input_shape=(25, 25, 1)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    optimizer = optimizers.Adam(learning_rate=0.0005)
    loss = 'mse'
    metrics = ['mae']
    
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model, optimizer, loss, metrics

def create_model_D():
    """Problem D için CNN modeli"""
    model = models.Sequential([
        # Giriş şeklini (25, 25, 1) olarak ayarla
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(25, 25, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(11, activation='softmax')  # 0-10 arası sınıflar
    ])
    
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
    return model, optimizer, loss, metrics

def create_model_E():
    """Problem E için CNN Modeli"""
    model = models.Sequential([
        # 1. Katman
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(25, 25, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # 2. Katman
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # 3. Katman
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Global Pooling ile boyut sorununu çözüyoruz
        layers.GlobalAveragePooling2D(),
        
        # Yoğun katmanlar
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dense(11, activation='softmax')  # 0-10 arası sınıflar
    ])
    
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
    return model, optimizer, loss, metrics

## 3. Problemler için Eğitim ve Değerlendirme Fonksiyonları

def analyze_test_results(model, X_test, y_test, problem_type):
    """Test sonuçlarını detaylı analiz eder ve kaydeder"""
    predictions = model.predict(X_test)
    
    # Sonuçları kaydetmek için klasör oluştur
    if not os.path.exists('test_results'):
        os.makedirs('test_results')
    
    # Doğru ve yanlış tahminleri analiz et
    if problem_type in ['D', 'E']:  # Sınıflandırma problemleri
        predictions = np.argmax(predictions, axis=1)
        correct_indices = np.where(predictions == y_test)[0]
        wrong_indices = np.where(predictions != y_test)[0]
        
        # Doğru tahminlerin analizi
        correct_examples = []
        for idx in correct_indices[:5]:  # İlk 5 doğru tahmin
            correct_examples.append({
                'input': X_test[idx].reshape(25, 25),
                'true_label': y_test[idx],
                'predicted_label': predictions[idx]
            })
        
        # Yanlış tahminlerin analizi
        wrong_examples = []
        for idx in wrong_indices[:5]:  # İlk 5 yanlış tahmin
            wrong_examples.append({
                'input': X_test[idx].reshape(25, 25),
                'true_label': y_test[idx],
                'predicted_label': predictions[idx]
            })
    else:  # Regresyon problemleri
        # Tahmin hatalarını hesapla
        errors = np.abs(predictions.flatten() - y_test)
        worst_indices = np.argsort(errors)[-5:]  # En kötü 5 tahmin
        best_indices = np.argsort(errors)[:5]    # En iyi 5 tahmin
        
        # En iyi tahminlerin analizi
        correct_examples = []
        for idx in best_indices:
            correct_examples.append({
                'input': X_test[idx].reshape(25, 25),
                'true_value': y_test[idx],
                'predicted_value': predictions[idx][0],
                'error': errors[idx]
            })
        
        # En kötü tahminlerin analizi
        wrong_examples = []
        for idx in worst_indices:
            wrong_examples.append({
                'input': X_test[idx].reshape(25, 25),
                'true_value': y_test[idx],
                'predicted_value': predictions[idx][0],
                'error': errors[idx]
            })
    
    # Sonuçları görselleştir ve kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Doğru tahminleri görselleştir
    plt.figure(figsize=(20, 15))  # Boyutu artırdık
    for i, example in enumerate(correct_examples):
        plt.subplot(2, 5, i+1)
        plt.imshow(example['input'], cmap='binary')
        plt.xticks(range(25), fontsize=8)  # Font boyutunu küçülttük
        plt.yticks(range(25), fontsize=8)  # Font boyutunu küçülttük
        plt.grid(True, which='both', color='gray', linewidth=0.5)  # Izgara ekledik
        if problem_type in ['D', 'E']:
            plt.title(f'True: {example["true_label"]}\nPred: {example["predicted_label"]}', fontsize=10)
        else:
            plt.title(f'True: {example["true_value"]:.2f}\nPred: {example["predicted_value"]:.2f}\nError: {example["error"]:.2f}', fontsize=10)
    plt.suptitle(f'Problem {problem_type} - Correct Predictions', fontsize=12)
    plt.tight_layout()  # Alt grafikler arasındaki boşlukları ayarla
    plt.savefig(f'test_results/{problem_type}_correct_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Yanlış tahminleri görselleştir
    plt.figure(figsize=(20, 15))  # Boyutu artırdık
    for i, example in enumerate(wrong_examples):
        plt.subplot(2, 5, i+1)
        plt.imshow(example['input'], cmap='binary')
        plt.xticks(range(25), fontsize=8)  # Font boyutunu küçülttük
        plt.yticks(range(25), fontsize=8)  # Font boyutunu küçülttük
        plt.grid(True, which='both', color='gray', linewidth=0.5)  # Izgara ekledik
        if problem_type in ['D', 'E']:
            plt.title(f'True: {example["true_label"]}\nPred: {example["predicted_label"]}', fontsize=10)
        else:
            plt.title(f'True: {example["true_value"]:.2f}\nPred: {example["predicted_value"]:.2f}\nError: {example["error"]:.2f}', fontsize=10)
    plt.suptitle(f'Problem {problem_type} - Wrong Predictions', fontsize=12)
    plt.tight_layout()  # Alt grafikler arasındaki boşlukları ayarla
    plt.savefig(f'test_results/{problem_type}_wrong_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return correct_examples, wrong_examples

def train_and_evaluate_problem(problem_type, model_creator, num_samples=1000, dataset_sizes=[200, 400, 800]):
    """Tek bir problemi çalıştırır"""
    # Dataseti yükle veya oluştur
    X_train_full, X_test, y_train_full, y_test = get_dataset(problem_type, num_samples)
    
    results = {}
    history_dict = {}  # Eğitim geçmişlerini saklamak için
    
    for size in dataset_sizes:
        print(f"\nTraining with {size} samples for problem {problem_type}")
        
        # Veri alt kümesini seç
        X_train = X_train_full[:size]
        y_train = y_train_full[:size]
        
        # Modeli oluştur ve hiperparametreleri al
        model, optimizer, loss, metrics = model_creator()
        
        # Model özeti
        print(f"\nModel Hyperparameters for Problem {problem_type}:")
        print(f"Optimizer: {optimizer}")
        print(f"Loss function: {loss}")
        print(f"Metrics: {metrics}")
        
        # Eğitim
        history = model.fit(X_train, y_train, 
                           epochs=50, 
                           batch_size=32, 
                           validation_split=0.2,
                           verbose=1)
        
        # Eğitim geçmişini sakla
        history_dict[size] = {
            'history': history.history,
            'hyperparams': {
                'optimizer': str(optimizer),
                'loss': loss,
                'metrics': metrics
            }
        }
        
        # Değerlendirme
        if problem_type in ['D', 'E']:  # Sınıflandırma problemleri
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            results[size] = accuracy
            print(f"Test Accuracy with {size} samples: {accuracy:.4f}")
        else:  # Regresyon problemleri
            loss, mae = model.evaluate(X_test, y_test, verbose=0)
            results[size] = mae
            print(f"Test MAE with {size} samples: {mae:.4f}")
        
        # Test sonuçlarını analiz et
        correct_examples, wrong_examples = analyze_test_results(model, X_test, y_test, problem_type)
        print(f"\nAnalyzed {len(correct_examples)} correct and {len(wrong_examples)} wrong predictions")
    
    # Sonuçları görselleştir
    problem_names = {
        'A': 'İki Nokta Arası Mesafe',
        'B': 'En Yakın İki Nokta Mesafesi',
        'C': 'En Uzak İki Nokta Mesafesi',
        'D': 'Nokta Sayısı Tahmini',
        'E': 'Kare Sayısı Tahmini'
    }
    
    title = f'{problem_names[problem_type]} - {"Accuracy" if problem_type in ["D", "E"] else "MAE"} vs Training Size'
    plot_results(results, title, problem_type)
    
    # Eğitim geçmişlerini görselleştir
    plot_training_history(history_dict, problem_type, problem_names[problem_type])
    
    return results, history_dict

def plot_training_history(history_dict, problem_type, problem_name):
    """Eğitim ve validation metriklerini görselleştirir ve kaydeder"""
    for size, data in history_dict.items():
        history = data['history']
        hyperparams = data['hyperparams']
        
        plt.figure(figsize=(12, 5))
        
        if problem_type in ['D', 'E']:  # Sınıflandırma problemleri
            # Accuracy grafiği
            plt.subplot(1, 2, 1)
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'Accuracy (Size: {size})')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Loss grafiği
            plt.subplot(1, 2, 2)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title(f'Loss (Size: {size})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        else:  # Regresyon problemleri
            # MAE grafiği
            plt.subplot(1, 2, 1)
            plt.plot(history['mae'], label='Training MAE')
            plt.plot(history['val_mae'], label='Validation MAE')
            plt.title(f'MAE (Size: {size})')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            # Loss grafiği
            plt.subplot(1, 2, 2)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title(f'Loss (Size: {size})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
        # Hiperparametre bilgilerini ekle
        hyperparam_text = (
            f"Optimizer: {hyperparams['optimizer']}\n"
            f"Loss: {hyperparams['loss']}\n"
            f"Metrics: {', '.join(hyperparams['metrics'])}"
        )
        
        plt.figtext(0.5, -0.1, hyperparam_text, wrap=True, 
                   horizontalalignment='center', fontsize=10)
        
        plt.suptitle(f'{problem_name} - Training Size: {size}')
        plt.tight_layout()
        
        # Grafiği kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_plots/{problem_type}_size_{size}_{timestamp}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved plot to {filename}")

def plot_results(results_dict, title, problem_type):
    """Sonuçları görselleştirir ve kaydeder"""
    sizes = list(results_dict.keys())
    values = list(results_dict.values())
    
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, values, marker='o')
    plt.title(title)
    plt.xlabel('Training Set Size')
    
    if 'Accuracy' in title:
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
    else:
        plt.ylabel('MAE')
    
    plt.grid(True)
    
    # Grafiği kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_plots/{problem_type}_results_{timestamp}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved results plot to {filename}")

## 4. Ana Program

def main():
    """Tüm problemleri sırayla çalıştırır"""
    problems = {
        'A': create_model_A,
        'B': create_model_B,
        'C': create_model_C,
        'D': create_model_D,
        'E': create_model_E
    }
    
    all_results = {}
    
    for problem_code, model_creator in problems.items():
        print(f"\n{'='*50}")
        print(f"PROBLEM {problem_code}")
        print(f"{'='*50}")
        
        results, histories = train_and_evaluate_problem(problem_code, model_creator)
        all_results[problem_code] = results
    
    # Sonuçları yazdır
    print("\nFinal Results Summary:")
    for problem, results in all_results.items():
        print(f"\nProblem {problem}:")
        for size, perf in results.items():
            if problem in ['D', 'E']:
                print(f"  {size} samples: Accuracy = {perf:.4f}")
            else:
                print(f"  {size} samples: MAE = {perf:.4f}")

if __name__ == "__main__":
    main()