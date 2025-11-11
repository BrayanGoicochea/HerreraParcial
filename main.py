import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

class FakeNewsDetectorApp:
    def __init__(self, root, model_pipeline, df_train, df_test, df_valid=None):
        self.root = root
        self.model = model_pipeline
        self.df_train = df_train
        self.df_test = df_test
        self.df_valid = df_valid
        
        self.setup_ui()
        self.setup_plots()
        
    def setup_ui(self):
        self.root.title("🔍 Detector de Noticias Falsas - LIAR Dataset")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Crear notebook (pestañas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Pestaña 1: Detector
        self.tab_detector = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_detector, text="🧠 Detector en Tiempo Real")
        
        # Pestaña 2: Análisis de Datos
        self.tab_analysis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analysis, text="📊 Análisis de Datos")
        
        # Pestaña 3: Pruebas con Ejemplos
        self.tab_examples = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_examples, text="📝 Ejemplos Predefinidos")
        
        self.setup_detector_tab()
        self.setup_analysis_tab()
        self.setup_examples_tab()
    
    def setup_detector_tab(self):
        # Título
        title_label = tk.Label(self.tab_detector, 
                              text="Detector de Noticias Falsas", 
                              font=('Arial', 16, 'bold'),
                              bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Área de texto para ingresar noticia
        input_frame = tk.Frame(self.tab_detector, bg='#f0f0f0')
        input_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(input_frame, text="Ingresa la noticia o declaración:", 
                font=('Arial', 12), bg='#f0f0f0').pack(anchor='w')
        
        self.text_input = scrolledtext.ScrolledText(input_frame, 
                                                   height=8, 
                                                   font=('Arial', 11),
                                                   wrap=tk.WORD)
        self.text_input.pack(fill='x', pady=5)
        
        # Botones de acción
        button_frame = tk.Frame(self.tab_detector, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="🔍 Analizar Noticia", 
                 command=self.analyze_text,
                 font=('Arial', 12), 
                 bg='#4CAF50', 
                 fg='white',
                 padx=20).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="🧹 Limpiar", 
                 command=self.clear_text,
                 font=('Arial', 12),
                 bg='#ff9800',
                 fg='white',
                 padx=20).pack(side='left', padx=5)
        
        # Resultados
        self.result_frame = tk.LabelFrame(self.tab_detector, 
                                         text="Resultado del Análisis", 
                                         font=('Arial', 12, 'bold'),
                                         bg='#f0f0f0')
        self.result_frame.pack(fill='x', padx=20, pady=10)
        
        self.result_label = tk.Label(self.result_frame, 
                                    text="Esperando análisis...", 
                                    font=('Arial', 14),
                                    bg='#f0f0f0')
        self.result_label.pack(pady=10)
        
        # Barra de confianza
        self.confidence_frame = tk.Frame(self.result_frame, bg='#f0f0f0')
        self.confidence_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Label(self.confidence_frame, text="Confianza:", 
                font=('Arial', 10), bg='#f0f0f0').pack(side='left')
        
        self.confidence_bar = ttk.Progressbar(self.confidence_frame, 
                                             length=200, 
                                             mode='determinate')
        self.confidence_bar.pack(side='left', padx=5)
        
        self.confidence_value = tk.Label(self.confidence_frame, 
                                        text="0%", 
                                        font=('Arial', 10),
                                        bg='#f0f0f0')
        self.confidence_value.pack(side='left')
    
    def setup_analysis_tab(self):
        # Estadísticas del dataset
        stats_frame = tk.LabelFrame(self.tab_analysis, 
                                   text="Estadísticas del Dataset", 
                                   font=('Arial', 12, 'bold'))
        stats_frame.pack(fill='x', padx=20, pady=10)
        
        # Crear frame para estadísticas
        stats_grid = tk.Frame(stats_frame)
        stats_grid.pack(padx=10, pady=10)
        
        # Calcular estadísticas
        train_size = len(self.df_train)
        test_size = len(self.df_test)
        
        real_count = len(self.df_train[self.df_train['binary_label'] == 1])
        fake_count = len(self.df_train[self.df_train['binary_label'] == 0])
        
        stats_data = [
            ("📊 Total de entrenamiento:", f"{train_size:,} declaraciones"),
            ("🧪 Total de prueba:", f"{test_size:,} declaraciones"),
            ("✅ Declaraciones REALES:", f"{real_count:,} ({real_count/train_size*100:.1f}%)"),
            ("❌ Declaraciones FALSAS:", f"{fake_count:,} ({fake_count/train_size*100:.1f}%)")
        ]
        
        # Agregar validación si existe
        if self.df_valid is not None:
            valid_size = len(self.df_valid)
            stats_data.insert(2, ("✅ Total de validación:", f"{valid_size:,} declaraciones"))
        
        for i, (label, value) in enumerate(stats_data):
            tk.Label(stats_grid, text=label, font=('Arial', 10, 'bold')).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            tk.Label(stats_grid, text=value, font=('Arial', 10)).grid(row=i, column=1, sticky='w', padx=5, pady=2)
        
        # Gráficos
        charts_frame = tk.Frame(self.tab_analysis)
        charts_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Figura para matplotlib
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def setup_examples_tab(self):
        # Ejemplos predefinidos
        examples_frame = tk.LabelFrame(self.tab_examples, 
                                      text="Ejemplos Predefinidos para Probar", 
                                      font=('Arial', 12, 'bold'))
        examples_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Lista de ejemplos (reales y falsos)
        self.examples = [
            ("REAL: The economy created 200,000 new jobs last month", 1),
            ("FALSO: The moon landing was filmed in a Hollywood studio", 0),
            ("REAL: Climate change is affecting global weather patterns", 1),
            ("FALSO: Vaccines contain microchips for tracking", 0),
            ("REAL: Regular exercise improves cardiovascular health", 1),
            ("FALSO: The Earth is flat and stationary", 0),
            ("REAL: Drinking water is essential for human health", 1),
            ("FALSO: COVID-19 was created as a biological weapon", 0)
        ]
        
        # Crear botones para cada ejemplo
        for i, (example_text, expected) in enumerate(self.examples):
            example_frame = tk.Frame(examples_frame, relief='raised', bd=1)
            example_frame.pack(fill='x', padx=5, pady=2)
            
            tk.Button(example_frame, 
                     text="Probar este ejemplo", 
                     command=lambda txt=example_text: self.load_example(txt),
                     width=15).pack(side='left', padx=5, pady=5)
            
            color = 'green' if expected == 1 else 'red'
            emoji = "✅" if expected == 1 else "❌"
            
            tk.Label(example_frame, 
                    text=f"{emoji} {example_text}", 
                    font=('Arial', 10),
                    wraplength=800,
                    justify='left').pack(side='left', padx=5, pady=5, fill='x', expand=True)
    
    def setup_plots(self):
        """Configurar los gráficos de análisis"""
        # Gráfico 1: Distribución de etiquetas
        labels_dist = self.df_train['label'].value_counts()
        self.ax1.clear()
        labels_dist.plot(kind='bar', ax=self.ax1, color='skyblue')
        self.ax1.set_title('Distribución de Etiquetas Originales')
        self.ax1.set_xlabel('Etiquetas')
        self.ax1.set_ylabel('Frecuencia')
        self.ax1.tick_params(axis='x', rotation=45)
        
        # Gráfico 2: Distribución binaria
        binary_dist = self.df_train['binary_label'].value_counts()
        self.ax2.clear()
        binary_dist.plot(kind='bar', ax=self.ax2, color=['red', 'green'])
        self.ax2.set_title('Distribución Binaria (REAL vs FAKE)')
        self.ax2.set_xlabel('Clasificación')
        self.ax2.set_ylabel('Frecuencia')
        self.ax2.set_xticklabels(['FAKE', 'REAL'], rotation=0)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def analyze_text(self):
        """Analizar el texto ingresado por el usuario"""
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Advertencia", "Por favor, ingresa una noticia o declaración para analizar.")
            return
        
        try:
            # Realizar predicción
            prediction = self.model.predict([text])[0]
            probability = self.model.predict_proba([text])[0]
            
            # Obtener la probabilidad de la clase predicha
            confidence = probability[prediction] * 100
            
            # Actualizar interfaz
            result_text = "✅ NOTICIA REAL" if prediction == 1 else "❌ NOTICIA FALSA"
            color = "green" if prediction == 1 else "red"
            
            self.result_label.config(text=result_text, fg=color)
            self.confidence_bar['value'] = confidence
            self.confidence_value.config(text=f"{confidence:.1f}%")
            
            # Mostrar análisis detallado
            self.show_detailed_analysis(text, prediction, probability, confidence)
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al analizar el texto: {str(e)}")
    
    def show_detailed_analysis(self, text, prediction, probability, confidence):
        """Mostrar análisis detallado en una nueva ventana"""
        detail_window = tk.Toplevel(self.root)
        detail_window.title("Análisis Detallado")
        detail_window.geometry("600x400")
        
        # Título
        tk.Label(detail_window, 
                text="📋 Análisis Detallado", 
                font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Texto analizado
        text_frame = tk.LabelFrame(detail_window, text="Texto Analizado")
        text_frame.pack(fill='x', padx=20, pady=5)
        
        text_display = scrolledtext.ScrolledText(text_frame, height=4, wrap=tk.WORD)
        text_display.insert(tk.END, text)
        text_display.config(state='disabled')
        text_display.pack(fill='x', padx=10, pady=10)
        
        # Resultados
        result_frame = tk.LabelFrame(detail_window, text="Resultados")
        result_frame.pack(fill='x', padx=20, pady=5)
        
        result_text = f"""
Clasificación: {'✅ NOTICIA REAL' if prediction == 1 else '❌ NOTICIA FALSA'}
Confianza: {confidence:.2f}%

Probabilidades:
• Probabilidad de ser REAL: {probability[1]*100:.2f}%
• Probabilidad de ser FALSO: {probability[0]*100:.2f}%

Longitud del texto: {len(text)} caracteres
Palabras: {len(text.split())}
        """
        
        result_display = tk.Text(result_frame, height=8, wrap=tk.WORD)
        result_display.insert(tk.END, result_text)
        result_display.config(state='disabled')
        result_display.pack(fill='x', padx=10, pady=10)
        
        # Botón para cerrar
        tk.Button(detail_window, 
                 text="Cerrar", 
                 command=detail_window.destroy,
                 bg='#2196F3',
                 fg='white',
                 font=('Arial', 12)).pack(pady=10)
    
    def clear_text(self):
        """Limpiar el área de texto"""
        self.text_input.delete("1.0", tk.END)
        self.result_label.config(text="Esperando análisis...", fg='black')
        self.confidence_bar['value'] = 0
        self.confidence_value.config(text="0%")
    
    def load_example(self, example_text):
        """Cargar un ejemplo predefinido en el detector"""
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", example_text)

# --- CÓDIGO COMPLETO CORREGIDO ---
# Aquí integramos tu código original con la interfaz gráfica:

def main():
    # Tu código original de carga y entrenamiento del modelo
    import pandas as pd
    import os
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report

    # --- PARTE 1: ORGANIZACIÓN DE ARCHIVOS Y CARGA DE DATOS ---
    DATASET_DIR = 'dataset'
    TRAIN_FILE = os.path.join(DATASET_DIR, 'train.tsv')
    TEST_FILE = os.path.join(DATASET_DIR, 'test.tsv')
    VALID_FILE = os.path.join(DATASET_DIR, 'valid.tsv')

    COLUMN_NAMES = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
        'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'
    ]

    def load_data(file_path):
        df = pd.read_csv(file_path, sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')
        return df

    # Cargar los datasets
    try:
        df_train = load_data(TRAIN_FILE)
        df_test = load_data(TEST_FILE)
        
        # Intentar cargar validación si existe, sino será None
        try:
            df_valid = load_data(VALID_FILE)
            print(f"✅ Archivos cargados: Entrenamiento ({len(df_train)}), Prueba ({len(df_test)}), Validación ({len(df_valid)})")
        except FileNotFoundError:
            df_valid = None
            print(f"✅ Archivos cargados: Entrenamiento ({len(df_train)}), Prueba ({len(df_test)})")
            print("ℹ️  Archivo de validación no encontrado, continuando sin él...")
            
    except FileNotFoundError as e:
        print(f"❌ ERROR: No se pudieron cargar los archivos: {e}")
        return

    # --- PARTE 2: PREPROCESAMIENTO Y CLASIFICACIÓN BINARIA ---
    def map_label_to_binary(label):
        if label in ['true', 'mostly-true', 'half-true']:
            return 1  # Real
        elif label in ['barely-true', 'false', 'pants-fire']:
            return 0  # Falso
        return -1

    # Aplicar el mapeo
    df_train['binary_label'] = df_train['label'].apply(map_label_to_binary)
    df_test['binary_label'] = df_test['label'].apply(map_label_to_binary)
    
    if df_valid is not None:
        df_valid['binary_label'] = df_valid['label'].apply(map_label_to_binary)

    # Definir características y objetivo
    X_train = df_train['statement']
    y_train = df_train['binary_label']
    X_test = df_test['statement']
    y_test = df_test['binary_label']

    # Crear el Pipeline del modelo
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])

    print("\n⚙️ Empezando el entrenamiento del modelo...")

    # Entrenar el modelo
    model_pipeline.fit(X_train, y_train)

    print("✅ Entrenamiento completado.")

    # Evaluar el modelo
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n📊 Precisión del modelo: {accuracy:.4f}")
    
    # Iniciar la interfaz gráfica
    print("🚀 Iniciando interfaz gráfica...")
    root = tk.Tk()
    app = FakeNewsDetectorApp(root, model_pipeline, df_train, df_test, df_valid)
    root.mainloop()

if __name__ == "__main__":
    main()