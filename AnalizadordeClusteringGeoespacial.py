import tkinter as tk
from tkinter import ttk, messagebox, font, filedialog
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.cluster.hierarchy import dendrogram
import threading
import os

class ClusteringApp:
    def __init__(self, root):
        self.root = root
        
        self.coords_df = None
        self.dist_matrix = None
        self.coords_filepath = tk.StringVar()
        self.matrix_filepaths = None
        self.cluster_results = None
        self._full_coords_path = None 

        self.setup_ui()


    def setup_ui(self):
        self.root.title("Analizador de Clustering Geoespacial")
        self.root.state('zoomed')
        self.root.configure(bg='#2c3e50')

        # Estilos
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background='#2c3e50')
        style.configure("TLabel", background='#2c3e50', foreground='white', font=('Helvetica', 12))
        style.configure("TRadiobutton", background='#2c3e50', foreground='white', font=('Helvetica', 11))
        style.configure("TCheckbutton", background='#2c3e50', foreground='white', font=('Helvetica', 11))
        style.map("TRadiobutton", background=[('active', '#34495e')])
        style.map("TCheckbutton", background=[('active', '#34495e')])
        style.configure("TButton", font=('Helvetica', 11, 'bold'), borderwidth=0)
        style.map("TButton",
                    background=[('active', '#1abc9c'), ('!disabled', '#16a085')],
                    foreground=[('!disabled', 'white')])
        style.configure("Title.TLabel", font=('Helvetica', 16, 'bold'))
        style.configure("Path.TLabel", font=('Helvetica', 9, 'italic'))

        # --- Panel de Control (Izquierda) ---
        control_frame = ttk.Frame(self.root, padding="20")
        control_frame.pack(side="left", fill="y", padx=10, pady=10)

        # --- SECCIÓN DE CARGA DE ARCHIVOS ---
        ttk.Label(control_frame, text="1. Cargar Archivos", style="Title.TLabel").pack(pady=(10, 15), anchor='w')

        ttk.Button(control_frame, text="Seleccionar Archivo de Coordenadas", command=self.select_coords_file).pack(fill='x', ipady=5)
        ttk.Label(control_frame, textvariable=self.coords_filepath, style="Path.TLabel").pack(anchor='w', pady=(2, 10))
        
        # BOTÓN PARA GENERAR LA MATRIZ DE DISTANCIA ---
        self.generate_matrix_button = ttk.Button(
            control_frame, 
            text="Generar y Guardar Matriz (Distancia Euclidiana)", 
            command=self.generate_matrix_file_thread,
            state="disabled" 
        )
        self.generate_matrix_button.pack(fill='x', ipady=5, pady=(0, 15))
        # -----------------------------------------------------------------

        ttk.Button(control_frame, text="Seleccionar Archivo(s) de Matriz", command=self.select_matrix_files).pack(fill='x', ipady=5)
        self.matrix_files_label = ttk.Label(control_frame, text="Ninguno seleccionado", style="Path.TLabel", wraplength=300)
        self.matrix_files_label.pack(anchor='w', pady=(2, 10))

        ttk.Button(control_frame, text="Cargar y Verificar Datos", command=self.load_data_thread).pack(fill='x', ipady=5, pady=(15, 0))
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=20)

        # --- SECCIÓN DE ANÁLISIS ---
        ttk.Label(control_frame, text="2. Configurar Análisis", style="Title.TLabel").pack(pady=(10, 15), anchor='w')
        
        self.clustering_type = tk.StringVar(value="particional")

        ttk.Label(control_frame, text="Tipo de agrupamiento:").pack(anchor='w', pady=(10, 5))
        ttk.Radiobutton(control_frame, text="Particional", variable=self.clustering_type, value="particional", command=self.update_options).pack(anchor='w', padx=20)
        ttk.Radiobutton(control_frame, text="Jerárquico", variable=self.clustering_type, value="jerarquico", command=self.update_options).pack(anchor='w', padx=20)

        # --- MODIFICACIÓN: Crear un frame para las opciones de K ---
        self.k_options_frame = ttk.Frame(control_frame)
        self.k_label = ttk.Label(self.k_options_frame, text="Número de grupos (K):")
        self.k_label.pack(anchor='w', pady=(10, 5))
        self.k_value = tk.IntVar(value=5)
        self.k_spinbox = ttk.Spinbox(self.k_options_frame, from_=2, to=50, textvariable=self.k_value, width=10, font=('Helvetica', 11))
        self.k_spinbox.pack(anchor='w', padx=20)
        
        self.options_label = ttk.Label(control_frame, text="Criterio para unir grupos:")
        self.method_var = tk.StringVar()
        self.method_menu = ttk.OptionMenu(control_frame, self.method_var, None)

        self.remove_outliers_var = tk.BooleanVar()
        self.remove_outliers_check = ttk.Checkbutton(control_frame, text="Eliminar Puntos Atípicos (Outliers)", variable=self.remove_outliers_var)

        self.run_button = ttk.Button(control_frame, text="Ejecutar Análisis", command=self.run_analysis_thread, state="disabled")
        
        action_button_frame = ttk.Frame(control_frame)
        self.view_clusters_button = ttk.Button(action_button_frame, text="Visualizar Grupos", command=self.show_clusters_window, state="disabled")
        self.view_clusters_button.pack(side="left", expand=True, fill='x', ipady=5, padx=(0, 5))
        exit_button = ttk.Button(action_button_frame, text="Salir del Sistema", command=self.root.destroy)
        exit_button.pack(side="right", expand=True, fill='x', ipady=5, padx=(5, 0))

        self.progress_bar = ttk.Progressbar(control_frame, orient='horizontal', mode='determinate')
        self.status_label = ttk.Label(control_frame, text="Listo. Por favor, cargue los archivos.", font=('Helvetica', 11, 'italic'))

        # --- Empaquetado final en el orden correcto ---
        self.k_options_frame.pack(fill='x', anchor='w')
        self.options_label.pack(anchor='w', pady=(10, 5))
        self.method_menu.pack(anchor='w', padx=20, fill='x')
        self.remove_outliers_check.pack(anchor='w', pady=(15, 0))
        self.run_button.pack(pady=20, fill='x', ipady=5)
        action_button_frame.pack(fill='x', pady=(0, 20))
        self.status_label.pack(pady=10, anchor='w')

        # --- Panel de Visualización y Explicación ---
        main_area = ttk.Frame(self.root)
        main_area.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(facecolor='#34495e')
        self.ax.set_facecolor('#34495e')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_color('none')
        self.ax.spines['right'].set_color('none')

        self.canvas = FigureCanvasTkAgg(self.fig, master=main_area)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.explanation_text = tk.Text(main_area, height=10, wrap="word", bg='#34495e', fg='white', font=('Helvetica', 11), relief='flat', padx=10, pady=10)
        self.explanation_text.pack(side="bottom", fill="x", pady=(10, 0))

        self.update_options()

    def select_coords_file(self):
        filepath = filedialog.askopenfilename(title="Seleccionar archivo de coordenadas", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filepath:
            self.coords_filepath.set(os.path.basename(filepath))
            self._full_coords_path = filepath
            # Habilitar el botón de generación de matriz si se seleccionó un archivo
            self.generate_matrix_button.config(state="normal")
        else:
            # Deshabilitar si no se seleccionó nada
            self.generate_matrix_button.config(state="disabled")
            self._full_coords_path = None


    def select_matrix_files(self):
        filepaths = filedialog.askopenfilenames(title="Seleccionar 1 o 2 archivos de matriz", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filepaths:
            self.matrix_filepaths = filepaths
            basenames = [os.path.basename(p) for p in filepaths]
            self.matrix_files_label.config(text=", ".join(basenames))

    def update_options(self):
        menu = self.method_menu["menu"]
        menu.delete(0, "end")
        
        self.options_label.config(text="Criterio para unir grupos:")
        options = {
            "Single": "single",
            "Average": "average",
            "Complete": "complete"
        }
        
        current_method = self.method_var.get()
        if not current_method: 
            self.method_var.set("average")
        
        def set_and_update(value):
            self.method_var.set(value)
            self.update_explanation()

        for display_name, value in options.items():
            menu.add_command(label=display_name, command=lambda v=value: set_and_update(v))

        # --- MODIFICACIÓN: Ocultar/mostrar el frame de K ---
        if self.clustering_type.get() == "jerarquico":
            self.k_options_frame.pack_forget()
            self.view_clusters_button.config(state="disabled")
        else: # particional
            self.k_options_frame.pack(fill='x', anchor='w', before=self.options_label)
            self.view_clusters_button.config(state="disabled")

        self.update_explanation()
        
        if hasattr(self, 'coords_df') and self.coords_df is not None:
            self.plot_data()


    def update_explanation(self, result_text=None):
        self.explanation_text.config(state="normal")
        self.explanation_text.delete(1.0, "end")
        
        c_type = self.clustering_type.get()
        method = self.method_var.get()
        
        linkage_desc = {
            "complete": "Criterio estricto. Une dos grupos solo si el miembro MÁS LEJANO de un grupo está cerca del MÁS LEJANO del otro. Crea grupos muy compactos y separados, como 'islas'.",
            "average": "Criterio balanceado. Calcula la distancia PROMEDIO entre TODOS los miembros de un grupo y TODOS los del otro. Es el más común y suele dar resultados muy intuitivos.",
            "single": "Criterio optimista. Para unir dos grupos, basta con que solo UN miembro de un grupo esté cerca de UNO del otro. A veces puede crear grupos largos, como una 'cadena'."
        }

        if c_type == "particional":
            info_text = (
                "AGRUPAMIENTO PARTICIONAL:\n\n"
                "El objetivo es dividir todos los puntos en un número K de grupos que tú defines. Es útil cuando ya sabes cuántos grupos esperas encontrar.\n\n"
                f"Criterio de unión: {linkage_desc.get(method, '')}"
            )
        else: # jerarquico
            info_text = (
                f"AGRUPAMIENTO JERÁRQUICO:\n\n"
                "Este método no necesita que le digas cuántos grupos buscar. En su lugar, construye un 'árbol familiar' (dendrograma) que muestra cómo se van uniendo los puntos y grupos desde lo más pequeño hasta que solo queda un gran grupo.\n\n"
                f"Criterio de unión: {linkage_desc.get(method, '')}"
            )

        if result_text:
            info_text += f"\n\n--- RESULTADO ---\n{result_text}"

        self.explanation_text.insert(1.0, info_text)
        self.explanation_text.config(state="disabled")

    def load_data_thread(self):
        if not hasattr(self, '_full_coords_path') or not self.matrix_filepaths:
            messagebox.showwarning("Archivos Faltantes", "Por favor, seleccione los archivos de coordenadas y matriz.")
            return
        
        self.status_label.config(text="Cargando datos...")
        self.run_button.config(state="disabled")
        self.view_clusters_button.config(state="disabled")

        self.progress_bar.pack(fill='x', pady=5, before=self.status_label)
        
        thread = threading.Thread(target=self.load_data, daemon=True)
        thread.start()

    def on_data_load_error(self, error_message):
        self.hide_progress_bar()
        self.progress_bar['value'] = 0
        self.status_label.config(text=error_message)

    def load_data(self):
        try:
            total_lines = 0
            for filepath in self.matrix_filepaths:
                # Contar líneas de forma robusta
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    total_lines += sum(1 for line in f)

            self.root.after(0, lambda: self.progress_bar.config(maximum=total_lines, value=0))
            
            self.processed_lines = 0
            def update_progress():
                self.processed_lines += 1
                self.progress_bar['value'] = self.processed_lines

            def load_dist_part_manually(filepath, has_header_and_index=False, update_callback=None):
                data = []
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    if has_header_and_index:
                        next(f) 
                    for line in f:
                        if update_callback:
                            self.root.after(0, update_callback) 
                        
                        start_index = 1 if has_header_and_index else 0
                        parts = line.strip().split('\t')[start_index:]
                        row = [float(p) for p in parts if p]
                        data.append(row)
                
                if not data: return np.array([])
                min_len = min(len(r) for r in data)
                truncated_data = [r[:min_len] for r in data]
                return np.array(truncated_data, dtype=np.float32)

            coords_path = self._full_coords_path
            
            # Cargar y limpiar coordenadas
            self.coords_df = pd.read_csv(coords_path, sep='\t')
            self.coords_df['Longitude'] = pd.to_numeric(self.coords_df['Longitude'], errors='coerce')
            self.coords_df['Latitude'] = pd.to_numeric(self.coords_df['Latitude'], errors='coerce')
            self.coords_df.dropna(subset=['Longitude', 'Latitude'], inplace=True)
            
            self.root.after(0, lambda: self.status_label.config(text="Cargando matriz (método robusto)..."))

            # Cargar matriz
            num_matrix_files = len(self.matrix_filepaths)
            if num_matrix_files == 1:
                self.dist_matrix = load_dist_part_manually(self.matrix_filepaths[0], has_header_and_index=False, update_callback=update_progress)
            elif num_matrix_files == 2:
                paths = sorted(self.matrix_filepaths)
                dist_p1_np = load_dist_part_manually(paths[0], has_header_and_index=True, update_callback=update_progress)
                dist_p2_np = load_dist_part_manually(paths[1], has_header_and_index=False, update_callback=update_progress)
                
                if dist_p1_np.shape[0] != dist_p2_np.shape[0]:
                    raise ValueError("Inconsistencia en el número de filas entre las partes de la matriz.")
                self.dist_matrix = np.hstack((dist_p1_np, dist_p2_np))
            else:
                raise ValueError("Por favor, seleccione 1 o 2 archivos para la matriz de disimilitud.")

            self.root.after(0, self.on_data_loaded)

        except Exception as e:
            messagebox.showerror("Error al Cargar Datos", f"Ocurrió un error inesperado:\n{e}")
            self.root.after(0, self.on_data_load_error, "Error al cargar los datos.")

    def on_data_loaded(self):
        self.progress_bar['value'] = self.progress_bar['maximum']
        self.status_label.config(text="Datos cargados. Listo para analizar.")
        self.run_button.config(state="normal")
        
        self.root.after(5000, self.hide_progress_bar)
        
        self.plot_data()
        self.update_explanation()

    def generate_matrix_file_thread(self):
        """Inicia la generación de la matriz de distancias en un hilo separado."""
        if self._full_coords_path is None:
            messagebox.showwarning("Archivo Faltante", "Por favor, seleccione un archivo de coordenadas primero.")
            return

        self.status_label.config(text="Calculando y generando matriz...")
        self.generate_matrix_button.config(state="disabled")
        self.run_button.config(state="disabled")

        self.progress_bar.pack(fill='x', pady=5, before=self.status_label)
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()

        thread = threading.Thread(target=self.generate_matrix_file, daemon=True)
        thread.start()

    def calculate_distance_matrix(self, coords_df):
        """Calcula la matriz de distancias euclidianas a partir de las coordenadas."""
        
        # 1. Preparar datos
        if coords_df is None or coords_df.empty:
            raise ValueError("No hay datos de coordenadas válidos para calcular la matriz.")
            
        X = coords_df[['Longitude', 'Latitude']].values
        
        # 2. Calcular la matriz de distancias (Euclidiana)
        dist_matrix = euclidean_distances(X)
        
        return dist_matrix

    def generate_matrix_file(self):
        """Calcula la matriz de distancias y la guarda en un archivo .txt."""
        
        try:
            # 1. Cargar coordenadas (usando la ruta guardada)
            self.root.after(0, lambda: self.status_label.config(text="Cargando coordenadas..."))
            coords_path = self._full_coords_path
            coords_df = pd.read_csv(coords_path, sep='\t')
            coords_df['Longitude'] = pd.to_numeric(coords_df['Longitude'], errors='coerce')
            coords_df['Latitude'] = pd.to_numeric(coords_df['Latitude'], errors='coerce')
            coords_df.dropna(subset=['Longitude', 'Latitude'], inplace=True)
            
            if coords_df.empty:
                raise ValueError("El archivo de coordenadas no contiene datos válidos de Longitud/Latitud.")
                
            # 2. Calcular la matriz
            self.root.after(0, lambda: self.status_label.config(text="Calculando matriz de distancia..."))
            dist_matrix_np = self.calculate_distance_matrix(coords_df)
            
            # 3. Preguntar la ruta de guardado
            save_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Guardar Matriz de Distancia Calculada"
            )
            
            if not save_path:
                self.root.after(0, lambda: self.status_label.config(text="Generación cancelada por el usuario."))
                return

            # 4. Guardar la matriz
            self.root.after(0, lambda: self.status_label.config(text=f"Guardando archivo en: {os.path.basename(save_path)}"))
            
            # Usamos np.savetxt para exportar la matriz con tabulaciones
            np.savetxt(save_path, dist_matrix_np, delimiter='\t', fmt='%.10f')
            
            # 5. Éxito
            self.root.after(0, lambda: self.progress_bar.stop())
            self.root.after(0, self.hide_progress_bar)
            self.root.after(0, lambda: messagebox.showinfo("Éxito", f"Matriz de distancia generada y guardada en:\n{save_path}"))
            self.root.after(0, lambda: self.status_label.config(text="Matriz de distancia generada. Listo."))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error al Generar Matriz", f"Ocurrió un error inesperado:\n{e}"))
            self.root.after(0, self.on_data_load_error, "Error al generar la matriz.")

        finally:
            self.root.after(0, lambda: self.progress_bar.stop())
            self.root.after(0, self.hide_progress_bar)
            self.root.after(0, lambda: self.generate_matrix_button.config(state="normal"))
            self.root.after(0, lambda: self.run_button.config(state="normal"))

    # -----------------------------------------------------------------
    
    def run_analysis_thread(self):
        self.run_button.config(state="disabled")
        self.view_clusters_button.config(state="disabled")
        self.status_label.config(text=f"Ejecutando {self.clustering_type.get()}...")
        
        thread = threading.Thread(target=self.perform_clustering, daemon=True)
        thread.start()

    def perform_clustering(self):
        c_type = self.clustering_type.get()
        method = self.method_var.get()
        
        try:
            current_coords = self.coords_df.copy()
            current_matrix = self.dist_matrix.copy()

            n_points_coords = len(current_coords)
            n_points_matrix = current_matrix.shape[0]
            if n_points_coords != n_points_matrix:
                min_points = min(n_points_coords, n_points_matrix)
                self.root.after(0, lambda: messagebox.showwarning("Ajuste de Datos", 
                    f"Se ajustarán los datos al tamaño más pequeño ({min_points} puntos)."))
                current_coords = current_coords.iloc[:min_points]
                current_matrix = current_matrix[:min_points, :min_points]

            analysis_title_info = ""
            if self.remove_outliers_var.get():
                mean_distances = np.mean(current_matrix, axis=1)
                q1 = np.percentile(mean_distances, 25)
                q3 = np.percentile(mean_distances, 75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                
                non_outlier_indices = np.where(mean_distances <= upper_bound)[0]
                num_outliers = len(current_coords) - len(non_outlier_indices)
                
                if num_outliers > 0:
                    analysis_title_info = f"\n(sin {num_outliers} puntos atípicos)"
                    self.root.after(0, lambda: messagebox.showinfo("Filtro de Atípicos", f"Se eliminaron {num_outliers} puntos atípicos."))
                    current_coords = current_coords.iloc[non_outlier_indices]
                    current_matrix = current_matrix[np.ix_(non_outlier_indices, non_outlier_indices)]
                else:
                    analysis_title_info = "\n(sin atípicos detectados)"
                    self.root.after(0, lambda: messagebox.showinfo("Filtro de Atípicos", "No se encontraron puntos atípicos con el método IQR."))
            else:
                analysis_title_info = "\n(con todos los puntos)"


            if c_type == 'particional':
                k = self.k_value.get()
                model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage=method)
                labels = model.fit_predict(current_matrix)
                
                medoid_indices = self.calculate_medoids(labels, current_matrix)
                
                self.cluster_results = current_coords.copy()
                self.cluster_results['Grupo'] = labels
                result_text = f"Se crearon {k} grupos. Los puntos más céntricos (medoides) se muestran como estrellas."
                self.root.after(0, self.on_analysis_complete, {'labels': labels, 'coords': current_coords, 'medoids': medoid_indices, 'result_text': result_text})

            else: # jerarquico
                model = AgglomerativeClustering(n_clusters=None, distance_threshold=0, metric='precomputed', linkage=method)
                model.fit(current_matrix)
                
                linkage_matrix = self.create_linkage_matrix(model)
                result_text = "Se ha generado el dendrograma jerárquico."
                self.root.after(0, self.on_analysis_complete, {'linkage_matrix': linkage_matrix, 'result_text': result_text, 'title_info': analysis_title_info})

        except Exception as e:
            messagebox.showerror("Error en el Análisis", f"Ocurrió un error:\n{e}")
            self.root.after(0, self.reset_ui_after_error)
    
    def hide_progress_bar(self):
        if self.progress_bar.winfo_ismapped():
            self.progress_bar.pack_forget()

    def calculate_medoids(self, labels, matrix):
        medoid_indices = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) == 1:
                medoid_indices.append(cluster_indices[0])
                continue
            if len(cluster_indices) == 0:
                continue
            sub_matrix = matrix[np.ix_(cluster_indices, cluster_indices)]
            sum_distances = np.sum(sub_matrix, axis=1)
            min_dist_idx_in_cluster = np.argmin(sum_distances)
            medoid_original_idx = cluster_indices[min_dist_idx_in_cluster]
            medoid_indices.append(medoid_original_idx)
        return medoid_indices

    def create_linkage_matrix(self, model):
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
        return linkage_matrix

    def show_clusters_window(self):
        if self.cluster_results is None:
            messagebox.showinfo("Sin Resultados", "Primero debe ejecutar un análisis particional.")
            return

        top = tk.Toplevel(self.root)
        top.title("Contenido de los Grupos")
        top.geometry("800x600")
        top.configure(bg='#34495e')

        notebook = ttk.Notebook(top)
        notebook.pack(pady=10, padx=10, expand=True, fill="both")

        unique_clusters = sorted(self.cluster_results['Grupo'].unique())
        for cluster_id in unique_clusters:
            frame = ttk.Frame(notebook, padding="10")
            notebook.add(frame, text=f'Grupo {cluster_id}')
            cluster_data = self.cluster_results[self.cluster_results['Grupo'] == cluster_id].drop(columns=['Grupo'])
            tree_frame = ttk.Frame(frame)
            tree_frame.pack(expand=True, fill='both')
            cols = list(cluster_data.columns)
            tree = ttk.Treeview(tree_frame, columns=cols, show='headings')
            for col in cols:
                tree.heading(col, text=col)
                tree.column(col, width=120, anchor='center')
            for index, row in cluster_data.iterrows():
                tree.insert("", "end", values=list(row))
            vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            vsb.pack(side='right', fill='y')
            hsb.pack(side='bottom', fill='x')
            tree.pack(expand=True, fill='both')

    def on_analysis_complete(self, results):
        result_text = results['result_text']
        if 'labels' in results:
            medoids = results.get('medoids')
            self.plot_data(results['labels'], results['coords'], medoids)
            if self.clustering_type.get() == "particional":
                self.view_clusters_button.config(state="normal")
        elif 'linkage_matrix' in results:
            title_info = results.get('title_info', '')
            self.plot_dendrogram(results['linkage_matrix'], title_info)

        self.status_label.config(text="Análisis completado.")
        self.run_button.config(state="normal")
        self.update_explanation(result_text)

    def reset_ui_after_error(self):
        self.hide_progress_bar()
        self.progress_bar['value'] = 0
        self.status_label.config(text="Error. Intente de nuevo.")
        self.run_button.config(state="normal")
        self.view_clusters_button.config(state="disabled")

    def plot_dendrogram(self, linkage_matrix, title_info=""):
        self.ax.clear()
        
        dendrogram(
            linkage_matrix,
            ax=self.ax,
            truncate_mode='lastp',
            p=12,
            show_leaf_counts=True,
            leaf_rotation=90.,
            leaf_font_size=8.,
            show_contracted=True,
            link_color_func=lambda k: '#1abc9c'
        )
        
        self.ax.set_yscale('log')
        
        full_title = "Dendrograma del Clustering Jerárquico" + title_info
        self.ax.set_title(full_title, color='white', fontsize=14)
        self.ax.set_xlabel("Índice del Cluster (o puntos en el cluster)", color='white')
        self.ax.set_ylabel("Distancia (escala logarítmica)", color='white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_data(self, labels=None, coords=None, medoids=None):
        self.ax.clear()
        
        plot_coords = coords if coords is not None else self.coords_df
        
        if plot_coords is None or plot_coords.empty:
            self.canvas.draw()
            return
            
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        else:
            colors = '#1abc9c'

        self.ax.scatter(
            plot_coords['Longitude'],
            plot_coords['Latitude'],
            c=labels if labels is not None else colors,
            cmap='viridis' if labels is not None else None,
            s=5,
            alpha=0.8
        )

        if medoids is not None and not plot_coords.empty:
            medoid_coords = plot_coords.iloc[medoids]
            self.ax.scatter(
                medoid_coords['Longitude'],
                medoid_coords['Latitude'],
                s=150,
                c='yellow',
                marker='*',
                edgecolors='black',
                zorder=10,
                label='Medoides'
            )

        self.ax.set_title("Distribución Espacial de Grupos en México", color='white', fontsize=14)
        self.ax.set_xlabel("Longitud", color='white')
        self.ax.set_ylabel("Latitud", color='white')
        self.ax.grid(True, linestyle='--', alpha=0.3)

        if labels is not None:
            unique_labels = np.unique(labels)
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Grupo {i}', 
                                    markerfacecolor=colors[i], markersize=8) for i in unique_labels]
            
            if medoids is not None:
                 handles.append(plt.Line2D([0], [0], marker='*', color='yellow', label='Medoide',
                                         markeredgecolor='black', markersize=12, linestyle='None'))

            self.ax.legend(handles=handles, title="Clusters", labelcolor='white', facecolor='#2c3e50', edgecolor='white')

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringApp(root)
    root.mainloop()
