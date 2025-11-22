import tkinter as tk
from tkinter import ttk, messagebox, font, filedialog
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.cluster.hierarchy import dendrogram
import threading
import os
from PIL import Image, ImageTk

class ClusteringApp:
    def __init__(self, root):
        self.root = root

        self.coords_df = None
        self.variables_df = None
        self.var_dict = {} # Diccionario para mapear códigos a nombres
        self.selected_vars = []
        self.dist_matrix = None
        self.coords_filepath = tk.StringVar()
        self.matrix_filepaths = None
        self.cluster_results = None
        self._full_coords_path = None
        
        self.GRAPH_DIR = "Imagenes_graficas"
        self.img_tk_left = None
        self.img_tk_right = None 
        
        self.COMPARE_IMG_WIDTH = 500
        self.COMPARE_IMG_HEIGHT = 500

        self.main_app_frame = ttk.Frame(self.root)
        self.comparison_frame = ttk.Frame(self.root)

        self.setup_ui()
        self.show_main_app()

    def setup_ui(self):
        self.root.title("Analizador Geoespacial y Semántico - Proyecto Final")
        self.root.state('zoomed')
        self.root.configure(bg='#2c3e50')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background='#2c3e50')
        style.configure("TLabel", background='#2c3e50', foreground='white', font=('Helvetica', 12))
        style.configure("TRadiobutton", background='#2c3e50', foreground='white', font=('Helvetica', 11))
        style.configure("TCheckbutton", background='#2c3e50', foreground='white', font=('Helvetica', 11))
        style.map("TRadiobutton", background=[('active', '#34495e')])
        style.map("TCheckbutton", background=[('active', '#34495e')])
        
        style.configure("TButton", 
                            font=('Helvetica', 10, 'bold'), 
                            borderwidth=0,
                            background='#16a085',  
                            foreground='white'   
        )
        
        style.map("TButton",
                      background=[
                          ('active', '#1abc9c'),     
                          ('hover', '#1abc9c'),      
                          ('disabled', '#5f6a70')    
                      ],
                      foreground=[
                          ('disabled', '#aab0b3')    
                      ]
        )
        
        style.configure("Zoom.TButton", 
                            font=('Helvetica', 10, 'bold'),
                            padding=(5, 2)) 

        
        style.configure("Title.TLabel", font=('Helvetica', 16, 'bold'))
        style.configure("Path.TLabel", font=('Helvetica', 9, 'italic'))
        style.configure("Header.TLabel", font=('Helvetica', 18, 'bold'), foreground='white', background='#2c3e50')
        style.configure("Compare.TFrame", background='#2c3e50')
        style.configure("CompareInner.TFrame", background='#34495e', relief='sunken', borderwidth=2)
        style.configure("Compare.TLabel", background='#34495e', foreground='white', font=('Helvetica', 14, 'bold'))
                
        style.configure("Turquoise.Horizontal.TProgressbar",
                            troughcolor='#34495e',    
                            background='#1abc9c',     
                            borderwidth=0,
                            relief='flat')
        
        style.configure("TPanedWindow", background='#2c3e50')

        self.setup_main_app(self.main_app_frame)
        self.setup_comparison_ui(self.comparison_frame)

    def show_main_app(self):
        self.comparison_frame.pack_forget()
        self.main_app_frame.pack(fill="both", expand=True)

    def show_comparison(self):
        self.main_app_frame.pack_forget()
        self.comparison_frame.pack(fill="both", expand=True)
        self.update_graph_lists()

    def setup_comparison_ui(self, parent_frame):
        parent_frame.config(style="Compare.TFrame", padding=20)
        
        header_frame = ttk.Frame(parent_frame, style="Compare.TFrame")
        header_frame.pack(fill='x', pady=(0, 20)) 
        
        ttk.Label(header_frame, text="Comparador de Gráficas Guardadas", style="Header.TLabel").pack(side='left', anchor='w', expand=True)
        
        update_btn = ttk.Button(header_frame, text="Actualizar Lista", command=self.update_graph_lists)
        update_btn.pack(side='right', padx=10, ipady=5)
        
        back_btn = ttk.Button(header_frame, text="Regresar al Menú", command=self.show_main_app)
        back_btn.pack(side='right', ipady=5)

        content_frame = ttk.Frame(parent_frame, style="Compare.TFrame")
        content_frame.pack(fill='both', expand=True)
        
        left_col = ttk.Frame(content_frame, style="CompareInner.TFrame", padding=15)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ttk.Label(left_col, text="Seleccionar Gráfica Izquierda:", style="Compare.TLabel").pack(anchor='w', pady=(0, 10), fill='x')
        
        list_frame_left = ttk.Frame(left_col) 
        list_frame_left.pack(fill='x', pady=(0, 15))
        
        sb_left = ttk.Scrollbar(list_frame_left, orient='vertical')
        self.listbox_left = tk.Listbox(list_frame_left, yscrollcommand=sb_left.set, height=6, exportselection=False, 
                                        background='#ecf0f1', foreground='#2c3e50', font=('Helvetica', 10), relief='flat', borderwidth=0)
        sb_left.config(command=self.listbox_left.yview)
        
        sb_left.pack(side='right', fill='y')
        self.listbox_left.pack(side='left', fill='x', expand=True)
        self.listbox_left.bind('<<ListboxSelect>>', self.on_select_left)
        
        zoom_frame_left = ttk.Frame(left_col, style="CompareInner.TFrame")
        zoom_frame_left.pack(fill='x', pady=(0, 5))
        
        ttk.Button(zoom_frame_left, text="+", style="Zoom.TButton", command=lambda: self.zoom_in("left")).pack(side='left', padx=2)
        ttk.Button(zoom_frame_left, text="-", style="Zoom.TButton", command=lambda: self.zoom_out("left")).pack(side='left', padx=2)
        ttk.Button(zoom_frame_left, text="Reset", style="Zoom.TButton", command=lambda: self.reset_zoom_pan("left")).pack(side='left', padx=2)

        
        self.image_label_left = tk.Label(left_col, 
                                         text="Imagen Izquierda", 
                                         background='#4a6076',  
                                         foreground='#bdc3c7',  
                                         font=('Helvetica', 16, 'italic'),
                                         anchor='center')
        self.image_label_left.pack(fill='both', expand=True, pady=(0, 0)) 
        self.image_label_left.bind("<Button-1>", lambda event: self.on_button_press(event, "left"))
        self.image_label_left.bind("<B1-Motion>", lambda event: self.on_mouse_drag(event, "left"))


        right_col = ttk.Frame(content_frame, style="CompareInner.TFrame", padding=15)
        right_col.pack(side="right", fill="both", expand=True, padx=(10, 0))

        ttk.Label(right_col, text="Seleccionar Gráfica Derecha:", style="Compare.TLabel").pack(anchor='w', pady=(0, 10), fill='x')
        
        list_frame_right = ttk.Frame(right_col) 
        list_frame_right.pack(fill='x', pady=(0, 15))
        
        sb_right = ttk.Scrollbar(list_frame_right, orient='vertical')
        self.listbox_right = tk.Listbox(list_frame_right, yscrollcommand=sb_right.set, height=6, exportselection=False, 
                                        background='#ecf0f1', foreground='#2c3e50', font=('Helvetica', 10), relief='flat', borderwidth=0)
        sb_right.config(command=self.listbox_right.yview)
        
        sb_right.pack(side='right', fill='y')
        self.listbox_right.pack(side='left', fill='x', expand=True)
        self.listbox_right.bind('<<ListboxSelect>>', self.on_select_right)
        
        zoom_frame_right = ttk.Frame(right_col, style="CompareInner.TFrame")
        zoom_frame_right.pack(fill='x', pady=(0, 5))
        
        ttk.Button(zoom_frame_right, text="+", style="Zoom.TButton", command=lambda: self.zoom_in("right")).pack(side='left', padx=2)
        ttk.Button(zoom_frame_right, text="-", style="Zoom.TButton", command=lambda: self.zoom_out("right")).pack(side='left', padx=2)
        ttk.Button(zoom_frame_right, text="Reset", style="Zoom.TButton", command=lambda: self.reset_zoom_pan("right")).pack(side='left', padx=2)
        
        self.image_label_right = tk.Label(right_col, 
                                          text="Imagen Derecha", 
                                          background='#4a6076',  
                                          foreground='#bdc3c7',  
                                          font=('Helvetica', 16, 'italic'),
                                          anchor='center')
        self.image_label_right.pack(fill='both', expand=True, pady=(0, 0)) 
        
        self.image_label_right.bind("<Button-1>", lambda event: self.on_button_press(event, "right"))
        self.image_label_right.bind("<B1-Motion>", lambda event: self.on_mouse_drag(event, "right"))


    def setup_main_app(self, parent_frame):
        control_frame = ttk.Frame(parent_frame, padding="20")
        control_frame.pack(side="left", fill="y", padx=10, pady=10)

        # --- ELEMENTOS FIJOS AL FONDO (BOTTOM) ---
        
        # 1. Estado
        self.status_label = ttk.Label(control_frame, text="Listo. Por favor, cargue los archivos.", font=('Helvetica', 11, 'italic'))
        self.status_label.pack(side='bottom', anchor='w', pady=10)

        # 2. Botones de Acción (Visualizar / Salir)
        action_button_frame = ttk.Frame(control_frame)
        action_button_frame.pack(side='bottom', fill='x', pady=(0, 10))
        
        self.view_clusters_button = ttk.Button(action_button_frame, text="Visualizar Grupos (Semántica)", command=self.show_clusters_window, state="disabled")
        self.view_clusters_button.pack(side="left", expand=True, fill='x', ipady=5, padx=(0, 5))
        
        exit_button = ttk.Button(action_button_frame, text="Salir", command=self.root.destroy)
        exit_button.pack(side="right", expand=True, fill='x', ipady=5, padx=(5, 0))

        # 3. Botón Ejecutar
        self.run_button = ttk.Button(control_frame, text="Ejecutar Análisis", command=self.run_analysis_thread, state="disabled")
        self.run_button.pack(side='bottom', fill='x', ipady=5, pady=(10, 10))

        # --- ELEMENTOS DESDE ARRIBA (TOP) - DISEÑO COMPACTO ---
        
        ttk.Label(control_frame, text="1. Cargar Archivos", style="Title.TLabel").pack(side='top', pady=(10, 10), anchor='w')

        # Fila 1: Coordenadas
        ttk.Button(control_frame, text="Seleccionar Coordenadas", command=self.select_coords_file).pack(side='top', fill='x', ipady=3)
        ttk.Label(control_frame, textvariable=self.coords_filepath, style="Path.TLabel").pack(side='top', anchor='w', pady=(2, 5))
        
        # Fila 2: Matriz (Lado a Lado)
        matrix_frame = ttk.Frame(control_frame)
        matrix_frame.pack(side='top', fill='x', pady=5)
        
        self.generate_matrix_button = ttk.Button(
            matrix_frame, 
            text="Generar Matriz", 
            command=self.generate_matrix_file_thread,
            state="disabled" 
        )
        self.generate_matrix_button.pack(side='left', fill='x', expand=True, ipady=3, padx=(0, 2))

        ttk.Button(matrix_frame, text="Cargar Matriz", command=self.select_matrix_files).pack(side='right', fill='x', expand=True, ipady=3, padx=(2, 0))
        
        self.matrix_files_label = ttk.Label(control_frame, text="(Opcional) Se calculará en memoria si está vacío", style="Path.TLabel", wraplength=300)
        self.matrix_files_label.pack(side='top', anchor='w', pady=(2, 5))

        # Fila 3: Verificar
        ttk.Button(control_frame, text="Cargar y Verificar Datos", command=self.load_data_thread).pack(side='top', fill='x', ipady=3, pady=(5, 0))
        
        # Separador y Sección Semántica
        ttk.Separator(control_frame, orient='horizontal').pack(side='top', fill='x', pady=10)
        ttk.Label(control_frame, text="Clase 12: Semántica", style="Title.TLabel").pack(side='top', pady=(5, 5), anchor='w')
        
        # Fila 4: Variables y Diccionario (Lado a Lado)
        vars_frame = ttk.Frame(control_frame)
        vars_frame.pack(side='top', fill='x', pady=5)

        ttk.Button(vars_frame, text="Cargar Tabla Vars", 
                   command=self.load_variables_file).pack(side='left', fill='x', expand=True, ipady=3, padx=(0,2))
        
        ttk.Button(vars_frame, text="Cargar Diccionario", 
                   command=self.load_var_dictionary).pack(side='right', fill='x', expand=True, ipady=3, padx=(2,0))
        
        # Labels de estado pequeños
        status_vars_frame = ttk.Frame(control_frame)
        status_vars_frame.pack(side='top', fill='x')
        self.vars_status_label = ttk.Label(status_vars_frame, text="Vars: No", style="Path.TLabel")
        self.vars_status_label.pack(side='left', anchor='w')
        self.dict_status_label = ttk.Label(status_vars_frame, text="| Dic: No", style="Path.TLabel")
        self.dict_status_label.pack(side='left', anchor='w', padx=5)

        # Gráficas
        graph_button_frame = ttk.Frame(control_frame)
        graph_button_frame.pack(side='top', fill='x', pady=(10, 0))

        self.save_graph_button = ttk.Button(graph_button_frame, text="Guardar gráfica", command=self.save_graph, state="disabled")
        self.save_graph_button.pack(side="left", expand=True, fill='x', ipady=3, padx=(0, 5))
        
        self.compare_button = ttk.Button(graph_button_frame, text="Comparar gráficas", command=self.show_comparison)
        self.compare_button.pack(side="right", expand=True, fill='x', ipady=3, padx=(5, 0))

        ttk.Separator(control_frame, orient='horizontal').pack(side='top', fill='x', pady=15)

        ttk.Label(control_frame, text="2. Configurar Análisis", style="Title.TLabel").pack(side='top', pady=(5, 10), anchor='w')
        
        self.clustering_type = tk.StringVar(value="particional-agglomerative") 
        # ttk.Label(control_frame, text="Tipo de agrupamiento:").pack(side='top', anchor='w', pady=(5, 2))
        
        # Radiobuttons más compactos
        ttk.Radiobutton(control_frame, text="Particional (K-Means)", 
                        variable=self.clustering_type, value="particional-kmeans", 
                        command=self.update_options).pack(side='top', anchor='w')
        
        ttk.Radiobutton(control_frame, text="Particional (Aglomerativo)", 
                        variable=self.clustering_type, value="particional-agglomerative", 
                        command=self.update_options).pack(side='top', anchor='w')
        
        ttk.Radiobutton(control_frame, text="Jerárquico (Dendrograma)", 
                        variable=self.clustering_type, value="jerarquico", 
                        command=self.update_options).pack(side='top', anchor='w')

        # Opciones dinámicas
        self.k_options_frame = ttk.Frame(control_frame)
        self.k_label = ttk.Label(self.k_options_frame, text="Número de grupos (K):")
        self.k_label.pack(side='left', pady=(5, 5))
        self.k_value = tk.IntVar(value=5)
        self.k_spinbox = ttk.Spinbox(self.k_options_frame, from_=2, to=50, textvariable=self.k_value, width=5, font=('Helvetica', 11))
        self.k_spinbox.pack(side='left', padx=10)
        
        self.options_label = ttk.Label(control_frame, text="Criterio:")
        self.method_var = tk.StringVar()
        self.method_menu = ttk.OptionMenu(control_frame, self.method_var, None)

        self.remove_outliers_var = tk.BooleanVar()
        self.remove_outliers_check = ttk.Checkbutton(control_frame, text="Eliminar Outliers", variable=self.remove_outliers_var)
        
        self.remove_outliers_check.pack(side='top', anchor='w', pady=(10, 0))

        self.progress_bar = ttk.Progressbar(control_frame, orient='horizontal', mode='determinate', style="Turquoise.Horizontal.TProgressbar")
        # Progress bar al fondo (encima de status)
        self.progress_bar.pack(side='bottom', fill='x', pady=5)

        # Configuración del área principal (Derecha)
        main_area = ttk.Frame(parent_frame)
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
        
        self.toolbar_frame = ttk.Frame(main_area)
        toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        toolbar.update()
        
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.explanation_text = tk.Text(main_area, height=10, wrap="word", bg='#34495e', fg='white', font=('Helvetica', 11), relief='flat', padx=10, pady=10)
        self.explanation_text.pack(side="bottom", fill="x", pady=(10, 0))

        self.update_options()
    
    def save_graph(self):
        try:
            os.makedirs(self.GRAPH_DIR, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Error de Directorio", f"No se pudo crear la carpeta '{self.GRAPH_DIR}':\n{e}")
            return

        try:
            base_name = os.path.splitext(self.coords_filepath.get())[0]
            if not base_name:
                base_name = "grafica" 

            tipo_agrupamiento = self.clustering_type.get()
            
            if tipo_agrupamiento == "particional-kmeans":
                k = self.k_value.get()
                suggested_name = f"{base_name}_KMeans_K{k}.png"
            elif tipo_agrupamiento == "particional-agglomerative":
                k = self.k_value.get()
                criterio = self.method_var.get()
                suggested_name = f"{base_name}_Agglomerative_K{k}_{criterio}.png"
            else: # jerarquico
                criterio = self.method_var.get()
                suggested_name = f"{base_name}_Dendrogram_{criterio}.png"
        
        except Exception:
            suggested_name = "grafica_guardada.png"


        filepath = filedialog.asksaveasfilename(
            initialdir=self.GRAPH_DIR,
            title="Guardar Gráfica",
            initialfile=suggested_name, 
            filetypes=[("PNG files", "*.png")],
            defaultextension=".png"
        )

        if not filepath:
            return

        try:
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=self.fig.get_facecolor())
            messagebox.showinfo("Éxito", f"Gráfica guardada en:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar la gráfica:\n{e}")

    def update_graph_lists(self):
        self.listbox_left.delete(0, "end")
        self.listbox_right.delete(0, "end")

        self.reset_zoom_pan("left")
        self.reset_zoom_pan("right")
        self.image_label_left.config(image=None, 
                                     text="Imagen Izquierda", 
                                     background='#4a6076', 
                                     foreground='#bdc3c7')
        self.image_label_right.config(image=None, 
                                      text="Imagen Derecha", 
                                      background='#4a6076', 
                                      foreground='#bdc3c7')
        self.img_tk_left = None
        self.img_tk_right = None

        if not os.path.isdir(self.GRAPH_DIR):
            self.listbox_left.insert("end", "No hay gráficas guardadas.")
            self.listbox_right.insert("end", "No hay gráficas guardadas.")
            return

        try:
            files = [f for f in os.listdir(self.GRAPH_DIR) if f.lower().endswith('.png')]
            if not files:
                self.listbox_left.insert("end", "No hay gráficas guardadas.")
                self.listbox_right.insert("end", "No hay gráficas guardadas.")
                return

            for f in files:
                self.listbox_left.insert("end", f)
                self.listbox_right.insert("end", f)
        except Exception as e:
            messagebox.showerror("Error al leer directorio", f"No se pudo leer la carpeta de gráficas:\n{e}")

    def on_select_left(self, event):
        if not self.listbox_left.curselection():
            return 
        
        try:
            selected_index = self.listbox_left.curselection()[0]
            filename = self.listbox_left.get(selected_index)
        except IndexError:
            return 

        if filename:
            self.load_image_to_label(filename, self.image_label_left, "left")

    def on_select_right(self, event):
        if not self.listbox_right.curselection():
            return 

        try:
            selected_index = self.listbox_right.curselection()[0]
            filename = self.listbox_right.get(selected_index)
        except IndexError:
            return 

        if filename:
            self.load_image_to_label(filename, self.image_label_right, "right")

    def load_image_to_label(self, filename, label, side):
        
        if not filename or filename == "No hay gráficas guardadas":
            label.config(text="Seleccione una imagen", 
                         image='', 
                         background='#4a6076', 
                         foreground='#bdc3c7')
            label._original_image = None
            label._zoom_level = 1.0
            label._pan_x = 0
            label._pan_y = 0
            label._last_x = 0
            label._last_y = 0
            if side == "left": self.img_tk_left = None
            else: self.img_tk_right = None
            return

        filepath = os.path.join(self.GRAPH_DIR, filename)
        if not os.path.exists(filepath):
            messagebox.showwarning("Error", f"El archivo no se encontró:\n{filename}")
            return
            
        try:
            img = Image.open(filepath)
            
            label._original_image = img
            
            self.reset_zoom_pan(side)


        except Exception as e:
            label.config(text=f"Error al cargar:\n{filename}\n{e}", 
                         image='', 
                         background='#4a6076', 
                         foreground='#bdc3c7')
            label._original_image = None
            label._zoom_level = 1.0
            label._pan_x = 0
            label._pan_y = 0
            label._last_x = 0
            label._last_y = 0
            if side == "left": self.img_tk_left = None
            else: self.img_tk_right = None

    def _update_display_image(self, label, side):
        """Actualiza la imagen que se muestra en el label según el zoom y paneo."""
        if not hasattr(label, '_original_image') or label._original_image is None:
            label.config(text="Imagen Izquierda" if side == "left" else "Imagen Derecha",
                         image='',
                         background='#4a6076',  
                         foreground='#bdc3c7')
            if side == "left": self.img_tk_left = None
            else: self.img_tk_right = None
            return

        original_width, original_height = label._original_image.size
        
        zoomed_width = original_width / label._zoom_level
        zoomed_height = original_height / label._zoom_level

        half_view_width = zoomed_width / 2
        half_view_height = zoomed_height / 2
        
        label._pan_x = max(half_view_width, min(original_width - half_view_width, label._pan_x))
        label._pan_y = max(half_view_height, min(original_height - half_view_height, label._pan_y))
        
        crop_left = label._pan_x - half_view_width
        crop_top = label._pan_y - half_view_height
        crop_right = label._pan_x + half_view_width
        crop_bottom = label._pan_y + half_view_height
        
        cropped_img = label._original_image.crop((int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)))
        
        scaled_img = cropped_img.resize((self.COMPARE_IMG_WIDTH, self.COMPARE_IMG_HEIGHT), Image.Resampling.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(scaled_img)
        label.config(image=img_tk, text="", background='#34495e')
        
        if side == "left":
            self.img_tk_left = img_tk
        else:
            self.img_tk_right = img_tk

    def zoom_in(self, side):
        label = self.image_label_left if side == "left" else self.image_label_right
        if not hasattr(label, '_original_image') or label._original_image is None:
            return
        
        label._zoom_level *= 1.2 
        label._zoom_level = min(label._zoom_level, 10.0) 
        self._update_display_image(label, side)

    def zoom_out(self, side):
        label = self.image_label_left if side == "left" else self.image_label_right
        if not hasattr(label, '_original_image') or label._original_image is None:
            return
            
        label._zoom_level /= 1.2 
        label._zoom_level = max(1.0, label._zoom_level) 
        self._update_display_image(label, side)

    def on_button_press(self, event, side):
        label = self.image_label_left if side == "left" else self.image_label_right
        if not hasattr(label, '_original_image') or label._original_image is None:
            return
        label._last_x = event.x
        label._last_y = event.y

    def on_mouse_drag(self, event, side):
        label = self.image_label_left if side == "left" else self.image_label_right
        if not hasattr(label, '_original_image') or label._original_image is None or label._zoom_level == 1.0:
            return 

        dx = event.x - label._last_x
        dy = event.y - label._last_y
        
        original_width, original_height = label._original_image.size
        
        scale_factor = (original_width / label._zoom_level) / self.COMPARE_IMG_WIDTH
        
        label._pan_x -= dx * scale_factor
        label._pan_y -= dy * scale_factor

        label._last_x = event.x
        label._last_y = event.y
        
        self._update_display_image(label, side)

    def reset_zoom_pan(self, side):
        """Resetea el zoom y paneo a la vista original."""
        label = self.image_label_left if side == "left" else self.image_label_right
        if not hasattr(label, '_original_image') or label._original_image is None:
            label.config(text="Imagen Izquierda" if side == "left" else "Imagen Derecha",
                         image='',
                         background='#4a6076',  
                         foreground='#bdc3c7')
            label._original_image = None
            label._zoom_level = 1.0
            label._pan_x = 0
            label._pan_y = 0
            label._last_x = 0
            label._last_y = 0
            return
        
        label._zoom_level = 1.0
        label._pan_x = label._original_image.width / 2 
        label._pan_y = label._original_image.height / 2
        self._update_display_image(label, side)

    def select_coords_file(self):
        filepath = filedialog.askopenfilename(title="Seleccionar archivo de coordenadas", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filepath:
            self.coords_filepath.set(os.path.basename(filepath))
            self._full_coords_path = filepath
            self.generate_matrix_button.config(state="normal")
        else:
            self.generate_matrix_button.config(state="disabled")
            self._full_coords_path = None

    def select_matrix_files(self):
        filepaths = filedialog.askopenfilenames(title="Seleccionar 1 o 2 archivos de matriz", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filepaths:
            self.matrix_filepaths = filepaths
            basenames = [os.path.basename(p) for p in filepaths]
            self.matrix_files_label.config(text=", ".join(basenames))

    def update_options(self):
        if hasattr(self, 'save_graph_button'):
             self.save_graph_button.config(state="disabled")

        c_type = self.clustering_type.get()
        
        # Configurar menú de enlace (solo para métodos aglomerativos)
        menu = self.method_menu["menu"]
        menu.delete(0, "end")
        
        self.options_label.config(text="Criterio para unir grupos:")
        
        # --- CAMBIO: AÑADIDO EL MÉTODO WARD ---
        options = {
            "Ward (Estándar)": "ward",
            "Single (Vecino más cercano)": "single",
            "Average (Promedio)": "average",
            "Complete (Vecino más lejano)": "complete"
        }
        
        current_method = self.method_var.get()
        if not current_method: 
            self.method_var.set("ward") # Default cambiado a Ward
        
        def set_and_update(value):
            self.method_var.set(value)
            self.update_explanation()

        for display_name, value in options.items():
            menu.add_command(label=display_name, command=lambda v=value: set_and_update(v))

        # --- CORRECCIÓN DE VISIBILIDAD (FIXED CRASH) ---
        # 1. Primero ocultamos TODOS los elementos dinámicos para limpiar la vista
        self.k_options_frame.pack_forget()
        self.options_label.pack_forget()
        self.method_menu.pack_forget()

        # 2. Volvemos a mostrar SOLO lo necesario, usando el orden correcto para 'before'
        # El widget referenciado en 'before' DEBE estar visible.
        
        if c_type == "particional-kmeans":
            # Solo K. Se pone antes de outliers (que siempre es visible)
            self.k_options_frame.pack(side='top', fill='x', anchor='w', before=self.remove_outliers_check)
            self.view_clusters_button.config(state="disabled")
            
        elif c_type == "particional-agglomerative":
            # K + Tipo de Enlace.
            # Orden de apilado (de abajo hacia arriba para usar before correctamente):
            # 1. Menu (antes de outliers)
            # 2. Label (antes de menu)
            # 3. K Frame (antes de label)
            
            self.method_menu.pack(side='top', anchor='w', padx=20, fill='x', before=self.remove_outliers_check)
            self.options_label.pack(side='top', anchor='w', pady=(5, 0), before=self.method_menu)
            self.k_options_frame.pack(side='top', fill='x', anchor='w', before=self.options_label)
            
            self.view_clusters_button.config(state="disabled")

        elif c_type == "jerarquico":
            # Solo Tipo de Enlace (sin K)
            self.method_menu.pack(side='top', anchor='w', padx=20, fill='x', before=self.remove_outliers_check)
            self.options_label.pack(side='top', anchor='w', pady=(5, 0), before=self.method_menu)
            
            self.view_clusters_button.config(state="disabled")

        self.update_explanation()
        
        if hasattr(self, 'coords_df') and self.coords_df is not None:
            self.plot_data()

    # --- FUNCIÓN DE EXPLICACIÓN ---
    def update_explanation(self, result_text=None):
        self.explanation_text.config(state="normal")
        self.explanation_text.delete(1.0, "end")
        
        c_type = self.clustering_type.get()
        method = self.method_var.get()
        
        linkage_desc = {
            "ward": "Ward Minimization. Minimiza la varianza dentro de los grupos. Es el estándar general para datos cuantitativos.",
            "complete": "Criterio estricto (Complete Linkage). Une dos grupos solo si el miembro MÁS LEJANO de un grupo está cerca del MÁS LEJANO del otro. Crea grupos muy compactos y separados, como 'islas'.",
            "average": "Criterio balanceado (Average Linkage). Calcula la distancia PROMEDIO entre TODOS los miembros de un grupo y TODOS los del otro. Es el más común y suele dar resultados muy intuitivos.",
            "single": "Criterio optimista (Single Linkage). Para unir dos grupos, basta con que solo UN miembro de un grupo esté cerca de UNO del otro. A veces puede crear grupos largos, como una 'cadena'."
        }

        info_text = ""
        if c_type == "particional-kmeans":
            info_text = (
                "AGRUPAMIENTO PARTICIONAL (K-MEANS):\n\n"
                "Este es el método particional más común. Define K 'centroides' (centros promedio, no puntos reales) y asigna cada punto al centroide más cercano.\n\n"
                "VENTAJA: Es muy rápido y escalable.\n"
                "REQUISITO: **Ignora la Matriz de Distancia.** Usa las **coordenadas originales** para calcular los promedios."
            )
        elif c_type == "particional-agglomerative":
            info_text = (
                "AGRUPAMIENTO PARTICIONAL (AGLOMERATIVO):\n\n"
                "Este método usa el algoritmo jerárquico pero lo detiene cuando se alcanza el número K de grupos deseado. Es útil cuando ya sabes cuántos grupos esperas encontrar.\n\n"
                "VENTAJA: **Usa la Matriz de Distancia** precalculada.\n"
                f"Criterio de unión: {linkage_desc.get(method, '')}"
            )
        else: # jerarquico
            info_text = (
                "AGRUPAMIENTO JERÁRQUICO (DENDROGRAMA):\n\n"
                "Este método no necesita que le digas cuántos grupos buscar. Construye un 'árbol familiar' (dendrograma) que muestra cómo se unen los puntos desde lo individual hasta formar un solo grupo.\n\n"
                "VENTAJA: **Usa la Matriz de Distancia** y permite ver la estructura completa de los datos.\n"
                f"Criterio de unión: {linkage_desc.get(method, '')}"
            )

        if result_text:
            info_text += f"\n\n--- RESULTADO ---\n{result_text}"

        self.explanation_text.insert(1.0, info_text)
        self.explanation_text.config(state="disabled")

    def load_data_thread(self):
        # Valida que se tengan coordenadas (para K-Means) Y matriz (para otros)
        if not hasattr(self, '_full_coords_path') or not self._full_coords_path:
            messagebox.showwarning("Archivos Faltantes", "Por favor, seleccione un archivo de coordenadas.")
            return
        
        self.status_label.config(text="Cargando datos...")
        self.run_button.config(state="disabled")
        self.view_clusters_button.config(state="disabled")
        self.save_graph_button.config(state="disabled")

        self.progress_bar.pack(side='bottom', fill='x', pady=5, before=self.status_label)
        
        thread = threading.Thread(target=self.load_data, daemon=True)
        thread.start()

    def on_data_load_error(self, error_message):
        self.hide_progress_bar()
        self.progress_bar['value'] = 0
        self.status_label.config(text=error_message)

    def load_data(self):
        try:
            # 1. Cargar Coordenadas
            coords_path = self._full_coords_path
            self.coords_df = pd.read_csv(coords_path, sep='\t')
            self.coords_df['Longitude'] = pd.to_numeric(self.coords_df['Longitude'], errors='coerce')
            self.coords_df['Latitude'] = pd.to_numeric(self.coords_df['Latitude'], errors='coerce')
            self.coords_df.dropna(subset=['Longitude', 'Latitude'], inplace=True)
            
            # 2. Cargar Matriz (Opcional)
            if self.matrix_filepaths:
                self.root.after(0, lambda: self.status_label.config(text="Cargando matriz (puede tardar)..."))
                
                # Pre-calcular líneas para barra de progreso
                total_lines = 0
                for filepath in self.matrix_filepaths:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        total_lines += sum(1 for line in f)
                self.root.after(0, lambda: self.progress_bar.config(maximum=total_lines, value=0))
                
                self.processed_lines = 0
                def update_progress():
                    self.processed_lines += 1
                    if self.processed_lines % 100 == 0: # Actualizar cada 100 líneas para no congelar UI
                        self.progress_bar['value'] = self.processed_lines

                def load_dist_part_manually(filepath, has_header_and_index=False, update_callback=None):
                    data = []
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        if has_header_and_index:
                            try: next(f)
                            except StopIteration: pass

                        for line in f:
                            if update_callback:
                                self.root.after(0, update_callback) 
                            
                            line = line.strip()
                            if not line: continue
                            
                            if '\t' in line: parts = line.split('\t')
                            else: parts = line.split()
                            
                            start_index = 1 if has_header_and_index else 0
                            
                            if len(parts) > start_index:
                                try:
                                    row = [float(p) for p in parts[start_index:] if p.strip()]
                                    data.append(row)
                                except ValueError: continue
                    
                    if not data: return np.array([])
                    min_len = min(len(r) for r in data)
                    truncated_data = [r[:min_len] for r in data]
                    return np.array(truncated_data, dtype=np.float32)

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
                
                # --- VALIDACIÓN DE MATRIZ ---
                if self.dist_matrix is not None:
                    rows, cols = self.dist_matrix.shape
                    # Check si cargó coordenadas en lugar de matriz (error común)
                    if cols == 3 and rows > 100: 
                         raise ValueError(f"El archivo cargado tiene dimensiones {rows}x{cols}.\nParece que seleccionaste el archivo de COORDENADAS en lugar de la MATRIZ.\nPor favor selecciona un archivo NxN o deja el campo vacío para calcular en memoria.")
                    
                    if rows != cols:
                         # Si no es cuadrada, invalidamos pero no crasheamos (aviso al usuario)
                         self.root.after(0, lambda: messagebox.showwarning("Matriz no cuadrada", f"La matriz cargada ({rows}x{cols}) no es cuadrada.\nSe intentará usar, pero puede causar errores. Se recomienda usar 'Generar Matriz'."))

            else:
                # Si no hay matriz, la calculamos en memoria
                self.root.after(0, lambda: self.status_label.config(text="Calculando matriz en memoria (Rápido)..."))
                self.dist_matrix = self.calculate_distance_matrix(self.coords_df)

            self.root.after(0, self.on_data_loaded)

        except Exception as e:
            messagebox.showerror("Error al Cargar Datos", f"Ocurrió un error:\n{e}")
            self.root.after(0, self.on_data_load_error, "Error al cargar los datos.")

    def on_data_loaded(self):
        self.progress_bar['value'] = self.progress_bar['maximum']
        self.status_label.config(text="Datos cargados. Listo para analizar.")
        self.run_button.config(state="normal")
        
        self.save_graph_button.config(state="disabled") 
        
        self.root.after(5000, self.hide_progress_bar)
        
        self.plot_data() 
        self.update_explanation()

    def generate_matrix_file_thread(self):
        if self._full_coords_path is None:
            messagebox.showwarning("Archivo Faltante", "Por favor, seleccione un archivo de coordenadas primero.")
            return

        self.status_label.config(text="Calculando y generando matriz...")
        self.generate_matrix_button.config(state="disabled")
        self.run_button.config(state="disabled")

        self.progress_bar.pack(side='bottom', fill='x', pady=5, before=self.status_label)
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()

        thread = threading.Thread(target=self.generate_matrix_file, daemon=True)
        thread.start()

    def calculate_distance_matrix(self, coords_df):
        if coords_df is None or coords_df.empty:
            raise ValueError("No hay datos de coordenadas válidos para calcular la matriz.")
            
        X = coords_df[['Longitude', 'Latitude']].values
        
        dist_matrix = euclidean_distances(X)
        
        return dist_matrix

    def generate_matrix_file(self):
        try:
            self.root.after(0, lambda: self.status_label.config(text="Cargando coordenadas..."))
            coords_path = self._full_coords_path
            coords_df = pd.read_csv(coords_path, sep='\t')
            coords_df['Longitude'] = pd.to_numeric(coords_df['Longitude'], errors='coerce')
            coords_df['Latitude'] = pd.to_numeric(coords_df['Latitude'], errors='coerce')
            coords_df.dropna(subset=['Longitude', 'Latitude'], inplace=True)
            
            if coords_df.empty:
                raise ValueError("El archivo de coordenadas no contiene datos válidos de Longitud/Latitud.")
                
            self.root.after(0, lambda: self.status_label.config(text="Calculando matriz de distancia..."))
            dist_matrix_np = self.calculate_distance_matrix(coords_df)
            
            save_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Guardar Matriz de Distancia Calculada"
            )
            
            if not save_path:
                self.root.after(0, lambda: self.status_label.config(text="Generación cancelada por el usuario."))
                return

            self.root.after(0, lambda: self.status_label.config(text=f"Guardando archivo en: {os.path.basename(save_path)}"))
            
            np.savetxt(save_path, dist_matrix_np, delimiter='\t', fmt='%.10f')
            
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
    
    # --- Carga y Selección de Variables ---
    def load_variables_file(self):
        filepath = filedialog.askopenfilename(title="Seleccionar Tabla de Variables", 
                                              filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not filepath: return

        try:
            # Cargar asumiendo tabuladores 
            self.variables_df = pd.read_csv(filepath, sep='\t', encoding='latin-1', low_memory=False)
            self.vars_status_label.config(text=f"Cargado: {os.path.basename(filepath)}")
            self.vars_status_label.config(text=f"Vars: OK") # Compacto
            
            # --- Ventana para elegir columnas ---
            self.select_variables_dialog()
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la tabla de variables:\n{e}")

    # --- NUEVO: CARGA DEL DICCIONARIO DE VARIABLES ---
    def load_var_dictionary(self):
        filepath = filedialog.askopenfilename(title="Seleccionar Archivo Diccionario (Variables)", 
                                              filetypes=[("Archivos de Datos", "*.xls *.xlsx *.csv *.txt"), ("Todos los archivos", "*.*")])
        if not filepath: return

        try:
            df = None
            # Intentar leer automáticamente
            try:
                if filepath.lower().endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(filepath)
                else:
                    df = pd.read_csv(filepath, sep=None, engine='python', encoding='latin-1')
            except ImportError:
                 messagebox.showerror("Error de Librería", "Falta la librería 'xlrd' o 'openpyxl' para leer Excel.\nInstala con: pip install xlrd openpyxl")
                 return
            except Exception:
                # Reintento cruzado: Si falla Excel, probar como CSV (a veces tienen extension .xls pero son texto)
                try:
                    df = pd.read_csv(filepath, sep=None, engine='python', encoding='latin-1')
                except:
                    pass # Si falla este también, df sigue siendo None

            if df is None:
                 messagebox.showerror("Error de Lectura", "No se pudo interpretar el formato del archivo.")
                 return

            # Limpiar nombres de columnas
            df.columns = df.columns.str.strip()
            
            # Buscar columna de claves y valores (Búsqueda flexible mejorada)
            key_col = None
            val_col = None
            
            # Normalizar nombres de columnas a mayúsculas para buscar
            cols_upper = [c.upper() for c in df.columns]
            
            # Posibles nombres para la columna CLAVE (Z1, Z2...)
            possible_keys = ['VARIABLE', 'CLAVE', 'CODIGO', 'ID', 'VAR']
            # Posibles nombres para la columna VALOR (Descripción)
            possible_vals = ['NOMBRE', 'DESCRIP', 'CONCEPTO', 'ETIQUETA', 'LABEL']

            # Buscar índice de la columna
            for i, col in enumerate(cols_upper):
                if any(k in col for k in possible_keys):
                    key_col = df.columns[i]
                if any(v in col for v in possible_vals):
                    val_col = df.columns[i]
            
            # Fallback: Si no encuentra cabeceras claras, usar columna 0 como clave y 1 como valor si hay pocas columnas
            if (key_col is None or val_col is None) and len(df.columns) >= 2:
                 key_col = df.columns[0]
                 val_col = df.columns[1]
            
            if key_col and val_col:
                # Crear diccionario
                self.var_dict = pd.Series(df[val_col].values, index=df[key_col]).to_dict()
                # Limpiar y normalizar
                self.var_dict = {str(k).strip(): str(v).strip() for k, v in self.var_dict.items()}
                
                self.dict_status_label.config(text=f"| Dic: OK ({len(self.var_dict)})") # Compacto
                messagebox.showinfo("Éxito", f"Se cargaron {len(self.var_dict)} definiciones de variables.\nColumna Clave detectada: {key_col}\nColumna Nombre detectada: {val_col}")
            else:
                messagebox.showwarning("Error de Estructura", f"No se pudieron identificar las columnas de 'Variable' y 'Descripción' automáticamente.\nColumnas encontradas: {list(df.columns)}")

        except Exception as e:
            messagebox.showerror("Error Crítico", f"Ocurrió un error al procesar el archivo:\n{e}")

    def select_variables_dialog(self):
        if self.variables_df is None: return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Selección de Variables (Semántica)")
        dialog.geometry("500x600") # Un poco más ancho para ver nombres si ya están cargados
        
        ttk.Label(dialog, text="Selecciona las variables para caracterizar los grupos:").pack(pady=10)
        
        # Lista con checkbox
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        vars_vars = {} # Diccionario para guardar estado de checkboxes
        
        # Omitimos la primera columna si es el ID/Codigo
        cols = self.variables_df.columns[1:] 
        
        for col in cols:
            var = tk.BooleanVar()
            
            # Si tenemos diccionario, mostramos el nombre real
            display_text = col
            if col in self.var_dict:
                display_text = f"{col} - {self.var_dict[col]}"
            
            chk = ttk.Checkbutton(scrollable_frame, text=display_text, variable=var)
            chk.pack(anchor='w', padx=10)
            vars_vars[col] = var
            
        def save_selection():
            self.selected_vars = [col for col, var in vars_vars.items() if var.get()]
            if not self.selected_vars:
                messagebox.showwarning("Atención", "No seleccionaste ninguna variable.")
                return
            messagebox.showinfo("Listo", f"Se analizarán {len(self.selected_vars)} variables en los grupos.")
            dialog.destroy()
            
        ttk.Button(dialog, text="Confirmar Selección", command=save_selection).pack(pady=10, fill='x')
    
    def run_analysis_thread(self):
        self.run_button.config(state="disabled")
        self.view_clusters_button.config(state="disabled")
        self.save_graph_button.config(state="disabled")
        self.status_label.config(text=f"Ejecutando {self.clustering_type.get()}...")
        
        thread = threading.Thread(target=self.perform_clustering, daemon=True)
        thread.start()

    def perform_clustering(self):
        c_type = self.clustering_type.get()
        method = self.method_var.get()
        
        try:
            current_coords = self.coords_df.copy()
            
            # --- SEGURIDAD SI NO HAY MATRIZ ---
            if self.dist_matrix is None:
                 self.root.after(0, lambda: messagebox.showinfo("Aviso", "Calculando matriz de distancias en memoria (esto puede tardar unos segundos)..."))
                 current_matrix = self.calculate_distance_matrix(current_coords)
            else:
                 current_matrix = self.dist_matrix.copy()

            n_points_coords = len(current_coords)
            n_points_matrix = current_matrix.shape[0]
            
            # Ajuste simple si hay diferencia pequeña (por encabezados o lineas vacías)
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

            # --- PREPARAR COORDENADAS (Usado por KMeans y Ward) ---
            X_coords = current_coords[['Longitude', 'Latitude']].values
            
            # --- LÓGICA PARA K-MEANS ---
            if c_type == 'particional-kmeans':
                k = self.k_value.get()
                
                # K-Means USA LAS COORDENADAS, NO LA MATRIZ
                model = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = model.fit_predict(X_coords)
                
                medoid_indices = self.calculate_medoids(labels, current_matrix)
                
                self.cluster_results = current_coords.copy()
                self.cluster_results['Grupo'] = labels
                result_text = f"Se crearon {k} grupos con K-Means. Los medoides (puntos reales más céntricos de cada grupo) se muestran como estrellas."
                self.root.after(0, self.on_analysis_complete, {'labels': labels, 'coords': current_coords, 'medoids': medoid_indices, 'result_text': result_text})

            # --- LÓGICA PARA AGLOMERATIVO ---
            elif c_type == 'particional-agglomerative':
                k = self.k_value.get()
                
                # CORRECCIÓN WARD: Si es Ward, usa 'euclidean' y coordenadas. Si no, usa 'precomputed' y matriz.
                if method == 'ward':
                    model = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
                    labels = model.fit_predict(X_coords)
                else:
                    model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage=method)
                    labels = model.fit_predict(current_matrix)
                
                medoid_indices = self.calculate_medoids(labels, current_matrix)
                
                self.cluster_results = current_coords.copy()
                self.cluster_results['Grupo'] = labels
                result_text = f"Se crearon {k} grupos con método Aglomerativo ({method}). Los puntos más céntricos (medoides) se muestran como estrellas."
                self.root.after(0, self.on_analysis_complete, {'labels': labels, 'coords': current_coords, 'medoids': medoid_indices, 'result_text': result_text})

            # --- LÓGICA JERÁRQUICA ---
            else: # 'jerarquico'
                # CORRECCIÓN WARD: Igual que arriba
                if method == 'ward':
                    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0, metric='euclidean', linkage='ward')
                    model.fit(X_coords)
                else:
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
        
        # Safety check for matrix dimensions
        if matrix.shape[0] < len(labels) or matrix.shape[1] < len(labels):
             print("Warning: Matrix dimension mismatch in calculate_medoids. Returning empty.")
             return []

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) == 1:
                medoid_indices.append(cluster_indices[0])
                continue
            if len(cluster_indices) == 0:
                continue
            try:
                sub_matrix = matrix[np.ix_(cluster_indices, cluster_indices)]
                sum_distances = np.sum(sub_matrix, axis=1)
                min_dist_idx_in_cluster = np.argmin(sum_distances)
                medoid_original_idx = cluster_indices[min_dist_idx_in_cluster]
                medoid_indices.append(medoid_original_idx)
            except Exception:
                pass # Fallback if calculation fails
                
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

    # --- CAMBIO IMPORTANTE: VENTANA DE RESULTADOS SEMÁNTICOS MEJORADA ---
    def show_clusters_window(self):
        if self.cluster_results is None:
            messagebox.showinfo("Sin Resultados", "Primero ejecuta el análisis.")
            return

        top = tk.Toplevel(self.root)
        top.title("Interpretación Semántica de Grupos - Semántica")
        top.geometry("1200x800") # Más ancho para el split
        top.configure(bg='#34495e')

        try:
            # Preparar datos
            analysis_df = self.cluster_results.copy()
            
            resumen_semantico = None
            if self.variables_df is not None and self.selected_vars:
                # Validar indices para evitar crash si los archivos no alinean perfecto
                common_indices = self.cluster_results.index.intersection(self.variables_df.index)
                
                # Alinear variables con coordenadas filtradas
                vars_subset = self.variables_df.loc[common_indices, self.selected_vars].copy()
                
                # --- CORRECCIÓN CRÍTICA: FORZAR A NÚMEROS ---
                # Esto evita que el groupby falle silenciosamente si hay texto en las celdas
                for col in vars_subset.columns:
                    vars_subset[col] = pd.to_numeric(vars_subset[col], errors='coerce')
                
                # Unir usando indices alineados
                analysis_df = analysis_df.loc[common_indices].join(vars_subset)
                
                # Calcular promedios
                resumen_semantico = analysis_df.groupby('Grupo')[self.selected_vars].mean()
                resumen_semantico = resumen_semantico.fillna(0) # Rellenar vacios con 0 para no romper la grafica
            
            # Calcular promedios globales para comparacion
            global_means = analysis_df[self.selected_vars].mean()

            notebook = ttk.Notebook(top)
            notebook.pack(pady=10, padx=10, expand=True, fill="both")

            # --- Pestaña 1: Interpretación Gráfica (LO NUEVO) ---
            if resumen_semantico is not None and not resumen_semantico.empty:
                frame_interp = ttk.Frame(notebook, padding="10")
                notebook.add(frame_interp, text='Interpretación (Gráfica y Texto)')
                
                # --- ESTRUCTURA DIVIDIDA ---
                # Frame Superior (Split Horizontal: Tabla | Gráfica)
                top_split_frame = ttk.Frame(frame_interp)
                top_split_frame.pack(side='top', fill='both', expand=True, pady=(0, 10))
                
                # Parte Izquierda: Tabla
                left_panel = ttk.Frame(top_split_frame)
                left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
                
                lbl = ttk.Label(left_panel, text="Tabla de Promedios (Máximos en amarillo):", font=('Helvetica', 10, 'bold'))
                lbl.pack(side='top', fill='x')
                
                self.create_table_in_frame(left_panel, resumen_semantico.reset_index(), highlight_max=True)

                # Parte Derecha: Gráfica
                right_panel = ttk.Frame(top_split_frame)
                right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
                
                fig_sem, ax_sem = plt.subplots(figsize=(5, 4)) # Ajustar tamaño para medio panel
                
                # Graficar solo si hay columnas válidas
                if not resumen_semantico.columns.empty:
                    # --- NUEVO: Usar nombres reales en la gráfica si existen ---
                    plot_data = resumen_semantico.copy()
                    new_cols = []
                    for col in plot_data.columns:
                        if col in self.var_dict:
                            # Cortar nombres muy largos para la gráfica
                            name = self.var_dict[col]
                            if len(name) > 15: name = name[:15] + "..."
                            new_cols.append(name)
                        else:
                            new_cols.append(col)
                    plot_data.columns = new_cols
                    
                    plot_data.plot(kind='bar', ax=ax_sem, width=0.8)
                    ax_sem.set_title('Perfil Semántico (Promedios)', fontsize=10)
                    ax_sem.set_xlabel('Grupos')
                    # ax_sem.set_ylabel('Valor Promedio')
                    ax_sem.grid(True, linestyle='--', alpha=0.5)
                    ax_sem.legend(fontsize='small')
                    plt.tight_layout()
                else:
                    ax_sem.text(0.5, 0.5, "No se pudieron calcular promedios numéricos.", ha='center')

                canvas_sem = FigureCanvasTkAgg(fig_sem, master=right_panel)
                canvas_sem.draw()
                canvas_sem.get_tk_widget().pack(fill='both', expand=True)

                # --- SECCIÓN DE INTERPRETACIÓN TEXTUAL (FONDO, MÁS ESPACIO) ---
                text_frame = ttk.Frame(frame_interp, padding="5")
                text_frame.pack(side='bottom', fill='both', expand=True) # Expand True para ocupar el resto
                
                lbl_text = ttk.Label(text_frame, text="Análisis Semántico Automatizado:", font=('Helvetica', 12, 'bold'), foreground='#2c3e50') 
                lbl_text.pack(anchor='w')
                
                interp_text_widget = tk.Text(text_frame, height=15, wrap="word", bg="#ecf0f1", fg="#2c3e50", font=('Helvetica', 11))
                scrollbar_text = ttk.Scrollbar(text_frame, orient="vertical", command=interp_text_widget.yview)
                interp_text_widget.configure(yscrollcommand=scrollbar_text.set)
                
                scrollbar_text.pack(side="right", fill="y")
                interp_text_widget.pack(side="left", fill="both", expand=True)
                
                # Generar texto
                interpretation = self.generate_text_interpretation(resumen_semantico, global_means)
                interp_text_widget.insert("1.0", interpretation)
                interp_text_widget.config(state="disabled")


            # Pestaña 2: Detalle completo (Datos crudos)
            frame_det = ttk.Frame(notebook, padding="10")
            notebook.add(frame_det, text='Detalle de Datos (Completo)')
            self.create_table_in_frame(frame_det, analysis_df)

        except Exception as e:
            messagebox.showerror("Error Visualización", f"Ocurrió un error al generar las gráficas:\n{e}")
            top.destroy()

    def generate_text_interpretation(self, group_means, global_means):
        text = "INFORME DE CARACTERIZACIÓN DE GRUPOS\n\n"
        
        # Función auxiliar para obtener nombre
        def get_name(code):
            if code in self.var_dict:
                return f"{self.var_dict[code]} ({code})"
            return code

        for group_idx, row in group_means.iterrows():
            text += f"▶ GRUPO {group_idx}:\n"
            
            # 1. Variables donde este grupo es el líder (Máximo valor)
            # Comparamos con el maximo de la columna en todo el dataframe de promedios
            is_max_mask = row == group_means.max()
            max_vars = row.index[is_max_mask].tolist()
            
            if max_vars:
                # Convertir códigos a nombres
                vars_str = ", ".join([get_name(v) for v in max_vars])
                text += f"   • ZONA DE ALTA CONCENTRACIÓN: Este grupo se distingue por ser la zona predominante para: {vars_str}.\n"
                text += f"     Interpretación: Los valores en estas variables son los más altos de toda la región analizada.\n"
            
            # 2. Comparación con promedio global (Variables por encima de la media)
            above_avg_mask = row > global_means
            above_vars = row.index[above_avg_mask].tolist()
            
            secondary_vars = [v for v in above_vars if v not in max_vars]
            
            if secondary_vars:
                 vars_str = ", ".join([get_name(v) for v in secondary_vars])
                 text += f"   • TENDENCIA POSITIVA: Además, presenta valores superiores al promedio general en: {vars_str}.\n"
            
            # 3. Variables muy bajas (Mínimo valor)
            is_min_mask = row == group_means.min()
            min_vars = row.index[is_min_mask].tolist()
            if min_vars:
                 vars_str = ", ".join([get_name(v) for v in min_vars])
                 text += f"   • INDICADORES BAJOS: Registra los niveles mínimos de la región en: {vars_str}.\n"
            
            text += "\n" + ("-"*50) + "\n\n"
            
        return text

    def create_table_in_frame(self, parent, dataframe, highlight_max=False):
        cols = list(dataframe.columns)
        tree = ttk.Treeview(parent, columns=cols, show='headings', height=8)
        
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')
            
        # Configurar tag para resaltar
        tree.tag_configure('max_val', background='#f1c40f', foreground='black') # Amarillo
        
        # Precalcular maximos por columna si se pide resaltar
        max_vals = {}
        if highlight_max:
            for col in cols:
                if col != 'Grupo': # No resaltar columna grupo
                    try:
                        max_vals[col] = dataframe[col].max()
                    except:
                        pass

        for index, row in dataframe.iterrows():
            formatted_row = []
            row_tags = [] 
            
            is_max_row = False
            
            for col_name in cols:
                item = row[col_name]
                if isinstance(item, float):
                    formatted_row.append(f"{item:.2f}")
                    # Chequeo simple de maximo
                    if highlight_max and col_name in max_vals and abs(item - max_vals[col_name]) < 0.001:
                         is_max_row = True # Marca la fila si tiene algún máximo
                else:
                    formatted_row.append(item)
            
            if is_max_row and highlight_max:
                tree.insert("", "end", values=formatted_row, tags=('max_val',))
            else:
                tree.insert("", "end", values=formatted_row)
            
        vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        tree.pack(expand=True, fill='both')

    def on_analysis_complete(self, results):
        result_text = results['result_text']
        
        if 'labels' in results:
            medoids = results.get('medoids')
            self.plot_data(results['labels'], results['coords'], medoids)
            
            c_type = self.clustering_type.get()
            if c_type == "particional-kmeans" or c_type == "particional-agglomerative":
                 self.view_clusters_button.config(state="normal")
                 
        elif 'linkage_matrix' in results:
            title_info = results.get('title_info', '')
            self.plot_dendrogram(results['linkage_matrix'], title_info)

        self.status_label.config(text="Análisis completado.")
        self.run_button.config(state="normal")
        self.save_graph_button.config(state="normal")
        self.update_explanation(result_text)

    def reset_ui_after_error(self):
        self.hide_progress_bar()
        self.progress_bar['value'] = 0
        self.status_label.config(text="Error. Intente de nuevo.")
        self.run_button.config(state="normal")
        self.view_clusters_button.config(state="disabled")
        self.save_graph_button.config(state="disabled")

    def plot_dendrogram(self, linkage_matrix, title_info=""):
        self.toolbar_frame.pack(side="top", fill="x", before=self.canvas.get_tk_widget())
        self.ax.clear()
        R = dendrogram(
            linkage_matrix,
            ax=self.ax,
            truncate_mode='level', 
            p=5,                   
            show_leaf_counts=True,
            leaf_rotation=90.,
            leaf_font_size=8.,
            show_contracted=True,
            link_color_func=lambda k: '#1abc9c'
        )
        
        min_dist = np.min(linkage_matrix[:, 2])
        if min_dist <= 0:
            self.ax.set_yscale('linear')
            self.ax.set_ylabel("Distancia (escala lineal)", color='white')
        else:
            self.ax.set_yscale('log')
            self.ax.set_ylabel("Distancia (escala logarítmica)", color='white')

        
        try:
            dcoords = [item for sublist in R['dcoord'] for item in sublist]
            dcoords_nonzero = [i for i in dcoords if i > 0]
            
            if dcoords_nonzero:
                max_visible_dist = max(dcoords_nonzero)
                min_visible_dist = min(dcoords_nonzero)
                
                if self.ax.get_yscale() == 'log':
                    ylim_top = max_visible_dist * 1.5 
                    ylim_bottom = min_visible_dist / 2 
                else:
                    margin = (max_visible_dist - min_visible_dist) * 0.1
                    ylim_top = max_visible_dist + margin
                    ylim_bottom = max(0, min_visible_dist - margin) 

                self.ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

        except Exception as e:
            print(f"No se pudo auto-ajustar el zoom del dendrograma: {e}")
        
        full_title = "Dendrograma del Clustering Jerárquico" + title_info
        self.ax.set_title(full_title, color='white', fontsize=14)
        self.ax.set_xlabel("Índice del Cluster (o puntos en el cluster)", color='white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_color('none')
        self.ax.spines['right'].set_color('none')
        self.ax.set_facecolor('#34495e')

        self.fig.tight_layout()
        self.canvas.draw()
        
        self.save_graph_button.config(state="normal")

    def plot_data(self, labels=None, coords=None, medoids=None):
        self.toolbar_frame.pack_forget()
        self.ax.clear()
        
        plot_coords = coords if coords is not None else self.coords_df
        
        if plot_coords is None or plot_coords.empty:
            self.canvas.draw()
            if hasattr(self, 'save_graph_button'):
                self.save_graph_button.config(state="disabled")
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
        
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_color('none')
        self.ax.spines['right'].set_color('none')
        self.ax.set_facecolor('#34495e')

        if labels is not None:
            unique_labels = np.unique(labels)
            color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Grupo {i}', 
                                    markerfacecolor=color_map[i], markersize=8) for i in unique_labels]
            
            if medoids is not None:
                 handles.append(plt.Line2D([0], [0], marker='*', color='yellow', label='Medoide',
                                            markeredgecolor='black', markersize=12, linestyle='None'))

            self.ax.legend(handles=handles, title="Clusters", labelcolor='white', facecolor='#2c3e50', edgecolor='white')

        self.fig.tight_layout()
        self.canvas.draw()
        
        if hasattr(self, 'save_graph_button'):
            self.save_graph_button.config(state="normal")


if __name__ == "__main__":
    try:
        os.makedirs("Imagenes_graficas", exist_ok=True)
    except OSError:
        print("No se pudo crear el directorio 'Imagenes_graficas'.")
        
    root = tk.Tk()
    app = ClusteringApp(root)
    root.mainloop()