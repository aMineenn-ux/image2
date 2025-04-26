class SVGApp:
    def __init__(self, master):
        self.master = master
        master.title("Contrôleur de Trajectoire Robot - Style Clam")
        master.geometry("1000x750")

        # --- Forcer le thème CLAM et récupérer les couleurs ---
        self.style = ttk.Style(master)
        self.clam_bg_color = "#d9d9d9" # Couleur typique clam (à ajuster)
        self.clam_fg_color = "#000000" # Noir
        self.clam_accent_color = "#0078D7" # Bleu windows/office comme accent
        self.clam_button_color = "#e1e1e1"
        self.clam_disabled_fg = "#a3a3a3"
        self.clam_plot_bg = "#ffffff" # Fond blanc pour le plot

        try:
            self.style.theme_use('clam')
            print("Thème 'clam' appliqué.")
            # Essayer de récupérer les vraies couleurs du thème clam
            # Note: Les clés peuvent varier (TFrame, TButton etc.)
            try: self.clam_bg_color = self.style.lookup('TFrame', 'background')
            except: pass # Garder la valeur par défaut
            try: self.clam_fg_color = self.style.lookup('TLabel', 'foreground')
            except: pass
            try: self.clam_button_color = self.style.lookup('TButton', 'background')
            except: pass
            # L'accent et plot bg sont souvent mieux définis manuellement
        except tk.TclError:
            print("AVERTISSEMENT: Thème 'clam' non trouvé, utilisation des styles par défaut Tkinter.")
            # Dans ce cas, les couleurs personnalisées seront quand même utilisées

        # Appliquer couleur de fond principale à la fenêtre racine
        master.config(bg=self.clam_bg_color)

        # --- Redéfinir les couleurs de classe avec les couleurs CLAM ---
        self.BG_COLOR = self.clam_bg_color
        self.FG_COLOR = self.clam_fg_color
        self.ACCENT_COLOR = self.clam_accent_color
        self.BUTTON_COLOR = self.clam_button_color
        self.BUTTON_FG = self.clam_fg_color # Texte noir sur bouton clair
        self.DISABLED_FG = self.clam_disabled_fg
        self.PLOT_BG = self.clam_plot_bg
        self.RED_COLOR = "#E53935" # Rouge un peu plus vif

        # --- Variables d'état (inchangées) ---
        self.selected_svg_file = None
        self.dataframe = None
        self.output_csv_file = "optimized_points.csv"
        # ... (autres variables d'état : simulation_index, simulation_running, etc.) ...
        self.simulation_index = 0
        self.simulation_running = False
        self.default_min_x, self.default_max_x = 0, 4000
        self.default_min_y, self.default_max_y = 0, 3000
        self.ani = None
        self.animation_running = False
        self.start_time = 0
        self.current_frame = 0
        self.logo_photo = None
        self.placeholder_img = None
        self.svg_preview_photo = None
        self.paused = False
        self.paused_time = 0
        self.accumulated_pause = 0
        self.pause_start_time = 0

        # Données pour l'animation (initialisées à None)
        self.regular_time_stamps = None
        self.smooth_x_interpolated = None
        self.smooth_y_interpolated = None
        self.colors = None # Note: c'était regular_colors dans l'original

        # --- Configuration des styles ttk (basés sur les couleurs Clam) ---
        self._configure_styles()

        # --- Layout Frames (utilise le style App.TFrame maintenant défini) ---
        top_frame = ttk.Frame(master, style="App.TFrame", padding=(10, 5))
        top_frame.pack(side=tk.TOP, fill=tk.X)

        action_frame = ttk.Frame(master, style="App.TFrame", padding=(10, 5))
        action_frame.pack(side=tk.TOP, fill=tk.X)

        plot_frame = ttk.Frame(master, style="App.TFrame", padding=(5, 0))
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        status_frame = ttk.Frame(master, style="Status.TFrame", padding=(0, 2))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)


        # --- Widgets Haut (Logo + Fichier) ---
        # Le logo utilisera un Label tk standard, donc besoin de configurer son bg
        try:
            logo_img_path = "amine png vangogh.png" # Garder ton nom de fichier
            if os.path.exists(logo_img_path):
                logo_img = Image.open(logo_img_path)
                logo_img = logo_img.resize((48, 48), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                logo_label = tk.Label(top_frame, image=self.logo_photo, bg=self.BG_COLOR) # bg tk Label
                logo_label.pack(side=tk.LEFT, padx=(10, 20), pady=5)
            else: print(f"Info: Logo '{logo_img_path}' non trouvé.")
        except Exception as e: print(f"Erreur chargement logo: {e}")

        # Boutons ttk utilisent les styles configurés
        self.btn_load = ttk.Button(top_frame, text="Charger SVG", command=self.load_svg, style="Accent.TButton", width=15)
        self.btn_load.pack(side=tk.LEFT, padx=5, pady=10)

        # Label ttk utilise style configuré
        self.lbl_filename = ttk.Label(top_frame, text="Aucun fichier SVG chargé", style="Filename.TLabel", width=50, anchor="w")
        self.lbl_filename.pack(side=tk.LEFT, padx=15, pady=10, fill=tk.X, expand=True)

        # --- Widgets Actions ---
        self.btn_process = ttk.Button(action_frame, text="Traiter SVG (points)", command=self.process_svg, state=tk.DISABLED)
        self.btn_process.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_arduino = ttk.Button(action_frame, text="Traiter pour Arduino", command=self.traitement_arduino, state=tk.DISABLED) # Désactivé au début
        self.btn_arduino.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_realtime = ttk.Button(action_frame, text="Simuler Trajectoire", command=self.start_real_time_simulation, style="Accent.TButton", state=tk.DISABLED) # Désactivé au début
        self.btn_realtime.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_pause = ttk.Button(action_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=5, pady=5)


        # --- Zone de Plot Matplotlib (fond blanc) ---
        self.fig = Figure(figsize=(7, 6), dpi=100, facecolor=self.PLOT_BG) # Fond blanc pour le plot
        self.ax = self.fig.add_subplot(111)
        self._setup_plot_style() # Configurer couleurs axes, etc.

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.config(bg=self.PLOT_BG) # Fond tk widget du canvas = fond du plot
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        # Charger l'image placeholder (vérifier le nom)
        self.placeholder_img_path = "coe.png" # Garder ton nom
        self.load_placeholder_image(self.placeholder_img_path)
        self.show_placeholder_image()

        # Barre d'outils Matplotlib (adapter au style Clam)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self._style_matplotlib_toolbar(toolbar) # Appliquer style adapté
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0,5))

        # --- Barre de Statut ---
        self.status_label = ttk.Label(status_frame, text="Prêt. Chargez un fichier SVG.", style="Status.TLabel", anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=3)

        self.update_button_states()