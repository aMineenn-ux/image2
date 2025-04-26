import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import boutondessin
import Codequimarche

class SVGAppWithBouton(Codequimarche.SVGApp):
    def __init__(self, master):
        super().__init__(master)
        # Ajout des boutons dans le même frame
        btn_boutondessin = ttk.Button(
            self.btn_process.master,
            text="Image→ SVG",
            command=self.image_to_svg_integrated,
            style="Accent.TButton",
            width=30
        )
        btn_boutondessin.pack(side=tk.LEFT, padx=5, pady=5)

        btn_paint = ttk.Button(
            self.btn_process.master,
            text="Ouvrir Paint",
            command=self.open_paint,
            style="Accent.TButton",
            width=30
        )
        btn_paint.pack(side=tk.LEFT, padx=5, pady=5)

    def open_paint(self):
        from paint import ModernDrawingProgram
        self.paint_window = ModernDrawingProgram(self.master)

    def image_to_svg_integrated(self):
        input_path = filedialog.askopenfilename(
            parent=self.master,
            title="Sélectionnez une image d'entrée",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Tous les fichiers", "*.*")]
        )
        if not input_path:
            self.set_status("Annulé : Pas de fichier d'entrée sélectionné.")
            return

        base_name = input_path.split("/")[-1].rsplit(".", 1)[0]
        suggested_svg_name = base_name + ".svg"
        initial_dir = "output"

        output_path = filedialog.asksaveasfilename(
            parent=self.master,
            title="Enregistrer le fichier SVG de sortie sous...",
            initialdir=initial_dir,
            initialfile=suggested_svg_name,
            defaultextension=".svg",
            filetypes=[("Fichier SVG", "*.svg"), ("Tous les fichiers", "*.*")]
        )
        if not output_path:
            self.set_status("Annulé : Pas de fichier de sortie sélectionné.")
            return

        self.set_status(f"Traitement en cours...\nEntrée: {base_name}\nSortie: {output_path.split('/')[-1]}")
        self.master.config(cursor="watch")
        self.master.update_idletasks()

        try:
            boutondessin.main(
                input_path=input_path,
                output_path=output_path,
                no_hatch=True,
                no_contour=False,
                show_bmp=False,
                no_cv_mode=boutondessin.no_cv
            )
            messagebox.showinfo("Succès", f"Conversion terminée !\nSVG enregistré sous:\n{output_path}", parent=self.master)
            self.set_status("Terminé avec succès.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue :\n{e}", parent=self.master)
            self.set_status(f"Erreur : {e}")
        finally:
            self.master.config(cursor="")

if __name__ == "__main__":
    import os
    
    # --- Crée la fenêtre principale UNE SEULE FOIS ---
    root = tk.Tk()

    # --- Crée l'instance de l'application UNE SEULE FOIS ---
    app = SVGAppWithBouton(root)

    # --- Démarre la boucle d'événements Tkinter pour la fenêtre principale ---
    root.mainloop()
