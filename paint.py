import tkinter as tk  # Use tk alias for clarity
from tkinter import ttk  # Import themed widgets
from tkinter import filedialog, messagebox, colorchooser
import svgwrite

class ModernDrawingProgram:
    def __init__(self, parent):
        self.root = tk.Toplevel(parent)
        self.root.title("Modern Drawing Program - Integrated")
        # More flexible starting size, allow resizing later if needed
        # self.root.geometry("800x520") # Removed fixed size for now
        self.root.minsize(600, 400) # Set a minimum size

        # --- Style Configuration ---
        self.style = ttk.Style(self.root)
        # Try different themes: 'clam', 'alt', 'default', 'vista' (Windows), 'aqua' (macOS)
        try:
            # 'clam' often looks cleaner across platforms
            self.style.theme_use('clam')
        except tk.TclError:
            print("Theme 'clam' not available, using default.")

        # Define custom styles if needed (example)
        self.style.configure('Tool.TButton', padding=5, font=('Helvetica', 10))
        self.style.configure('Control.TFrame', background='#e0e0e0') # Light grey background for controls
        self.style.configure('Canvas.TFrame', background='white')

        # --- Variables ---
        self.pen_color = "black"
        self.eraser_color = "white" # Canvas background color should match eraser
        self.active_tool_color = self.pen_color # Stores the actual color to draw with
        self.canvas_bg_color = 'white' # Define canvas bg color explicitly
        self.drawn_objects = []

        # Tracks mouse status and position
        self.left_but = "up"
        self.previous_x = None
        self.previous_y = None

        # Tracks the active tool button for visual feedback
        self.active_button = None

        # --- Main Layout ---
        self.root.configure(background='#d0d0d0') # Overall window background

        # Configure grid weights for resizing behavior
        # Column 0 (Controls) fixed width, Column 1 (Canvas) expands
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        # Row 0 expands vertically
        self.root.grid_rowconfigure(0, weight=1)

        # --- Control Panel (Left Side) ---
        self.control_frame = ttk.Frame(self.root, padding=10, style='Control.TFrame')
        self.control_frame.grid(row=0, column=0, sticky='nswe') # North, South, West, East
        self.control_frame.grid_rowconfigure(5, weight=1) # Add weight to push size slider down if needed

        # --- Canvas Area (Right Side) ---
        self.canvas_frame = ttk.Frame(self.root, style='Canvas.TFrame', relief=tk.SUNKEN, borderwidth=1)
        self.canvas_frame.grid(row=0, column=1, sticky='nswe', padx=5, pady=5)
        # Make canvas expand within its frame
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)


        # --- Create Widgets ---
        self._create_control_widgets()
        self._create_canvas()

        # --- Initialize Tool State ---
        self.use_pen() # Start with pen active


    def _create_control_widgets(self):
        """Creates widgets for the control panel."""
        current_row = 0

        # --- Tool Buttons ---
        self.pen_button = ttk.Button(self.control_frame, text='Pen',
                                     command=self.use_pen, style='Tool.TButton', width=10)
        self.pen_button.grid(row=current_row, column=0, pady=5, sticky='ew')
        current_row += 1

        self.eraser_button = ttk.Button(self.control_frame, text='Eraser',
                                        command=self.use_eraser, style='Tool.TButton', width=10)
        self.eraser_button.grid(row=current_row, column=0, pady=5, sticky='ew')
        current_row += 1

        # --- Color Picker ---
        self.color_button = ttk.Button(self.control_frame, text='Color',
                                       command=self.choose_color, style='Tool.TButton', width=10)
        self.color_button.grid(row=current_row, column=0, pady=5, sticky='ew')
        # Small label/box to show the selected color
        self.color_preview = tk.Label(self.control_frame, bg=self.pen_color, width=4, height=1, relief=tk.RIDGE)
        self.color_preview.grid(row=current_row, column=1, pady=5, padx=5, sticky='w')
        current_row += 1


        # --- Action Buttons ---
        self.clear_button = ttk.Button(self.control_frame, text='Clear',
                                       command=self.clear_canvas, style='Tool.TButton', width=10)
        self.clear_button.grid(row=current_row, column=0, pady=5, sticky='ew')
        current_row += 1

        self.save_button = ttk.Button(self.control_frame, text='Save SVG',
                                      command=self.save_as_svg, style='Tool.TButton', width=10)
        self.save_button.grid(row=current_row, column=0, pady=(5, 20), sticky='ew') # More padding below save
        current_row += 1

        # --- Pen Size Scale ---
        # Using a standard tk Scale as ttk.Scale styling can be tricky
        self.pen_size_frame = ttk.LabelFrame(self.control_frame, text="Brush Size", padding=(10, 5))
        self.pen_size_frame.grid(row=current_row, column=0, columnspan=2, pady=10, sticky='ew')
        # Make scale expand horizontally within its frame
        self.pen_size_frame.grid_columnconfigure(0, weight=1)

        self.pen_size = tk.Scale(self.pen_size_frame, orient=tk.HORIZONTAL, from_=1, to=50,
                                 bg=self.style.lookup('Control.TFrame', 'background'), # Match frame bg
                                 troughcolor='#c0c0c0', # Color of the slider groove
                                 length=120, sliderlength=20)
        self.pen_size.set(5)
        self.pen_size.grid(row=0, column=0, sticky='ew')
        current_row += 1


    def _create_canvas(self):
        """Creates the drawing canvas."""
        canvas_width=700
        canvas_height=500
        self.canvas = tk.Canvas(self.canvas_frame, bg=self.canvas_bg_color,
                                 width=canvas_width, height=canvas_height, # Start with a size
                                 highlightthickness=0) # Remove default border
        self.canvas.grid(row=0, column=0, sticky='nswe')

        # Store size for SVG export
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height


        # Bind mouse events
        self.canvas.bind("<Motion>", self.motion) # Track motion even if button is up
        self.canvas.bind("<ButtonPress-1>", self.left_but_down)
        self.canvas.bind("<ButtonRelease-1>", self.left_but_up)
        self.canvas.bind("<B1-Motion>", self.paint)


    # --- Event Handlers ---

    def left_but_down(self, event=None):
        self.left_but = "down"
        # Record position immediately on click
        self.previous_x = event.x
        self.previous_y = event.y

    def left_but_up(self, event=None):
        self.left_but = "up"
        # Reset last position to avoid disconnected lines on next drag
        self.previous_x = None
        self.previous_y = None

    def motion(self, event=None):
        # Could be used for cursor updates or hover effects later
        pass

    def paint(self, event):
        if self.left_but == "down":
            current_x, current_y = event.x, event.y
            if self.previous_x is not None and self.previous_y is not None:
                # Get size and active color
                size = self.pen_size.get()
                color = self.active_tool_color

                # Draw line segment
                line = self.canvas.create_line(self.previous_x, self.previous_y, current_x, current_y,
                                               fill=color, width=size,
                                               capstyle=tk.ROUND, smooth=tk.TRUE) # Nicer lines

                # Store *only* if not erasing
                if color != self.eraser_color:
                    self.drawn_objects.append(
                        (self.previous_x, self.previous_y, current_x, current_y,
                         color, size) # Store the actual color used
                    )
            # Update previous position for the next segment
            self.previous_x = current_x
            self.previous_y = current_y

    # --- Tool Selection ---

    def _activate_button(self, button_to_activate):
        """Visually activates a tool button using ttk states."""
        # Reset the previously active button state if it exists
        if self.active_button:
            # Clear the 'pressed' state from the old button
            # '!pressed' means "not pressed"
            s = self.active_button.state()
            if 'pressed' in s: # Check if it was pressed before clearing
                self.active_button.state(['!pressed'])

        # Activate the new button by setting the 'pressed' state
        # 'pressed' usually makes the button look sunken or active
        button_to_activate.state(['pressed'])
        self.active_button = button_to_activate

    def use_pen(self):
        self.active_tool_color = self.pen_color # Use the stored pen color
        self._activate_button(self.pen_button)

    def use_eraser(self):
        self.active_tool_color = self.eraser_color
        self._activate_button(self.eraser_button)

    def choose_color(self):
        color_code = colorchooser.askcolor(title="Choose Pen Color", initialcolor=self.pen_color)
        if color_code and color_code[1]: # Check if a color was selected (color_code[1] is the hex string)
            self.pen_color = color_code[1]
            self.color_preview.config(bg=self.pen_color)
            self.use_pen() # Switch back to pen tool with the new color

    # --- Actions ---

    def clear_canvas(self):
        if messagebox.askyesno("Clear Canvas", "Are you sure you want to clear everything?"):
            self.canvas.delete("all")
            self.drawn_objects = [] # Clear saved drawing data too

    def save_as_svg(self):
        try:
            # Suggest a filename, filter for SVG files
            filename = filedialog.asksaveasfilename(
                defaultextension='.svg',
                filetypes=[("Scalable Vector Graphics", "*.svg"), ("All Files", "*.*")],
                title="Save Drawing As SVG"
            )

            # Check if the user cancelled the dialog
            if not filename:
                return

            # Get current canvas size (might have changed if window resized)
            # Note: This reads the *widget* size, which might differ slightly from drawn content bounds.
            # For perfect SVG bounds, you might need to calculate the bounding box of self.drawn_objects
            # but using widget size is usually sufficient.
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Create SVG drawing object
            dwg = svgwrite.Drawing(filename, profile='tiny', size=(canvas_width, canvas_height))

            # Add a background rectangle if desired (matches canvas color)
            # dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill=self.canvas_bg_color))

            # Add drawn lines
            for x0, y0, x1, y1, color, width in self.drawn_objects:
                # SVG uses 'stroke' for line color and 'stroke-width'
                dwg.add(dwg.line(start=(x0, y0), end=(x1, y1),
                                 stroke=color, stroke_width=width,
                                 stroke_linecap='round')) # Match canvas capstyle

            # Save the file
            dwg.save()
            messagebox.showinfo('Drawing Saved', f'Image saved as\n{filename}')

        except Exception as e:
            messagebox.showerror("Save Error", f"Unable to save image:\n{e}")


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernDrawingProgram(root)
    root.mainloop()
