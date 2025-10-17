# utils/plot_utils.py
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def embed_plot(parent, figure: Figure):
    """Embed a matplotlib Figure in a Tkinter Frame."""
    canvas = FigureCanvasTkAgg(figure, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas
