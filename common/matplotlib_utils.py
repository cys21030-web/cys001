from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def configure_matplotlib_chinese() -> str:
    """Configure Matplotlib to use a Chinese-capable font with Latin fallback."""
    preferred_fonts = [
        "Droid Sans Fallback",
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "sans-serif",
    ]

    available_fonts = {font.name for font in fm.fontManager.ttflist}
    selected_font = "DejaVu Sans"
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            selected_font = font_name
            break

    mpl.rcParams.update({
        "font.family": [selected_font, "DejaVu Sans", "sans-serif"],
        "font.sans-serif": [selected_font, "DejaVu Sans", "Arial Unicode MS", "Noto Sans CJK TC", "Noto Sans CJK SC"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "mathtext.fontset": "dejavusans",
    })
    plt.rcParams.update(mpl.rcParams)
    return selected_font
