from imgui_bundle import imgui

print("imgui module attrs:", [a for a in dir(imgui) if "glyph" in a.lower() or "range" in a.lower()][:50])
io = imgui.get_io()
print("io.fonts attrs:", [a for a in dir(io.fonts) if "glyph" in a.lower() or "range" in a.lower()][:50])
print("imgui version:", imgui.get_version())
