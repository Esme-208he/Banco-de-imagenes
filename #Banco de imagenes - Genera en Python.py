
"""
Banco de Imágenes - Explorador de Archivos > Imágenes
======================================================
Abre el Explorador de Archivos de Windows en la carpeta Imágenes,
te deja elegir carpeta / imágenes, las muestra como explorador
(miniaturas, nombre, tamaño) y extrae características con diferentes
algoritmos con parámetros configurables.

Requisitos: pip install opencv-python numpy pandas scikit-image pillow

Uso:
    python banco_imagenes_explorador.py
"""
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk

try:
    from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ================= 1. PARÁMETROS POR ALGORITMO =================
PARAMS = {
    "color_hist": {"bins_h": 32, "bins_s": 32, "bins_v": 32, "espacio_color": "HSV", "normalizar": True},
    "hog": {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2), "block_norm": "L2-Hys", "tamano_resize": (128, 128)},
    "sift": {"nfeatures": 500, "nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6},
    "orb": {"nfeatures": 500, "scaleFactor": 1.2, "nlevels": 8, "edgeThreshold": 31, "patchSize": 31},
    "lbp": {"P": 24, "R": 3, "method": "uniform", "bins_hist": 26, "tamano_resize": (128, 128)},
    "hu": {"umbral_binario": 127, "max_val": 255, "log_transform": True},
    "glcm": {"distances": [1, 2], "angles": [0, np.pi/4, np.pi/2, 3*np.pi/4], "levels": 256,
             "propiedades": ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"],
             "tamano_resize": (128, 128)},
}
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ================= 2. EXTRACCIÓN =================
def extract_color_hist(img, p=PARAMS["color_hist"]):
    conv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if p["espacio_color"]=="HSV" else img
    hist = cv2.calcHist([conv],[0,1,2],None,[p["bins_h"],p["bins_s"],p["bins_v"]],[0,180,0,256,0,256])
    if p["normalizar"]: hist = cv2.normalize(hist, hist).flatten()
    return hist.flatten()

def extract_hog(img, p=PARAMS["hog"]):
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), p["tamano_resize"])
    if HAS_SKIMAGE:
        return np.array(hog(gray, orientations=p["orientations"], pixels_per_cell=p["pixels_per_cell"],
                            cells_per_block=p["cells_per_block"], block_norm=p["block_norm"], feature_vector=True))
    h = cv2.HOGDescriptor(_winSize=p["tamano_resize"], _blockSize=(16,16), _blockStride=(8,8), _cellSize=(8,8), _nbins=p["orientations"])
    return h.compute(gray).flatten()

def extract_lbp(img, p=PARAMS["lbp"]):
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), p["tamano_resize"])
    if HAS_SKIMAGE:
        lbp_img = local_binary_pattern(gray, P=p["P"], R=p["R"], method=p["method"])
        hist,_ = np.histogram(lbp_img.ravel(), bins=p["bins_hist"], range=(0,p["bins_hist"]))
        return (hist.astype(float)/(hist.sum()+1e-7))
    return np.array([gray.mean(), gray.std()])

def extract_hu(img, p=PARAMS["hu"]):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, p["umbral_binario"], p["max_val"], cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    hu = cv2.HuMoments(cv2.moments(bw)).flatten()
    if p["log_transform"]: hu = -np.sign(hu)*np.log10(np.abs(hu)+1e-10)
    return hu

def extract_glcm(img, p=PARAMS["glcm"]):
    if not HAS_SKIMAGE: return np.zeros(18)
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), p["tamano_resize"])
    glcm = graycomatrix(gray, distances=p["distances"], angles=p["angles"], levels=p["levels"], symmetric=True, normed=True)
    feats=[]
    for prop in p["propiedades"]:
        v = graycoprops(glcm, prop); feats.extend(v.flatten().tolist()); feats.append(float(v.mean())); feats.append(float(v.std()))
    return np.array(feats)

def extract_keypoints(img, alg="orb"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if alg=="sift":
        p=PARAMS["sift"]
        try: det=cv2.SIFT_create(nfeatures=p["nfeatures"], nOctaveLayers=p["nOctaveLayers"], contrastThreshold=p["contrastThreshold"], edgeThreshold=p["edgeThreshold"], sigma=p["sigma"])
        except: det=cv2.SIFT_create()
    else:
        p=PARAMS["orb"]; det=cv2.ORB_create(nfeatures=p["nfeatures"], scaleFactor=p["scaleFactor"], nlevels=p["nlevels"], edgeThreshold=p["edgeThreshold"], patchSize=p["patchSize"])
    kp, des = det.detectAndCompute(gray, None)
    return kp, des

def extraer_todo(ruta):
    img = cv2.imread(str(ruta))
    if img is None: return None, None
    return img, {
        "color_hist": extract_color_hist(img),
        "hog": extract_hog(img),
        "lbp": extract_lbp(img),
        "hu": extract_hu(img),
        "glcm": extract_glcm(img),
        "orb_kp": len(extract_keypoints(img,"orb")[0]),
        "sift_kp": len(extract_keypoints(img,"sift")[0]),
    }

# ================= 3. EXPLORADOR DE ARCHIVOS / IMÁGENES =================
class ExploradorImagenes(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Banco de Imágenes — Explorador de Archivos > Imágenes")
        self.geometry("1100x650")
        self.carpeta = Path.home() / "Pictures"  # Carpeta Imágenes de Windows
        if not self.carpeta.exists():
            self.carpeta = Path.home() / "Imágenes"
        self.archivos = []
        self.thumbs = []
        self._build_ui()
        self.cargar_carpeta(self.carpeta)

    def _build_ui(self):
        top = ttk.Frame(self); top.pack(fill="x", padx=8, pady=8)
        ttk.Label(top, text="Carpeta:").pack(side="left")
        self.lbl_ruta = ttk.Label(top, text=str(self.carpeta), width=70); self.lbl_ruta.pack(side="left", padx=6)
        ttk.Button(top, text="📂 Abrir Explorador...", command=self.abrir_explorador).pack(side="left")
        ttk.Button(top, text="🖼️ Carpeta Imágenes", command=lambda: self.cargar_carpeta(Path.home()/"Pictures")).pack(side="left", padx=4)
        ttk.Button(top, text="📥 Exportar CSV", command=self.exportar_csv).pack(side="left", padx=4)

        # Panel parámetros
        p = ttk.LabelFrame(self, text="Parámetros de los algoritmos"); p.pack(fill="x", padx=8)
        self.vars = {}
        controles = [("HOG orientaciones", "hog", "orientations", 9), ("LBP P vecinos", "lbp", "P", 24),
                     ("LBP Radio R", "lbp", "R", 3), ("ORB nfeatures", "orb", "nfeatures", 500),
                     ("Color bins H", "color_hist", "bins_h", 32)]
        for i,(label, alg, key, default) in enumerate(controles):
            ttk.Label(p, text=label).grid(row=0, column=i*2, padx=4, pady=4, sticky="e")
            v = tk.IntVar(value=default); self.vars[(alg,key)] = v
            ttk.Spinbox(p, from_=1, to=1000, textvariable=v, width=7, command=self.aplicar_params).grid(row=0, column=i*2+1, padx=2)

        paned = ttk.PanedWindow(self, orient="horizontal"); paned.pack(fill="both", expand=True, padx=8, pady=8)
        # Izquierda: lista estilo explorador
        left = ttk.Frame(paned); paned.add(left, weight=2)
        self.tree = ttk.Treeview(left, columns=("nombre","tamano","dim"), show="headings")
        for c,t,w in [("nombre","Nombre",280),("tamano","Tamaño",90),("dim","Dimensiones",110)]:
            self.tree.heading(c, text=t); self.tree.column(c, width=w)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.al_seleccionar)
        # Derecha: vista previa + características
        right = ttk.Frame(paned); paned.add(right, weight=1)
        self.lbl_preview = ttk.Label(right, text="Selecciona una imagen"); self.lbl_preview.pack(pady=6)
        self.txt = tk.Text(right, height=22, width=48); self.txt.pack(fill="both", expand=True, padx=4)

    def abrir_explorador(self):
        # Abre el diálogo del Explorador de Archivos en la carpeta Imágenes
        carpeta = filedialog.askdirectory(title="Elige la carpeta de imágenes", initialdir=str(self.carpeta))
        if carpeta: self.cargar_carpeta(Path(carpeta))
        # Si quieres abrir el Explorador de Windows tal cual:
        # os.startfile(carpeta)  # solo Windows

    def cargar_carpeta(self, carpeta):
        self.carpeta = Path(carpeta); self.lbl_ruta.config(text=str(self.carpeta))
        self.archivos = sorted([f for f in self.carpeta.rglob("*") if f.suffix.lower() in EXTENSIONES])
        for i in self.tree.get_children(): self.tree.delete(i)
        for f in self.archivos:
            try: dim = Image.open(f).size; dim_s = f"{dim[0]}x{dim[1]}"
            except: dim_s = "-"
            self.tree.insert("", "end", values=(f.name, f"{f.stat().st_size//1024} KB", dim_s), tags=(str(f),))
        self.title(f"Banco de Imágenes — {len(self.archivos)} imágenes en {self.carpeta.name}")

    def aplicar_params(self):
        for (alg,key), var in self.vars.items(): PARAMS[alg][key] = var.get()

    def al_seleccionar(self, _event=None):
        sel = self.tree.selection()
        if not sel: return
        ruta = Path(self.tree.item(sel[0])["tags"][0])
        self.aplicar_params()
        img_cv, feats = extraer_todo(ruta)
        if feats is None: return
        # Vista previa como en el explorador
        pil = Image.open(ruta); pil.thumbnail((380, 280))
        photo = ImageTk.PhotoImage(pil); self.lbl_preview.config(image=photo, text=""); self.lbl_preview.image = photo
        # Mostrar características
        self.txt.delete("1.0","end")
        self.txt.insert("end", f"📁 {ruta.name}\n📂 {ruta.parent}\n\n")
        for k,v in feats.items():
            if isinstance(v, np.ndarray): self.txt.insert("end", f"• {k}: dim={len(v)} | media={v.mean():.4f} | min={v.min():.4f} | max={v.max():.4f}\n")
            else: self.txt.insert("end", f"• {k}: {v}\n")
        self.txt.insert("end", "\nParámetros actuales:\n")
        for alg, d in PARAMS.items(): self.txt.insert("end", f"  {alg}: {d}\n")

    def exportar_csv(self):
        import pandas as pd
        if not self.archivos: messagebox.showwarning("Sin imágenes","Primero abre una carpeta con imágenes."); return
        ruta_csv = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")], initialdir=str(self.carpeta), title="Guardar características como...")
        if not ruta_csv: return
        rows=[]
        for f in self.archivos:
            _, feats = extraer_todo(f)
            if not feats: continue
            row={"archivo": f.name, "ruta": str(f)}
            for k,v in feats.items():
                if isinstance(v, np.ndarray):
                    for i,val in enumerate(v): row[f"{k}_{i}"]=float(val)
                else: row[k]=v
            rows.append(row)
        pd.DataFrame(rows).to_csv(ruta_csv, index=False)
        messagebox.showinfo("Listo", f"Guardado en:\n{ruta_csv}\n{len(rows)} imágenes")

if __name__ == "__main__":
    ExploradorImagenes().mainloop()