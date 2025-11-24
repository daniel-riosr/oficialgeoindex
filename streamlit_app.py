import streamlit as st
import pandas as pd
import numpy as np
import time
import uuid
from abc import ABC, abstractmethod
from shapely.geometry import Point, box, shape, Polygon
import folium
from streamlit_folium import st_folium
import plotly.express as px

# ==========================================
# 0. CONFIGURACIÓN INICIAL
# ==========================================
st.set_page_config(layout="wide", page_title="GeoIndex Explorer: Comparativa Visual")

# ==========================================
# 1. MÓDULO DE SIMULACIÓN DE DISCO (LRU)
# ==========================================
class DiskSimulator:
    """
    Simula un gestor de buffer para contar accesos a 'disco'.
    Usa política LRU (Least Recently Used).
    """
    def __init__(self, page_size=4096, buffer_slots=50):
        self.page_size = page_size
        self.buffer_slots = buffer_slots
        self.buffer = [] 
        self.disk_accesses = 0
        self.hits = 0

    def reset_stats(self):
        self.disk_accesses = 0
        self.hits = 0
        self.buffer = []

    def access_node(self, node_id):
        if node_id in self.buffer:
            self.hits += 1
            # Mover al final (más reciente)
            self.buffer.remove(node_id)
            self.buffer.append(node_id)
        else:
            self.disk_accesses += 1
            if len(self.buffer) >= self.buffer_slots:
                self.buffer.pop(0) 
            self.buffer.append(node_id)

# Instancia global del simulador
if "disk_manager" not in st.session_state:
    st.session_state["disk_manager"] = DiskSimulator()

disk_manager = st.session_state["disk_manager"]

# ==========================================
# 2. CLASES DE ESTRUCTURAS DE DATOS
# ==========================================

class SpatialIndex(ABC):
    @abstractmethod
    def build(self, data_points): pass
    @abstractmethod
    def query(self, query_geometry): pass
    @abstractmethod
    def get_visual_geometry(self): pass

# --- KD-Tree ---
class KDNode:
    def __init__(self, point, axis, bounds):
        self.id = str(uuid.uuid4())
        self.point = point
        self.left = None
        self.right = None
        self.axis = axis
        self.bounds = bounds 

class KDTree(SpatialIndex):
    def __init__(self):
        self.root = None
        self.visual_splits = []

    def build(self, points):
        if not points: return
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        bounds = (min(lats), min(lons), max(lats), max(lons))
        self.root = self._build_recursive(points, 0, bounds)

    def _build_recursive(self, points, depth, bounds):
        if not points: return None
        axis = depth % 2 
        points.sort(key=lambda x: x[axis])
        mid = len(points) // 2
        node = KDNode(points[mid], axis, bounds)
        
        min_lat, min_lon, max_lat, max_lon = bounds
        if axis == 0: # Lat
            bounds_left = (min_lat, min_lon, node.point[0], max_lon)
            bounds_right = (node.point[0], min_lon, max_lat, max_lon)
            self.visual_splits.append({"type": "line", "coords": [(node.point[0], min_lon), (node.point[0], max_lon)]})
        else: # Lon
            bounds_left = (min_lat, min_lon, max_lat, node.point[1])
            bounds_right = (min_lat, node.point[1], max_lat, max_lon)
            self.visual_splits.append({"type": "line", "coords": [(min_lat, node.point[1]), (max_lat, node.point[1])]})

        node.left = self._build_recursive(points[:mid], depth + 1, bounds_left)
        node.right = self._build_recursive(points[mid+1:], depth + 1, bounds_right)
        return node

    def query(self, query_shape):
        found = []
        visited_nodes = []
        def _search(node):
            if node is None: return
            disk_manager.access_node(node.id)
            visited_nodes.append(node.bounds)
            if query_shape.contains(Point(node.point[0], node.point[1])):
                found.append(node.point)
            
            # Lógica simple de recorrido
            go_left = go_right = True # En implementación real se valida contra bounds
            if go_left: _search(node.left)
            if go_right: _search(node.right)

        if self.root: _search(self.root)
        return found, visited_nodes

    def get_visual_geometry(self):
        return self.visual_splits

# --- Quadtree ---
class QuadNode:
    def __init__(self, bbox, points=None, capacity=4):
        self.id = str(uuid.uuid4())
        self.bbox = bbox
        self.points = points or []
        self.children = []
        self.capacity = capacity

    def subdivide(self):
        minx, miny, maxx, maxy = self.bbox.bounds
        mx, my = (minx + maxx)/2, (miny + maxy)/2
        self.children = [
            QuadNode(box(minx, my, mx, maxy), capacity=self.capacity),
            QuadNode(box(mx, my, maxx, maxy), capacity=self.capacity),
            QuadNode(box(minx, miny, mx, my), capacity=self.capacity),
            QuadNode(box(mx, miny, maxx, my), capacity=self.capacity)
        ]
        for p in self.points:
            for c in self.children:
                if c.bbox.contains(Point(p[0], p[1])):
                    c.points.append(p)
                    break
        self.points = []

class QuadTree(SpatialIndex):
    def __init__(self, capacity=10):
        self.root = None
        self.capacity = capacity

    def build(self, points):
        if not points: return
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        bbox = box(min(lats)-0.001, min(lons)-0.001, max(lats)+0.001, max(lons)+0.001)
        self.root = QuadNode(bbox, points=list(points), capacity=self.capacity)
        nodes = [self.root]
        while nodes:
            n = nodes.pop(0)
            if len(n.points) > n.capacity:
                n.subdivide()
                nodes.extend(n.children)

    def query(self, query_shape):
        found = []
        visited_bounds = []
        def _search(node):
            if not node.bbox.intersects(query_shape): return
            disk_manager.access_node(node.id)
            visited_bounds.append(node.bbox.bounds)
            if not node.children:
                for p in node.points:
                    if query_shape.contains(Point(p[0], p[1])): found.append(p)
            else:
                for c in node.children: _search(c)
        if self.root: _search(self.root)
        return found, visited_bounds

    def get_visual_geometry(self):
        rects = []
        if not self.root: return []
        nodes = [self.root]
        while nodes:
            n = nodes.pop(0)
            if not n.children: rects.append(n.bbox.bounds)
            else: nodes.extend(n.children)
        return rects

# --- Grid File ---
class GridFile(SpatialIndex):
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.grid = {}
        self.bounds = None
        self.dx = 0; self.dy = 0

    def build(self, points):
        if not points: return
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        self.bounds = (min(lats), min(lons), max(lats), max(lons))
        self.dx = (self.bounds[2] - self.bounds[0]) / self.grid_size
        self.dy = (self.bounds[3] - self.bounds[1]) / self.grid_size
        for p in points:
            idx = min(int((p[0] - self.bounds[0]) / self.dx), self.grid_size - 1)
            idy = min(int((p[1] - self.bounds[1]) / self.dy), self.grid_size - 1)
            if (idx, idy) not in self.grid: self.grid[(idx, idy)] = []
            self.grid[(idx, idy)].append(p)

    def query(self, query_shape):
        found = []
        visited = []
        q_bounds = query_shape.bounds
        min_idx = max(0, int((q_bounds[0] - self.bounds[0]) / self.dx))
        max_idx = min(self.grid_size-1, int((q_bounds[2] - self.bounds[0]) / self.dx))
        min_idy = max(0, int((q_bounds[1] - self.bounds[1]) / self.dy))
        max_idy = min(self.grid_size-1, int((q_bounds[3] - self.bounds[1]) / self.dy))

        for i in range(min_idx, max_idx + 1):
            for j in range(min_idy, max_idy + 1):
                disk_manager.access_node(f"grid_{i}_{j}")
                visited.append((
                    self.bounds[0] + i*self.dx, self.bounds[1] + j*self.dy,
                    self.bounds[0] + (i+1)*self.dx, self.bounds[1] + (j+1)*self.dy
                ))
                if (i, j) in self.grid:
                    for p in self.grid[(i,j)]:
                        if query_shape.contains(Point(p[0], p[1])): found.append(p)
        return found, visited

    def get_visual_geometry(self):
        cells = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                 cells.append((
                    self.bounds[0] + i*self.dx, self.bounds[1] + j*self.dy,
                    self.bounds[0] + (i+1)*self.dx, self.bounds[1] + (j+1)*self.dy
                ))
        return cells

# --- R-Tree (STR Bulk Load) ---
class RTreeNode:
    def __init__(self, is_leaf=False):
        self.id = str(uuid.uuid4())
        self.is_leaf = is_leaf
        self.children = [] 
        self.bbox = None

    def update_bbox(self):
        if not self.children: return
        if self.is_leaf:
            xs = [p[0] for p in self.children]
            ys = [p[1] for p in self.children]
        else:
            xs = [c.bbox[0] for c in self.children] + [c.bbox[2] for c in self.children]
            ys = [c.bbox[1] for c in self.children] + [c.bbox[3] for c in self.children]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))

class RTree(SpatialIndex):
    def __init__(self, max_children=4):
        self.root = None
        self.max_children = max_children

    def build(self, points):
        if not points: return
        self.root = self._str_bulk_load(points)
    
    def _str_bulk_load(self, points):
        if len(points) <= self.max_children:
            node = RTreeNode(is_leaf=True)
            node.children = points; node.update_bbox()
            return node
        
        num_slices = int(len(points)**0.5)
        points.sort(key=lambda p: p[0]) 
        slices = np.array_split(points, num_slices)
        children = []
        for sl in slices:
            sl_list = sl.tolist() if isinstance(sl, np.ndarray) else sl
            sl_list.sort(key=lambda p: p[1]) 
            chunks = [sl_list[i:i + self.max_children] for i in range(0, len(sl_list), self.max_children)]
            for chunk in chunks:
                if not chunk: continue
                leaf = RTreeNode(is_leaf=True)
                leaf.children = chunk; leaf.update_bbox()
                children.append(leaf)
        return self._build_tree_levels(children)

    def _build_tree_levels(self, nodes):
        if len(nodes) == 1: return nodes[0]
        parents = []
        chunks = [nodes[i:i + self.max_children] for i in range(0, len(nodes), self.max_children)]
        for chunk in chunks:
            node = RTreeNode(is_leaf=False)
            node.children = chunk; node.update_bbox()
            parents.append(node)
        return self._build_tree_levels(parents)

    def query(self, query_shape):
        found = []
        visited = []
        def _search(node):
            if not node: return
            node_box = box(node.bbox[0], node.bbox[1], node.bbox[2], node.bbox[3])
            if not node_box.intersects(query_shape): return
            disk_manager.access_node(node.id)
            visited.append(node.bbox)
            if node.is_leaf:
                for p in node.children:
                    if query_shape.contains(Point(p[0], p[1])): found.append(p)
            else:
                for c in node.children: _search(c)
        if self.root: _search(self.root)
        return found, visited

    def get_visual_geometry(self):
        rects = []
        if not self.root: return []
        queue = [self.root]
        while queue:
            n = queue.pop(0)
            rects.append(n.bbox)
            if not n.is_leaf: queue.extend(n.children)
        return rects

def generate_data(n, dist_type, center):
    if dist_type == "Normal (Cluster)":
        lats = np.random.normal(center[0], 0.05, n)
        lons = np.random.normal(center[1], 0.05, n)
    else:
        lats = np.random.uniform(center[0]-0.1, center[0]+0.1, n)
        lons = np.random.uniform(center[1]-0.1, center[1]+0.1, n)
    return list(zip(lats, lons))

# ==========================================
# 3. INTERFAZ: SIDEBAR
# ==========================================

# Persistencia de Puntos
if "points" not in st.session_state: st.session_state["points"] = []
if "indexes" not in st.session_state: st.session_state["indexes"] = {}

st.sidebar.title("🛠️ Configuración")
st.sidebar.markdown("### 1. Datos")
data_source = st.sidebar.radio("Fuente", ["Generar Aleatorios", "Subir CSV"])

if data_source == "Generar Aleatorios":
    n_points = st.sidebar.slider("Cantidad de puntos", 100, 5000, 500)
    dist_type = st.sidebar.selectbox("Distribución", ["Normal (Cluster)", "Uniforme"])
    
    if st.sidebar.button("🔄 Generar Nuevos Puntos") or not st.session_state["points"]:
        st.session_state["points"] = generate_data(n_points, dist_type, (6.2442, -75.5812))
        st.session_state["indexes"] = {}
        st.success("Nuevos puntos generados.")
else:
    up_file = st.sidebar.file_uploader("CSV (cols: lat, lon)", type="csv")
    if up_file:
        file_details = {"filename": up_file.name, "size": up_file.size}
        if "last_file" not in st.session_state or st.session_state["last_file"] != file_details:
            try:
                df = pd.read_csv(up_file)
                cols = [c.lower() for c in df.columns]
                if 'lat' in cols and 'lon' in cols:
                    st.session_state["points"] = list(zip(df['lat'], df['lon']))
                    st.session_state["last_file"] = file_details
                    st.session_state["indexes"] = {}
                else:
                    st.sidebar.error("CSV debe tener 'lat' y 'lon'")
            except Exception as e: st.sidebar.error(f"Error: {e}")

points = st.session_state["points"]

st.sidebar.markdown("### 2. Estructuras")
struct_options = ["KD-Tree", "Quadtree", "Grid File", "R-Tree"]
selected_structs = st.sidebar.multiselect("Comparar", struct_options, default=["Quadtree", "R-Tree"])

st.sidebar.markdown("### 3. Simulación de Disco")
disk_manager.buffer_slots = st.sidebar.number_input("Buffer (Páginas)", 10, 500, 50)

if st.sidebar.button("🏗️ Construir Estructuras"):
    if not points:
        st.sidebar.error("Carga datos primero.")
    else:
        with st.spinner("Construyendo..."):
            st.session_state["indexes"] = {}
            if "KD-Tree" in selected_structs:
                idx = KDTree(); idx.build(points)
                st.session_state["indexes"]["KD-Tree"] = idx
            if "Quadtree" in selected_structs:
                idx = QuadTree(capacity=8); idx.build(points)
                st.session_state["indexes"]["Quadtree"] = idx
            if "Grid File" in selected_structs:
                idx = GridFile(grid_size=10); idx.build(points)
                st.session_state["indexes"]["Grid File"] = idx
            if "R-Tree" in selected_structs:
                idx = RTree(max_children=6); idx.build(points)
                st.session_state["indexes"]["R-Tree"] = idx
        st.sidebar.success("Listo.")

# ==========================================
# 4. INTERFAZ: PESTAÑAS Y MAPA
# ==========================================

st.title("🛰️ GeoIndex Explorer")
st.markdown("**Simulación Interactiva de Índices Espaciales Multidimensionales**")

# ---> AQUÍ ESTÁ LA LÍNEA QUE FALTABA <---
tab1, tab2, tab3 = st.tabs(["🗺️ Visualización & Consultas", "📊 Comparativa de Métricas", "📘 Documentación"])

# --- TAB 1: Visualización ---
with tab1:
    col_map, col_ctrl = st.columns([3, 1])
    
    with col_ctrl:
        st.subheader("Panel de Consulta")
        query_type = st.radio("Tipo de Consulta", ["Rango (Vista Actual)", "Polígono (Dibujo)"])
        active_idx_name = st.selectbox("Visualizar Estructura:", options=["Ninguna"] + list(st.session_state["indexes"].keys()))
        run_query = st.button("🚀 Ejecutar Consulta")

    with col_map:
        # Estado persistente del mapa
        if "map_center" not in st.session_state:
            st.session_state["map_center"] = [6.2442, -75.5812]
            st.session_state["map_zoom"] = 13

        m = folium.Map(location=st.session_state["map_center"], zoom_start=st.session_state["map_zoom"], tiles="CartoDB positron")
        
        # Puntos
        display_points = points[:1000] if len(points) > 1000 else points
        for lat, lon in display_points:
            folium.CircleMarker([lat, lon], radius=2, color="#3388ff", fill=True, fill_opacity=0.6).add_to(m)

        # Estructuras
        if active_idx_name and active_idx_name != "Ninguna" and active_idx_name in st.session_state["indexes"]:
            idx = st.session_state["indexes"][active_idx_name]
            visuals = idx.get_visual_geometry()
            color = {"R-Tree": "green", "Quadtree": "orange", "KD-Tree": "red", "Grid File": "purple"}.get(active_idx_name, "blue")
            
            if len(visuals) > 1500: visuals = visuals[:1500]
            for item in visuals:
                if isinstance(item, dict) and item.get("type") == "line":
                    folium.PolyLine(item["coords"], color=color, weight=2, opacity=0.7).add_to(m)
                elif isinstance(item, tuple) and len(item) == 4:
                    folium.Rectangle([[item[0], item[1]], [item[2], item[3]]], color=color, fill=False, weight=1.5).add_to(m)

        # Dibujo
        draw = folium.plugins.Draw(export=False, draw_options={'polyline':False,'circle':False,'marker':False,'circlemarker':False,'polygon':True,'rectangle':True}, edit_options={'edit':True})
        draw.add_to(m)
        
        # Render Mapa
        map_out = st_folium(m, width="100%", height=600, returned_objects=["last_active_drawing", "bounds", "zoom", "center"])

        # Guardar estado
        if map_out["center"]: st.session_state["map_center"] = [map_out["center"]["lat"], map_out["center"]["lng"]]
        if map_out["zoom"]: st.session_state["map_zoom"] = map_out["zoom"]

    # Ejecución de Consulta
    if run_query:
        if not st.session_state["indexes"]:
            st.error("⚠️ Construye estructuras primero.")
        else:
            query_geom = None
            if query_type == "Polígono (Dibujo)":
                if map_out["last_active_drawing"]:
                    coords = map_out["last_active_drawing"]["geometry"]["coordinates"][0]
                    query_geom = Polygon([(c[1], c[0]) for c in coords]) # Swap Lon,Lat -> Lat,Lon
                else: st.warning("Dibuja un polígono.")
            else:
                # Lectura segura de bounds
                default_b = {'_southWest': {'lat': 6.2, 'lng': -75.6}, '_northEast': {'lat': 6.3, 'lng': -75.5}}
                b = map_out["bounds"] if (map_out and "bounds" in map_out) else default_b
                # Asegurar que b no sea None
                if not b: b = default_b
                query_geom = box(b['_southWest']['lat'], b['_southWest']['lng'], b['_northEast']['lat'], b['_northEast']['lng'])

            if query_geom:
                st.divider()
                st.subheader("📊 Resultados")
                cols = st.columns(len(st.session_state["indexes"]))
                res_sum = []
                for i, (name, idx) in enumerate(st.session_state["indexes"].items()):
                    disk_manager.reset_stats()
                    t0 = time.time()
                    pts, vis = idx.query(query_geom)
                    dt = (time.time() - t0)*1000
                    with cols[i]:
                        st.markdown(f"**{name}**")
                        st.metric("Puntos", len(pts))
                        st.metric("Nodos", len(vis))
                        st.metric("I/O", disk_manager.disk_accesses)
                        st.caption(f"{dt:.2f} ms")
                    res_sum.append({"Estructura": name, "Tiempo (ms)": dt, "Accesos Disco": disk_manager.disk_accesses, "Nodos": len(vis)})
                st.session_state["last_results"] = res_sum

# --- TAB 2: Métricas ---
with tab2:
    st.header("Comparativa de Rendimiento")
    if "last_results" in st.session_state:
        df_res = pd.DataFrame(st.session_state["last_results"])
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.bar(df_res, x="Estructura", y="Tiempo (ms)", color="Estructura", title="Tiempo Respuesta"), use_container_width=True)
        with c2: st.plotly_chart(px.bar(df_res, x="Estructura", y="Accesos Disco", color="Estructura", title="I/O Simulado"), use_container_width=True)
        st.dataframe(df_res)
    else: st.info("Ejecuta una consulta primero.")

# --- TAB 3: Doc ---
with tab3:
    st.markdown("""
    ### Documentación Técnica
    * **KD-Tree**: Divide el espacio alternando ejes (Lat/Lon). Visualmente son líneas infinitas que cortan el plano.
    * **Quadtree**: Divide recursivamente en 4 cuadrantes. Visualmente es una rejilla jerárquica.
    * **Grid File**: Rejilla fija con celdas. Eficiente si la distribución es uniforme.
    * **R-Tree (STR)**: Agrupa objetos cercanos en rectángulos mínimos (MBR). Ideal para datos espaciales reales y clusters.
    
    **Simulación de Disco (LRU)**: Se simula un buffer limitado. Si un nodo no está en memoria, cuenta como acceso a disco (lento).
    """)