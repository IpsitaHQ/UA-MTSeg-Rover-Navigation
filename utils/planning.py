import numpy as np
import cv2
import heapq

# Prior semantic risks based on user specs (ENGINEERING PRIORS, NOT GROUND TRUTH)
SEMANTIC_RISK = {
    0: 1.00,  # Void
    1: 0.05,  # Compact
    2: 0.25,  # Grass
    3: 0.65,  # Bedrock
    4: 0.35,  # Sandy
    5: 0.50,  # Pebble/Gravel
    6: 0.90,  # Rock
    7: 0.85,  # Vegetation/Bush
}

def get_semantic_cost(semantic_pred):
    c_s = np.zeros_like(semantic_pred, dtype=float)
    for class_id, risk in SEMANTIC_RISK.items():
        c_s[semantic_pred == class_id] = risk
    return c_s

def estimate_normals(depth, K_inv=None):
    """
    Estimate surface normals from depth map.
    If camera intrinsics K are unavailable, use an approximation.
    """
    H, W = depth.shape
    
    # 1. Backproject to 3D space
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    # Use normalized image coordinates if K_inv is not provided
    x = (u - W/2) * depth / W
    y = (v - H/2) * depth / H
    z = depth
    
    # Calculate gradients of 3D points
    dz_dx = cv2.Sobel(z, cv2.CV_64F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(z, cv2.CV_64F, 0, 1, ksize=3)
    
    dx_dx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)
    dy_dy = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)
    
    # Normal vector N = dX/dx x dX/dy
    # Simplification assuming dx/dy ~ 0 and dy/dx ~ 0
    Nx = -dz_dx * dy_dy
    Ny = -dz_dy * dx_dx
    Nz = dx_dx * dy_dy
    
    norm = np.sqrt(Nx**2 + Ny**2 + Nz**2) + 1e-8
    normals = np.stack([Nx, Ny, Nz], axis=-1) / norm[..., np.newaxis]
    
    # Slope is angle between normal and vertical axis (0, 0, 1)
    # Using dot product
    slope = np.arccos(np.clip(np.abs(normals[..., 2]), -1.0, 1.0))
    return slope

def compute_roughness(depth, window_size=5):
    """
    Compute surface roughness as standard deviation of depth in a local window.
    """
    # Use generic filter to calculate local variance
    mean_depth = cv2.blur(depth, (window_size, window_size))
    mean_depth_sq = cv2.blur(depth**2, (window_size, window_size))
    variance = np.clip(mean_depth_sq - mean_depth**2, 0, None)
    roughness = np.sqrt(variance)
    return roughness

def get_geometry_cost(depth_map):
    # Depth map should be normalized for cost calculation
    slope = estimate_normals(depth_map)
    roughness = compute_roughness(depth_map, window_size=5)
    
    # Normalize to [0,1]
    norm_slope = cv2.normalize(slope, None, 0, 1, cv2.NORM_MINMAX)
    norm_roughness = cv2.normalize(roughness, None, 0, 1, cv2.NORM_MINMAX)
    
    c_g = 0.6 * norm_slope + 0.4 * norm_roughness
    return c_g

def generate_cost_map(c_s, c_g, c_u, w_s=1.0, w_g=1.0, w_u=1.0):
    cost = w_s * c_s + w_g * c_g + w_u * c_u
    # Normalize final cost map for planning
    cost = cv2.normalize(cost, None, 0, 1, cv2.NORM_MINMAX)
    return cost

def a_star_grid(cost_map, start, goal):
    """
    Simplistic A* search on a 2D grid cost map.
    cost_map: 2D numpy array [0,1]
    start, goal: tuple (y, x)
    """
    H, W = cost_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    
    g_score = {start: 0}
    
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    f_score = {start: heuristic(start, goal)}
    
    directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1] # reverse path
            
        for dy, dx in directions:
            neighbor = (current[0] + dy, current[1] + dx)
            
            if 0 <= neighbor[0] < H and 0 <= neighbor[1] < W:
                # Base step cost + terrain cost
                step_cost = np.sqrt(dx**2 + dy**2)
                terrain_penalty = cost_map[neighbor] * 10.0 # Weight the terrain risk
                
                tentative_g_score = g_score[current] + step_cost + terrain_penalty
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
                    
    return [] # No path found
