import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 2048
steps = 1500
gravity = 3.8
target = [0.8, 0.2]

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss_actuation = scalar()
total_loss = scalar()
rect_param_loss = scalar()
rect_param_overlap_loss = scalar() # Need some separate loss variables for debugging
rect_param_disconnect_loss = scalar()
rect_connected = ti.field(dtype=ti.i32)

n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()
actuation_omega = 20
act_strength = 4

# Rectangle variables
rect_params = ti.Vector.field(4, dtype=real)  # [x, y, width, height]
rect_actuation = ti.field(dtype=ti.i32)       # actuation ID for each rectangle

# Number of rectangles (doesn't need to be placed in a field, will be declared globally)
n_rectangles = 0
# Set maximum number of rectangles
max_rectangles = 10

particle_counter = scalar()
offset_x = scalar()
offset_y = scalar()

debug_field = ti.Vector.field(4, dtype=real)  # Store rectangle params

# Need to store this to maintain the same number of particles across iterations
initial_rect_dims = ti.Vector.field(2, dtype=real)  # [width, height] for each rectangle

def allocate_fields():
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)

    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss_actuation, x_avg)
    
    # New parameter allocation
    # Allocate space for rectangle parameters
    ti.root.dense(ti.i, max_rectangles).place(rect_params)  # [x, y, width, height]
    ti.root.dense(ti.i, max_rectangles).place(rect_actuation)  # actuation ID for each rectangle
    
    ti.root.place(rect_param_loss, rect_param_overlap_loss, rect_param_disconnect_loss) 
    ti.root.place(total_loss)
    ti.root.dense(ti.i, max_rectangles).place(rect_connected)
    
    ti.root.place(particle_counter, offset_x, offset_y)
    ti.root.dense(ti.i, n_rectangles).place(debug_field)
    ti.root.dense(ti.i, max_rectangles).place(initial_rect_dims)
    
    ti.root.lazy_grad()

@ti.kernel
def reset_simulation():
    """Reset particle state to initial configuration without Scene parameter"""
    for i in range(n_particles):
        x[0, i] = [0.0, 0.0]  # Reset positions to origin
        v[0, i] = ti.Vector([0.0, 0.0])  # Reset velocities
        F[0, i] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])  # Reset deformation gradient
        C[0, i] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])  # Reset affine velocity

@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0

@ti.kernel
def clear_rect_params_grad():
    for i in range(n_rectangles):
        rect_params.grad[i] = [0.0, 0.0, 0.0, 0.0]

@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]

        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 0.5


@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        # Clamp particle positions to valid range
        x[f + 1, p] = ti.Vector([
            ti.min(ti.max(x[f, p][0] + dt * v[f + 1, p][0], 3*dx), 1.0 - 3*dx),
            x[f + 1, p][1]
        ])
        C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            act += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                          2 * math.pi / n_sin_waves * j)
        act += bias[i]
        actuation[t, i] = ti.tanh(act) 

# This had to be a NON-kernel function since we have to update all the particles in SEQUENTIAL order
# We cannot do so in parallel since we need to update the particle_counter field
@ti.ad.grad_replaced
def update_particles_from_rect_params(s: ti.i32):
    """Sequential Python function that calls parallel kernel for each rectangle"""
    for rect_idx in range(n_rectangles):
        
        params = rect_params[rect_idx]
        # x_pos, y_pos, w, h = params[0], params[1], params[2], params[3]
        
        x_pos, y_pos = params[0], params[1]  # Only use position from current params
        w, h = initial_rect_dims[rect_idx][0], initial_rect_dims[rect_idx][1]  # Use initial dimensions
        
        # Store in debug field
        debug_field[rect_idx] = params
        
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        
        for i in range(w_count):
            for j in range(h_count):
                particle_idx = int(particle_counter[None])
                particle_counter[None] += 1
                x[s, particle_idx] = [
                    x_pos + (i + 0.5) * real_dx + offset_x[None],
                    y_pos + (j + 0.5) * real_dy + offset_y[None]
                ]

@ti.ad.grad_for(update_particles_from_rect_params)
def update_particles_from_rect_params_grad(s: ti.i32):
    """Gradient function to update rect_params based on particle gradients"""
    particle_idx = 0
                
    for rect_idx in range(n_rectangles):
        w = initial_rect_dims[rect_idx][0]
        h = initial_rect_dims[rect_idx][1]
        n_particles = int(w/dx) * 2 * int(h/dx) * 2
        
        # Sum gradients from all particles in this rectangle
        for p in range(particle_idx, particle_idx + n_particles):
            # Particle gradients affect rectangle position (only gradients for x[s] are populated)
            rect_params.grad[rect_idx][0] += x.grad[s, p][0]
            rect_params.grad[rect_idx][1] += x.grad[s, p][1]
        
        particle_idx += n_particles
        
@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])


@ti.kernel
def compute_actuation_loss():
    dist = x_avg[None][0]
    # loss_actuation[None] = -dist
    # # Normalize to [0,1] while preserving direction
    loss_actuation[None] = 1.0 - ti.min(1.0, dist)  # 0 is best (far right), 1 is worst (far left)


@ti.ad.grad_replaced
def compute_rect_param_loss():
    for i in range(n_rectangles):
        params = rect_params[i]
        rect_connected[i] = i # Intially, assume each rectangle is connected to itself
        
        w_i = initial_rect_dims[i][0]
        h_i = initial_rect_dims[i][1]  
        
        # Check connectivity with all other rectangles
        for j in range(n_rectangles):
            if i != j:  # Don't check against self
                params_j = rect_params[j]
                
                w_j = initial_rect_dims[j][0]
                h_j = initial_rect_dims[j][1]
                
                # Calculate centers
                center_i_x = params[0] + w_i / 2.0
                center_i_y = params[1] + h_i / 2.0
                center_j_x = params_j[0] + w_j / 2.0
                center_j_y = params_j[1] + h_j / 2.0
                
                # Calculate distances between centers
                distance_x = ti.abs(center_i_x - center_j_x)
                distance_y = ti.abs(center_i_y - center_j_y)
                
                # Check if rectangles are touching
                # Use small overlap (0.001) to consider them "touching"
                touching = (distance_x <= (w_i + w_j) / 2.0 + 0.001) and \
                           (distance_y <= (h_i + h_j) / 2.0 + 0.001)
                
                if touching:
                    # Mark rectangles as connected
                    min_idx = ti.min(rect_connected[i], rect_connected[j])
                    rect_connected[i] = min_idx
                    rect_connected[j] = min_idx
                    
                # Check for overlap and penalize if exists
                overlap_x = ti.max(0.0, (w_i + w_j) / 2.0 - distance_x)
                overlap_y = ti.max(0.0, (h_i + h_j) / 2.0 - distance_y)
                if overlap_x > 0.0 and overlap_y > 0.0:
                    overlap_area = overlap_x * overlap_y
                    # normalized_overlap = overlap_area / (w_i * h_i) # Normalize by area of rectangle so its scale-invariant
                    rect_param_loss[None] += overlap_area
                    rect_param_overlap_loss[None] += overlap_area
        
        # Second pass: check if all rectangles are in same connected component
        for rect_idx in range(n_rectangles):
            if rect_connected[rect_idx] != rect_connected[0]: # Check to see if every component is connected to the first component
                # Add large penalty for disconnected structure
                rect_param_loss[None] += 0.1
                rect_param_disconnect_loss[None] += 0.1
    
@ti.ad.grad_for(compute_rect_param_loss)
def compute_rect_param_loss_grad():
    """Custom gradient computation combining particle and constraint gradients"""
    # Initialize gradients
    overlap_grad_scale = 1
    disconnect_grad_scale = 1
    
    for i in range(n_rectangles):
        rect_connected[i]
                
    for i in range(n_rectangles):
        params = rect_params[i]
        rect_connected[i] = i # Intially, assume each rectangle is connected to itself
        
        w_i = initial_rect_dims[i][0]
        h_i = initial_rect_dims[i][1]  
        
        # Check connectivity with all other rectangles
        for j in range(n_rectangles):
            if i != j:  # Don't check against self
                params_j = rect_params[j]
                
                w_j = initial_rect_dims[j][0]
                h_j = initial_rect_dims[j][1]
                
                # Calculate centers
                center_i_x = params[0] + w_i / 2.0
                center_i_y = params[1] + h_i / 2.0
                center_j_x = params_j[0] + w_j / 2.0
                center_j_y = params_j[1] + h_j / 2.0
                
                # Calculate distances between centers
                distance_x = ti.abs(center_i_x - center_j_x)
                distance_y = ti.abs(center_i_y - center_j_y)
                
                # Check if rectangles are touching
                # Use small overlap (0.001) to consider them "touching"
                touching = (distance_x <= (w_i + w_j) / 2.0 + 0.001) and \
                           (distance_y <= (h_i + h_j) / 2.0 + 0.001)
                
                if touching:
                    # Mark rectangles as connected
                    min_idx = ti.min(rect_connected[i], rect_connected[j])
                    rect_connected[i] = min_idx
                    rect_connected[j] = min_idx
                
                # Check for overlap and penalize if exists
                overlap_x = ti.max(0.0, (w_i + w_j) / 2.0 - distance_x)
                overlap_y = ti.max(0.0, (h_i + h_j) / 2.0 - distance_y)
                
                # # print out center xs and distance_x
                # print("Center i x: ", center_i_x, " Center j x: ", center_j_x, " distance_x: ", distance_x)
                    
                # There is only overlap if both overlap_x and overlap_y are positive
                if overlap_x > 0.0 and overlap_y > 0.0:
                    if overlap_x > 0.0:
                        # print("Overlap in x direction: ", "rectangle ", i, " and rectangle ", j, "overlap_x: ", overlap_x)
                        # Move to the right if we are to the left of the other rectangle (would increase overlap)
                        grad_dir_x = 1.0 if center_i_x < center_j_x else -1.0 
                        # Gradient magniute (scaled by overlap_grad_scale)
                        grad_x_magnitude = overlap_x * overlap_grad_scale
                        # Update gradient for x position
                        rect_params.grad[i][0] += grad_dir_x * grad_x_magnitude
                        # print("Adding to rect_params.grad[", i, "][0]: ", grad_dir_x * grad_x_magnitude)
                    if overlap_y > 0.0:
                        # print("Overlap in y direction", "rectangle ", i, " and rectangle ", j, "overlap_y: ", overlap_y)
                        grad_dir_y = 1.0 if center_i_y < center_j_y else -1.0
                        grad_y_magnitude = overlap_y * overlap_grad_scale
                        rect_params.grad[i][1] += grad_dir_y * grad_y_magnitude
                        # print("Adding to rect_params.grad[", i, "][1]: ", grad_dir_y * grad_y_magnitude)
                
                    
        # Second pass: check if all rectangles are in same connected component
        # This is hard to optimize so to simplify we try to move all rectangles to the first component
        for rect_idx in range(n_rectangles):
            if rect_connected[rect_idx] != rect_connected[0]:
                # Direction of gradient increases loss (which is when we move away from the first connected component)
                grad_dir_x = -1.0 if rect_params[rect_idx][0] < rect_params[0][0] else 1.0
                grad_dir_y = -1.0 if rect_params[rect_idx][1] < rect_params[0][1] else 1.0
                # Update gradients
                rect_params.grad[rect_idx][0] += grad_dir_x * disconnect_grad_scale
                rect_params.grad[rect_idx][1] += grad_dir_y * disconnect_grad_scale
        
        
@ti.kernel
def compute_total_loss():
    # Weight the losses differently
    # Actuation loss (moving right) is primary objective
    # Rectangle constraints are secondary objectives
    movement_weight = 1.0
    constraint_weight = 1.0  
    
    total_loss[None] = movement_weight * loss_actuation[None] + \
                       constraint_weight * rect_param_loss[None]
    
def forward(total_steps=steps):
    
    # Update particles from new rectangle parameters (updated at the end of the main loop in previous iteration)
    # particle_counter[None] = 0
    # print("Now updating particles from rectangle parameters...")
    # update_particles_from_rect_params(0)
    # print("Particle counter:", particle_counter[None])
    
    # Simulation
    for s in range(total_steps - 1):
        advance(s)

    # Computing the actuation loss
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_actuation_loss()
    
    # Computing the particle loss
    rect_param_loss[None] = 0
    rect_param_overlap_loss[None] = 0
    rect_param_disconnect_loss[None] = 0
    for i in range(n_rectangles):
        rect_connected[i] = i
    compute_rect_param_loss()
    
    print("Actuation loss:", loss_actuation[None])
    print("Rectangle parameter loss:", rect_param_loss[None])
    print("Rectangle overlap loss:", rect_param_overlap_loss[None])
    print("Rectangle disconnect loss:", rect_param_disconnect_loss[None])

    # Computing the total loss
    compute_total_loss() 

@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(advance)
def advance_grad(s):
    # print("Beginning advance_grad function")
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)
    # print("Ending advance_grad function")

class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act
    
    def set_n_rectangles(self, n_legs):
        global n_rectangles
        n_rectangles = n_legs + 1
        
# def fish(scene):
#     scene.add_rect(0.025, 0.025, 0.95, 0.1, -1, ptype=0)
#     scene.add_rect(0.1, 0.2, 0.15, 0.05, -1)
#     scene.add_rect(0.1, 0.15, 0.025, 0.05, 0)
#     scene.add_rect(0.125, 0.15, 0.025, 0.05, 1)
#     scene.add_rect(0.2, 0.15, 0.025, 0.05, 2)
#     scene.add_rect(0.225, 0.15, 0.025, 0.05, 3)
#     scene.set_n_actuators(4)


# def robot(scene):
#     scene.set_offset(0.1, 0.03)
#     scene.add_rect(0.0, 0.1, 0.3, 0.1, -1)
#     scene.add_rect(0.0, 0.0, 0.05, 0.1, 0)
#     scene.add_rect(0.05, 0.0, 0.05, 0.1, 1)
#     scene.add_rect(0.2, 0.0, 0.05, 0.1, 2)
#     scene.add_rect(0.25, 0.0, 0.05, 0.1, 3)
#     scene.set_n_actuators(4)

# Parameterized function to inialize rectangles based on user inputted number of legs
def intialize_rectangles(scene,num_legs=4):
    
    # Store intial values in Python lists
    intial_params = []
    intial_actuation = []
    
    scene.set_offset(0.1, 0.2)
    
    # Add body rectangle
    scene.add_rect(0.0, 0.1, 0.1*num_legs, 0.1, 0)
    intial_params.append([0.0, 0.1, 0.1*num_legs, 0.1])
    intial_actuation.append(0)
    
    # Loop to add leg rectangles
    for i in range(num_legs):
        # Add leg rectangles with a different actuation ID each
        scene.add_rect(0.1*i, 0.0, 0.05, 0.1, i+1)
        intial_params.append([0.1*i, 0.1, 0.05, 0.1])
        intial_actuation.append(i+1)
        
    # Set actuator count to number of legs + body
    scene.set_n_actuators(num_legs + 1)
    
    # Set number of rectangles to number of legs + body
    scene.set_n_rectangles(num_legs)
    
    scene.finalize()
    allocate_fields()
    
    # Set the rectangle parameters to the intial values
    for i in range(len(intial_params)):
        rect_params[i] = intial_params[i]
        rect_actuation[i] = intial_actuation[i]
        initial_rect_dims[i] = [intial_params[i][2], intial_params[i][3]]  # Store width and height


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def visualize(s, folder):
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')

def print_rect_params():
    """Print current parameters for all rectangles"""
    print("-" * 50)
    print(f"{'Rectangle':<10} {'X':^10} {'Y':^10} {'Width':^10} {'Height':^10}")
    print("-" * 50)
    for i in range(n_rectangles):
        params = rect_params[i].to_numpy()
        print(f"{i:<10} {params[0]:^10.3f} {params[1]:^10.3f} {params[2]:^10.3f} {params[3]:^10.3f}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    # Add in input parameter for number of legs
    parser.add_argument('--num_legs', type=int, default=4)
    options = parser.parse_args()

    # initialization
    scene = Scene()
    # Intializes rectangles based input parameters and adds them to the scene
    # Will also finalize the scene and allocate fields
    intialize_rectangles(scene,options.num_legs)
    
    # print("Initial rectangle parameters:")
    # print_rect_params()  

    for i in range(n_actuators):
        for j in range(n_sin_waves):
            weights[i, j] = np.random.randn() * 0.01

    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]
        
    offset_x[None] = scene.offset_x
    offset_y[None] = scene.offset123_y

    total_losses = []
    actuation_losses = []
    rect_param_losses = []
    
    for iter in range(options.iters):
        
        # Visualize the first iteration before optimization
        if iter == 0:
            forward(steps)
            print("Visualizing initial result before any optimizations...")
            for s in range(15, steps, 16):
                visualize(s, 'diffmpm/iter{:03d}/'.format(iter))
        
        # New rect_params after optimization
        print("\nIteration", iter)
        print_rect_params()
        
        # Clear gradients per iteration (to avoid exploding gradients)
        clear_particle_grad()
        clear_actuation_grad()
        clear_rect_params_grad()
        
        # Reset simulation 
        reset_simulation()
                
        # with open("particle_locations.txt", "w") as f:
        #     # print out all the particle positions
        #     for i in range(int(particle_counter[None])):
        #         f.write(f"Particle {i}: {x[0,i].to_numpy()}\n")
        
        with ti.ad.Tape(total_loss):
            particle_counter[None] = 0
            print("Now updating particles from rectangle parameters...")
            update_particles_from_rect_params(0)
            print("Particle counter:", particle_counter[None])
            # Run forward simulation
            forward(steps)
            
        l = total_loss[None]
        r_l = rect_param_loss[None]
        a_l = loss_actuation[None]
        total_losses.append(l)
        actuation_losses.append(a_l)
        rect_param_losses.append(r_l)
        print('iter=', iter, 'rect_param_loss=', r_l, 'actuation_loss=', a_l, 'total_loss=', l)
        
        # Learning rates for actuation and particle loss
        actuation_lr = 0.1
        rect_param_lr = 0.1

        # Update actuation parameters
        for i in range(n_actuators):
            for j in range(n_sin_waves):
                # print(weights.grad[i, j])
                weights[i, j] -= actuation_lr * weights.grad[i, j]
            bias[i] -= actuation_lr * bias.grad[i]
            
        # Update rectangle parameters 
        for i in range(n_rectangles):
            # Only update x,y positions, ignore width/height gradients
            # Print out the gradients for debugging
            print(rect_params.grad[i])
            
            w = initial_rect_dims[i][0]  # width
            h = initial_rect_dims[i][1]  # height
            
            # Add offsets before applying updates
            curr_x = rect_params[i][0] + offset_x[None]
            curr_y = rect_params[i][1] + offset_y[None]
            
            # Calculate new positions without offsets
            new_x = curr_x - rect_param_lr * rect_params.grad[i][0]
            new_y = curr_y - rect_param_lr * rect_params.grad[i][1]
            
            # Clamp positions (in the space without offsets)
            clamped_x = max(3*dx, min(1.0 - 3*dx - w, new_x))
            clamped_y = max(3*dx, min(1.0 - 3*dx - h, new_y))
            
            # Remove offsets back before storing
            rect_params[i][0] = clamped_x - offset_x[None]
            rect_params[i][1] = clamped_y - offset_y[None]
           
    
    # Last visualization at the end of all training iterations
    forward(steps)
    print("Visualizing final result...")
    for s in range(15, steps, 16):
        visualize(s, 'diffmpm/iter{:03d}/'.format(iter))
        
    print("\nFinal rectangle parameters:")
    print_rect_params()
    
    # ti.profiler_print()
    plt.title("Optimization of Movement to the Right")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(total_losses, label="Total Loss")
    plt.plot(actuation_losses, label="Actuation Loss")  
    plt.plot(rect_param_losses, label="Rectangle Parameter Loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
