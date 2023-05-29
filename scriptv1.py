import numpy as np
import matplotlib.pyplot as plt

# Define the geometry - microfluidic channel
L = 1 # length - channel
H = 0.1 # height - channel
Nx = 100 # number of grid points x direction
Ny = 10 # number of grid points y direction
dx = L/Nx # grid spacing x direction
dy = H/Ny # grid spacing y direction
x = np.linspace(0, 2, Nx)
y = np.linspace(0, 2, Ny)
X, Y = np.meshgrid(x, y)

# Define the fluid flow equations
mu = 0.01 # viscosity -fluid
rho = 1000 # density -fluid
u = np.zeros((Ny,Nx)) # x-component -fluid velocity
v = np.zeros((Ny,Nx)) # y-component -fluid velocity
p = np.zeros((Ny,Nx)) # pressure -fluid

# Define the boundary conditions
u[0,:] = 1 # inlet velocity
u[-1,:] = 0 # outlet velocity
v[:,0] = 0 # left wall
v[:,-1] = 0 # right wall

# Define the simulation parameters
dt = 0.001 # time step size
T = 1 # total simulation time

# Implement the numerical method
for n in range(int(T/dt)): # number of time steps
    # Navier stokes finite difference method
    un = u.copy()
    vn = v.copy()
    pold = p.copy()
    
    b = np.zeros((Ny,Nx))
    b[1:-1,1:-1] = rho*(1/dt*((u[1:-1,2:]-u[1:-1,:-2])/(2*dx) + (v[2:,1:-1]-v[:-2,1:-1])/(2*dy)) - ((u[1:-1,2:]-u[1:-1,:-2])/(2*dx))**2 - 2*((u[2:,1:-1]-u[:-2,1:-1])/(2*dy)*(v[1:-1,2:]-v[1:-1,:-2])/(2*dx)) - ((v[2:,1:-1]-v[:-2,1:-1])/(2*dy))**2)

    for q in range(100):
        pn = p.copy()
        p[1:-1,1:-1] = (((pn[1:-1,2:]+pn[1:-1,:-2])*dy**2 + (pn[2:,1:-1]+pn[:-2,1:-1])*dx**2) / (2*(dx**2+dy**2)) - dx**2*dy**2/(2*(dx**2+dy**2))*b[1:-1,1:-1])
        p[:,0] = p[:,1] # left wall
        p[:,-1] = p[:,-2] # right wall
        p[0,:] = p[1,:] # top wall
        p[-1,:] = 0 # bottom wall

        u[1:-1,1:-1] = un[1:-1,1:-1] - un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,:-2]) - vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[:-2,1:-1]) - dt/(2*rho*dx)*(p[1:-1,2:]-p[1:-1,:-2]) + mu*(dt/dx**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,:-2]) + dt/dy**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[:-2,1:-1]))

        v[1:-1,1:-1] = vn[1:-1,1:-1] - un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,:-2]) - vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[:-2,1:-1]) - dt/(2*rho*dy)*(p[2:,1:-1]-p[:-2,1:-1]) + mu*(dt/dx**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,:-2]) + dt/dy**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[:-2,1:-1]))

        u[0,:] = 1 # inlet velocity
        u[-1,:] = 0 # outlet velocity
        v[:,0] = 0 # left wall
        v[:,-1] = 0 # right wall

fig = plt.figure(figsize=(11,7), dpi=100)
plt.contourf(X,Y,p,alpha=0.5,cmap='jet')
plt.colorbar()
plt.contour(X,Y,p,cmap='jet')
plt.streamplot(X,Y,u,v)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
