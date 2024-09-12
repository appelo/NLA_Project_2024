using Plots, TimerOutputs, OffsetArrays, SparseArrays
using DelimitedFiles

include("InsNLA.jl")

function main()
    to = TimerOutput()
    dense_LA_P = true
    dense_LA_UV = true
    extract_matrices = true
    plot_stride = 100
    # Number of gridpoints
    nx = 100
    ny = 100
    # Number of timesteps
    nt = 1000
    # Set up the problem
    INSp = InsNLA.INSProb(nx,ny,nt)
    # Viscosity
    INSp.nu = 0.001
    # timestep
    INSp.dt = 0.001
    println("We will compute until final time: ",INSp.dt*nt)
    # Divergence cleaning parameter
    INSp.alpha = 0.1/INSp.dt

    # Pre-allocate arrays for speed.
    # Use offset arrays to make the indexing consistent with Fortran / math.
    U = zeros(nx+3,ny+3)
    u = OffsetArray(U, -1:nx+1, -1:ny+1) # u will have axes (-1:nx+1, -1:ny+1)
    V = zeros(nx+3,ny+3)
    v = OffsetArray(V, -1:nx+1, -1:ny+1) # v will have axes (-1:nx+1, -1:ny+1)
    P = zeros(nx+3,ny+3)
    p = OffsetArray(P, -1:nx+1, -1:ny+1) # p will have axes (-1:nx+1, -1:ny+1)

    Uold = zeros(nx+3,ny+3)
    uold = OffsetArray(Uold, -1:nx+1, -1:ny+1) # u will have axes (-1:nx+1, -1:ny+1)
    Vold = zeros(nx+3,ny+3)
    vold = OffsetArray(Vold, -1:nx+1, -1:ny+1) # v will have axes (-1:nx+1, -1:ny+1)
    Pold = zeros(nx+3,ny+3)
    pold = OffsetArray(Pold, -1:nx+1, -1:ny+1) # p will have axes (-1:nx+1, -1:ny+1)

    Leu = zeros(nx+1,ny+1)
    leu = OffsetArray(Leu, 0:nx, 0:ny)
    Lev = zeros(nx+1,ny+1)
    lev = OffsetArray(Lev, 0:nx, 0:ny)

    Liu = zeros(nx+1,ny+1)
    liu = OffsetArray(Liu, 0:nx, 0:ny)
    Liv = zeros(nx+1,ny+1)
    liv = OffsetArray(Liv, 0:nx, 0:ny)

    Leuold = zeros(nx+1,ny+1)
    leuold = OffsetArray(Leuold, 0:nx, 0:ny)
    Levold = zeros(nx+1,ny+1)
    levold = OffsetArray(Levold, 0:nx, 0:ny)

    Liuold = zeros(nx+1,ny+1)
    liuold = OffsetArray(Liuold, 0:nx, 0:ny)
    Livold = zeros(nx+1,ny+1)
    livold = OffsetArray(Livold, 0:nx, 0:ny)

    # for plotting
    VORZ = zeros(nx+1,ny+1)
    vorz = OffsetArray(VORZ, 0:nx, 0:ny)

    # sizes of different matrices
    sys_size_p = (nx+1)*(ny+1)
    sys_size_pbig = sys_size_p + 1
    sys_size_uv = (nx-1)*(ny-1)

    println("Setting up Laplacians")
    if extract_matrices
        # setup pressure laplacian
        LapP = zeros(sys_size_p,sys_size_p)
        LapUV = zeros(sys_size_uv,sys_size_uv)
        !
        INSp.p .= 0.0
        lpvec = zeros(sys_size_p)
        @timeit to "Setup Laplacians" for i = 1:sys_size_p
            INSp.p[i] = 1.0
            InsNLA.apply_pressure_laplacian!(lpvec,INSp)
            INSp.p[i] = 0.0
            LapP[:,i] = lpvec
        end
        # setup uv laplacian
        INSp.u .= 0.0
        luvvec = zeros(sys_size_uv)
        @timeit to "Setup Laplacians" for i = 1:sys_size_uv
            INSp.u[i] = 1.0
            InsNLA.apply_velocity_laplacian!(luvvec,INSp)
            INSp.u[i] = 0.0
            LapUV[:,i] .= -0.5*INSp.nu*luvvec
            LapUV[i,i] = LapUV[i,i] + 1.0/INSp.dt
        end
        LapUV = sparse(LapUV)
        droptol!(LapUV,1e-12)
        LapPBig = ones(sys_size_p+1,sys_size_p+1)
        LapPBig[sys_size_p+1,sys_size_p+1] = 0.0
        LapPBig[1:sys_size_p,1:sys_size_p] = LapP
        LapPBig = sparse(LapPBig)
        LapP = sparse(LapP)
        droptol!(LapPBig,1e-12)
    else
        # The matrices could be set-up using spdiagm and kron
        # 
    end
    println("Done setting up Laplacians")
    
    initial_data = 3
    if initial_data == 1
        # Initial data for a lid-driven cavity flow
        INSp.guy[:,1] .= 0.0
        INSp.guy[:,2] .= 1.0
        INSp.gvx[:,1] .= 0.0
        INSp.gvx[:,2] .= 0.0
        INSp.u .= 0.0
        INSp.v .= 0.0
        @timeit to "Swap" for j = -1:ny+1
            for i = -1:nx+1
                u[i,j] = 0.0
                v[i,j] = 0.0
            end
        end
    elseif initial_data == 2
        # A single vortex
        INSp.guy .= 0.0
        INSp.gvx .= 0.0
        @timeit to "Swap" for j = 1:ny-1
            for i = 1:nx-1
                utmp,vtmp = InsNLA.taylor_vortex(INSp.x[1+i],INSp.y[1+j],0.5,0.5,0.1,1.0)
                INSp.u[i+(j-1)*(nx-1)] = utmp
                INSp.v[i+(j-1)*(nx-1)] = vtmp
                u[i,j] = utmp
                v[i,j] = vtmp
            end
        end
    elseif initial_data == 3
        # Initial data for a lid-driven cavity flow
        INSp.guy .= 0.0
        INSp.gvx .= 0.0
        @timeit to "Swap" for j = 1:ny-1
            for i = 1:nx-1
                utmp1,vtmp1 = InsNLA.taylor_vortex(INSp.x[1+i],INSp.y[1+j],0.4,0.5,0.1,10.0)
                utmp2,vtmp2 = InsNLA.taylor_vortex(INSp.x[1+i],INSp.y[1+j],0.6,0.5,0.1,10.0)
                utmp3,vtmp3 = InsNLA.taylor_vortex(INSp.x[1+i],INSp.y[1+j],0.5,0.4,0.1,10.0)
                utmp4,vtmp4 = InsNLA.taylor_vortex(INSp.x[1+i],INSp.y[1+j],0.5,0.6,0.1,10.0)
                u[i,j] = utmp1+utmp2+utmp3+utmp4
                v[i,j] = vtmp1+vtmp2+vtmp3+vtmp4
            end
        end

    end
    # start up the computation with a single forward Euler step (backwards in time).
    # We need to find the current pressure to compute the advection term.
    @timeit to "updateBCforU" InsNLA.updateBCforU!(u,v,INSp)
    @timeit to "computeAndUpdateGPforU" InsNLA.computeAndUpdateGPforU!(u,v,INSp)
    @timeit to "computeGPforP" InsNLA.computeGPforP!(u,v,INSp)
    @timeit to "setupRhsideP" InsNLA.setupRhsideP!(u,v,INSp)

    if dense_LA_P
        @timeit to "P solve" INSp.pvecBig = LapPBig\INSp.bPBig
        @timeit to "Swap" for j = 0:ny
            for i = 0:nx
                p[i,j] = INSp.pvecBig[1+i+(nx+1)*j]
            end
        end
    else
        # Your code goes here
    end
    #
    @timeit to "computeLE" InsNLA.computeLE!(leu,lev,u,v,p,INSp)
    @timeit to "computeLI" InsNLA.computeLI!(liu,liv,u,v,INSp)
    #
    @timeit to "Swap" for j = 0:ny
        for i = 0:nx
            uold[i,j] = u[i,j] - INSp.dt*(leu[i,j] + liu[i,j])
            vold[i,j] = v[i,j] - INSp.dt*(lev[i,j] + liv[i,j])
        end
    end
    @timeit to "updateBCforU" InsNLA.updateBCforU!(uold,vold,INSp)
    @timeit to "computeAndUpdateGPforU" InsNLA.computeAndUpdateGPforU!(uold,vold,INSp)
    @timeit to "computeGPforP" InsNLA.computeGPforP!(uold,vold,INSp)
    @timeit to "setupRhsideP" InsNLA.setupRhsideP!(uold,vold,INSp)
    #
    if dense_LA_P
        @timeit to "P solve" INSp.pvecBig = LapPBig\INSp.bPBig
        @timeit to "Swap" for j = 0:ny
            for i = 0:nx
                pold[i,j] = INSp.pvecBig[1+i+(nx+1)*j]
            end
        end
    else
        # Your code goes here
    end
    @timeit to "computeLE" InsNLA.computeLE!(leuold,levold,uold,vold,pold,INSp)
    @timeit to "computeLI" InsNLA.computeLI!(liuold,livold,uold,vold,INSp)
    for it = 1:nt
        # semi-implicit method
        # Get boundary conditions at time n+1
        # Boundary conditions are constant in
        # time in current implementation so no update
        @timeit to "setupRhsideUV"  InsNLA.setupRhsideUV!(leu,lev,leuold,levold,liu,liv,u,v,INSp)
        # solve for new u and v
        if dense_LA_UV
            @timeit to "UV solve" INSp.u = LapUV\INSp.bU
            @timeit to "UV solve" INSp.v = LapUV\INSp.bV
        else
            # !!! YOUR CODE GOES HERE !!!!
        end
        # Swap long vector into 2D array
        @timeit to "Swap" for j = 1:ny-1
            for i = 1:nx-1
                u[i,j] = INSp.u[i+(j-1)*(nx-1)]
                v[i,j] = INSp.v[i+(j-1)*(nx-1)]
            end
        end
        
        @timeit to "updateBCforU" InsNLA.updateBCforU!(u,v,INSp)
        @timeit to "computeAndUpdateGPforU" InsNLA.computeAndUpdateGPforU!(u,v,INSp)
        @timeit to "computeGPforP" InsNLA.computeGPforP!(u,v,INSp)
        @timeit to "setupRhsideP" InsNLA.setupRhsideP!(u,v,INSp)
        if dense_LA_P
            @timeit to "P solve" INSp.pvecBig = LapPBig\INSp.bPBig
            for j = 0:ny
                for i = 0:nx
                    p[i,j] = INSp.pvecBig[1+i+(nx+1)*j]
                end
            end
        else
            # Your code goes here
        end
        # Swap
        @timeit to "Swap" for j = 0:ny
            for i = 0:nx
                leuold[i,j] = leu[i,j]
                levold[i,j] = lev[i,j]
            end
        end
        # Compute new "operators"
        @timeit to "computeLE" InsNLA.computeLE!(leu,lev,u,v,p,INSp)
        @timeit to "computeLI" InsNLA.computeLI!(liu,liv,u,v,INSp)
        # Check for blowup
        vel = sqrt.(u.^2 .+ v.^2)
        if maximum(vel) > 100
            println("Max: ",maximum(vel)," : ",nt)
            return nothing
        end
        if mod(it,plot_stride) == 0
            @timeit to "plot" for j = 0:ny
                for i = 0:nx
                    vorz[i,j] = -((v[i+1,j]-v[i-1,j])/(2*INSp.hx) - (u[i,j+1]-u[i,j-1])/(2*INSp.hy))
                end
            end
            if initial_data == 1 
                pl = contour(INSp.x,INSp.y,transpose(VORZ),
                             aspect_ratio = :equal,
                             color=:turbo,
                             levels = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6],
                             clims=(-6, 6),
                             xlim =(0,1),
                             ylim =(0,1),
                             lw=2)
            else
                pl = contour(INSp.x,INSp.y,transpose(VORZ),
                             aspect_ratio = :equal,
                             color=:turbo,
                             xlim =(0,1),
                             ylim =(0,1),
                             lw=2)
            end
            display(pl)
                println("Plotting at timestep ",it)
            sleep(0.01)
        end
    end
    show(to)
    return nothing
end
