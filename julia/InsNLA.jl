module InsNLA
using Plots, OffsetArrays, OffsetArrays

export INSProb, apply_pressure_laplacian!, apply_velocity_laplacian!,
    updateBCforU!, computeAndUpdateGPforU!, computeGPforP!, setupRhsideP!,
    computeLE!, computeLI!, setupRhsideUV!, taylor_vortex

macro TTime(T, vars...)
    expr = Expr(:block)
    append!(expr.args, map(var -> :($var::$T), vars))
    return esc(expr)
end

mutable struct INSProb
    nx::Int64
    ny::Int64
    nsteps::Int64
    Lx::Float64
    Ly::Float64
    tfinal::Float64
    nu::Float64
    alpha::Float64
    nuv::Int64
    np::Int64
    hx::Float64
    hy::Float64
    dt::Float64
    @TTime Vector{Float64} x y u v uold vold p pold bP bPBig pvecBig bU bV
    @TTime Vector{Float64} leu lev liu liv leuold levold liuold livold
    @TTime Matrix{Float64} gux guy gvx gvy pbx pby
    function INSProb(
        nx::Int64,
        ny::Int64,
        nsteps::Int64;
        Lx::Float64 = 1.0,
        Ly::Float64 = 1.0,
        tfinal::Float64 = 1.0,
        nu::Float64 = 10.0,
        alpha::Float64 = 1.0e-3,
        )
        nuv = (nx-1)*(ny-1)
        np = (nx+1)*(ny+1)
        x = collect(LinRange(0.0,Lx,nx+1))
        y = collect(LinRange(0.0,Ly,ny+1))
        hx = x[2]-x[1]
        hy = y[2]-y[1]
        u = zeros(nuv)
        v = zeros(nuv)
        uold = zeros(nuv)
        vold = zeros(nuv)
        p = zeros(np)
        pold = zeros(np)
        bP = zeros(np)
        bPBig = zeros(np+1)
        pvecBig = zeros(np+1)
        bU = zeros(nuv)
        bV = zeros(nuv)
        dt = tfinal / nsteps
        leu = zeros(nuv)
        lev = zeros(nuv)
        liu = zeros(nuv)
        liv = zeros(nuv)
        leuold = zeros(nuv)
        levold = zeros(nuv)
        liuold = zeros(nuv)
        livold = zeros(nuv)
        # These are the arrays for the boundary forcing u=gu, v=gv on the
        # boundary. We order them
        # gux[:,1] = "left"
        # gux[:,2] = "right"
        # guy[:,1] = "bottom"
        # guy[:,2] = "top"
        gux = zeros(ny+1,2)
        guy = zeros(nx+1,2)
        gvx = zeros(ny+1,2)
        gvy = zeros(nx+1,2)
        pbx = zeros(ny+1,2)
        pby = zeros(nx+1,2)
        new(
            nx,ny,nsteps,Lx,Ly,tfinal,nu,alpha,
            nuv,np,hx,hy,dt,x,y,u,v,uold,vold,p,pold,
            bP,bPBig,pvecBig,bU,bV,
            leu, lev, liu, liv, leuold, levold, liuold, livold,
            gux, guy, gvx, gvy, pbx, pby
        )
    end
end

function apply_pressure_laplacian!(lp::Vector{Float64},prob::INSProb)

    hxi2 = 1.0/prob.hx^2
    hyi2 = 1.0/prob.hy^2

    mx = (prob.nx+1)
    my = (prob.ny+1)
    for j=1:my
        for i=1:mx
            k = i+(j-1)*mx
            kp = k+1
            km = k-1
            kb = k-mx
            kt = k+mx
            lp[k] = -2.0*(hxi2+hyi2)*prob.p[k]
            if i < mx
                if i==1
                    lp[k] = lp[k] + 2.0*hxi2*prob.p[kp]
                else
                    lp[k] = lp[k] + hxi2*prob.p[kp]
                end
            end
            if  i > 1
                if i==mx
                    lp[k] = lp[k] + 2.0*hxi2*prob.p[km]
                else
                    lp[k] = lp[k] + hxi2*prob.p[km]
                end
            end
            if j < my
                if j == 1
                    lp[k] = lp[k] + 2.0*hyi2*prob.p[kt]
                else
                    lp[k] = lp[k] + hyi2*prob.p[kt]
                end
            end
            if  j > 1
                if  j == my
                    lp[k] = lp[k] + 2.0*hyi2*prob.p[kb]
                else
                    lp[k] = lp[k] + hyi2*prob.p[kb]
                end
            end
        end
    end
end


function apply_velocity_laplacian!(lu::Vector{Float64},prob::INSProb)

    hxi2 = 1.0/prob.hx^2
    hyi2 = 1.0/prob.hy^2

    mx = (prob.nx-1)
    my = (prob.ny-1)
    for j=1:my
        for i=1:mx
            k = i+(j-1)*mx
            kp = k+1
            km = k-1
            kb = k-mx
            kt = k+mx
            lu[k] = -2.0*(hxi2+hyi2)*prob.u[k]
            if  i < mx
                lu[k] = lu[k] + hxi2*prob.u[kp]
            end
            if i > 1
                lu[k] = lu[k] + hxi2*prob.u[km]
            end
            if  j < my
                lu[k] = lu[k] + hyi2*prob.u[kt]
            end
            if j > 1
                lu[k] = lu[k] + hyi2*prob.u[kb]
            end
        end
    end
end

function updateBCforU!(u,v,prob::INSProb)
    # We update u and v to the left and right.
    nx = prob.nx
    ny = prob.nx
    # Left and right.
    for j = 1:ny-1
        u[0,j]  = prob.gux[1+j,1]
        u[nx,j] = prob.gux[1+j,2]
        v[0,j]  = prob.gvx[1+j,1]
        v[nx,j] = prob.gvx[1+j,2]
    end
    for i = 1:nx-1
        # Then we do the top and bottom.
        u[i,0]  = prob.guy[1+i,1]
        u[i,ny] = prob.guy[1+i,2]
        v[i,0]  = prob.gvy[1+i,1]
        v[i,ny] = prob.gvy[1+i,2]
    end
    # Take averages in the corners
    u[0,0]   = 0.5*(prob.gux[1,1] + prob.guy[1,1])
    u[0,ny]  = 0.5*(prob.gux[1+ny,1] + prob.guy[1,2])
    u[nx,0]  = 0.5*(prob.gux[1,2] + prob.guy[1+nx,1])
    u[nx,ny]  = 0.5*(prob.gux[1+ny,2] + prob.guy[1+nx,2])
    v[0,0]   = 0.5*(prob.gvx[1,1] + prob.gvy[1,1])
    v[0,ny]  = 0.5*(prob.gvx[1+ny,2] + prob.gvy[1,2])
    v[nx,0]  = 0.5*(prob.gvx[1,2] + prob.gvy[1+nx,1])
    v[nx,ny]  = 0.5*(prob.gvx[1+ny,2] + prob.gvy[1+nx,2])
end


function computeAndUpdateGPforU!(u,v,prob::INSProb)
    # This routine assumes the interior point and the physical boundary
    # conditions have been updated.

    nx = prob.nx
    ny = prob.ny
    hx = prob.hx
    hy = prob.hy

    Ubx = zeros(ny+1,2)
    ubx = OffsetArray(Ubx, 0:ny, 1:2) # u will have axes 
    Vbx = zeros(ny+1,2)
    vbx = OffsetArray(Vbx, 0:ny, 1:2) # u will have axes 

    Uby = zeros(nx+1,2)
    uby = OffsetArray(Uby, 0:nx, 1:2) # u will have axes 
    Vby = zeros(ny+1,2)
    vby = OffsetArray(Vby, 0:nx, 1:2) # u will have axes 
    
    # We extrapolate v to the left and right
    # We extrapolate u to the bottom and top
    # to update ghost point values
    for j = 0:ny
        vbx[j,1] = 3.0*v[0,j]  - 3.0*v[1,j]    + v[2,j]
        vbx[j,2] = 3.0*v[nx,j] - 3.0*v[nx-1,j] + v[nx-2,j]
    end
    for i = 0:nx
        uby[i,1] = 3.0*u[i,0]  - 3.0*u[i,1]    + u[i,2]
        uby[i,2] = 3.0*u[i,ny] - 3.0*u[i,ny-1] + u[i,ny-2]
    end
    for j = 0:ny
        v[-1,j]  = vbx[j,1]
        v[nx+1,j] = vbx[j,2]
    end
    for i = 0:nx
        u[i,-1]  = uby[i,1]
        u[i,ny+1] = uby[i,2]
    end

    # we also need to extrapolate data for the stencil
    u[-1,0]    = 3.0*u[0,0]   - 3.0*u[1,0]     + u[2,0]
    u[nx+1,0]  = 3.0*u[nx,0]  - 3.0*u[nx-1,0]  + u[nx-2,0]
    u[-1,ny]   = 3.0*u[0,ny]  - 3.0*u[1,ny]    + u[2,ny]
    u[nx+1,ny] = 3.0*u[nx,ny] - 3.0*u[nx-1,ny] + u[nx-2,ny]
    #
    v[0,-1]    = 3.0*v[0,0]   - 3.0*v[0,1]     + v[0,2]
    v[0,ny+1]  = 3.0*v[0,ny]  - 3.0*v[0,ny-1]  + v[0,ny-2]
    v[nx,-1]   = 3.0*v[nx,0]  - 3.0*v[nx,1]    + v[nx,2]
    v[nx,ny+1] = 3.0*v[nx,ny] - 3.0*v[nx,ny-1] + v[nx,ny-2]

    # To the left and right we get u from the zero divergence condition to the left and right
    # At the bottom and top we get v from the zero divergence condition at the bottom and top
    # to update ghost point values
    for j = 0:ny
        ubx[j,1] = u[1,j]    + (hx/hy)*(v[0,j+1]-v[0,j-1])
        ubx[j,2] = u[nx-1,j] - (hx/hy)*(v[nx,j+1]-v[nx,j-1])
    end
    for i = 0:nx
        vby[i,1] = v[i,1]    + (hy/hx)*(u[i+1,0]-u[i-1,0])
        vby[i,2] = v[i,ny-1] - (hy/hx)*(u[i+1,ny]-u[i-1,ny])
    end
    for j = 0:ny
        u[-1,j]   = ubx[j,1]
        u[nx+1,j] = ubx[j,2]
    end
    for i = 0:nx
        v[i,-1]   = vby[i,1]
        v[i,ny+1] = vby[i,2]
    end

    # Finally we extrapolate to the corners
    u[-1,-1]     = 3.0*u[0,-1]    - 3.0*u[1,-1]      + u[2,-1]
    u[nx+1,-1]   = 3.0*u[nx,-1]   - 3.0*u[nx-1,-1]   + u[nx-2,-1]
    u[-1,ny+1]   = 3.0*u[0,ny+1]  - 3.0*u[1,ny+1]    + u[2,ny+1]
    u[nx+1,ny+1] = 3.0*u[nx,ny+1] - 3.0*u[nx-1,ny+1] + u[nx-2,ny+1]
    v[-1,-1]     = 3.0*v[-1,0]    - 3.0*v[-1,1]      + v[-1,2]
    v[-1,ny+1]   = 3.0*v[-1,ny]   - 3.0*v[-1,ny-1]   + v[-1,ny-2]
    v[nx+1,-1]   = 3.0*v[nx+1,0]  - 3.0*v[nx+1,1]    + v[nx+1,2]
    v[nx+1,ny+1] = 3.0*v[nx+1,ny] - 3.0*v[nx+1,ny-1] + v[nx+1,ny-2]

end

function computeGPforP!(u,v,prob::INSProb)
    # We compute "ghost values" for p
    # pbx will be used to the right hand side in the Poisson eq.
    # This approximation uses the curl-curl condition

    nx = prob.nx
    ny = prob.ny
    nu = prob.nu
    hx = prob.hx
    hy = prob.hy
    for j = 0:ny
        # left
        prob.pbx[1+j,1] = -2.0*hx*(-prob.gux[1+j,1]*(u[1,j]-u[-1,j])/(2.0*hx)-
            prob.gvx[1+j,1]*(u[0,j+1]-u[0,j-1])/(2.0*hy)+
            nu*(-(v[1,j+1]-v[-1,j+1]-v[1,j-1]+v[-1,j-1])/(4.0*hx*hy)+
            (u[0,j+1]-2.0*u[0,j]+u[0,j-1])/(hy^2)))
        # right
        prob.pbx[1+j,2] =  2.0*hx*(-prob.gux[1+j,2]*(u[nx+1,j]-u[nx-1,j])/(2.0*hx)-
            prob.gvx[1+j,2]*(u[nx,j+1]-u[nx,j-1])/(2.0*hy)+
            nu*(-(v[nx+1,j+1]-v[nx-1,j+1]-v[nx+1,j-1]+v[nx-1,j-1])/(4.0*hx*hy)+
            (u[nx,j+1]-2.0*u[nx,j]+u[nx,j-1])/(hy^2)))
    end
    for i = 0:nx
        # bottom
        prob.pby[1+i,1] = -2.0*hy*(-prob.gvy[1+i,1]*(v[i,1]-v[i,-1])/(2.0*hy)-
            prob.guy[1+i,1]*(v[i+1,0]-v[i-1,0])/(2.0*hx)+
            nu*(-(u[i+1,1]-u[i-1,1]-u[i+1,-1]+u[i-1,-1])/(4.0*hy*hx)+
            (v[i+1,0]-2.0*v[i,0]+v[i-1,0])/(hx^2)))
        # top
        prob.pby[1+i,2] = +2.0*hy*(-prob.gvy[1+i,2]*(v[i,ny+1]-v[i,ny-1])/(2.0*hy)-
            prob.guy[1+i,2]*(v[i+1,ny]-v[i-1,ny])/(2.0*hx)+
            nu*(-(u[i+1,ny+1]-u[i+1,ny-1]-u[i-1,ny+1]+u[i-1,ny-1])/(4.0*hy*hx)+
            (v[i+1,ny]-2.0*v[i,ny]+v[i-1,ny])/(hx^2)))
    end
end

function setupRhsideP!(u,v,prob::INSProb)

    nx = prob.nx
    ny = prob.ny
    hx = prob.hx
    hy = prob.hy
    alpha = prob.alpha
    hxi2 = 1.0/hx^2
    hyi2 = 1.0/hy^2

    prob.bP .= 0.0
    prob.bPBig .= 0.0

    for j = 1:ny-1
        for i = 1:nx-1
            D0xu = (u[i+1,j]-u[i-1,j])*(1.0/(2.0*hx))
            D0xv = (v[i+1,j]-v[i-1,j])*(1.0/(2.0*hx))
            D0yu = (u[i,j+1]-u[i,j-1])*(1.0/(2.0*hy))
            D0yv = (v[i,j+1]-v[i,j-1])*(1.0/(2.0*hy))
            F = alpha*(D0xu+D0yv)-(D0xu^2+D0yv^2+2*D0xv*D0yu)
            # inner points
            k = 1+i+j*(nx+1)
            prob.bP[k] = F
        end
    end

    # Bottom and top sides
    j = 0
    for i=1:nx-1
        k = 1+i + j*(nx+1)
        D0xu = (u[i+1,j]-u[i-1,j])*(1.0/(2.0*hx))
        D0xv = (v[i+1,j]-v[i-1,j])*(1.0/(2.0*hx))
        D0yu = (u[i,j+1]-u[i,j-1])*(1.0/(2.0*hy))
        D0yv = (v[i,j+1]-v[i,j-1])*(1.0/(2.0*hy))
        F = alpha*(D0xu+D0yv)-(D0xu^2+D0yv^2+2*D0xv*D0yu)
        prob.bP[k] = F - prob.pby[1+i,1]*hyi2
    end

    j = ny
    for i=1:nx-1
        k = 1+i + j*(nx+1)
        D0xu = (u[i+1,j]-u[i-1,j])*(1.0/(2.0*hx))
        D0xv = (v[i+1,j]-v[i-1,j])*(1.0/(2.0*hx))
        D0yu = (u[i,j+1]-u[i,j-1])*(1.0/(2.0*hy))
        D0yv = (v[i,j+1]-v[i,j-1])*(1.0/(2.0*hy))
        F = alpha*(D0xu+D0yv)-(D0xu^2+D0yv^2+2*D0xv*D0yu)
        prob.bP[k] = F - prob.pby[1+i,2]*hyi2
    end
    # Left and right sides
    i = 0
    for j=1:ny-1
        k = 1 + i + j*(nx+1)
        D0xu = (u[i+1,j]-u[i-1,j])*(1.0/(2.0*hx))
        D0xv = (v[i+1,j]-v[i-1,j])*(1.0/(2.0*hx))
        D0yu = (u[i,j+1]-u[i,j-1])*(1.0/(2.0*hy))
        D0yv = (v[i,j+1]-v[i,j-1])*(1.0/(2.0*hy))
        F = alpha*(D0xu+D0yv)-(D0xu^2+D0yv^2+2*D0xv*D0yu)
        prob.bP[k] = F - prob.pbx[1+j,1]*hxi2
    end
    i = nx
    for j=1:ny-1
        k = 1 + i + j*(nx+1)
        D0xu = (u[i+1,j]-u[i-1,j])*(1.0/(2.0*hx))
        D0xv = (v[i+1,j]-v[i-1,j])*(1.0/(2.0*hx))
        D0yu = (u[i,j+1]-u[i,j-1])*(1.0/(2.0*hy))
        D0yv = (v[i,j+1]-v[i,j-1])*(1.0/(2.0*hy))
        F = alpha*(D0xu+D0yv)-(D0xu^2+D0yv^2+2*D0xv*D0yu)
        prob.bP[k] = F - prob.pbx[1+j,2]*hxi2
    end

    # corners
    for j = (0,ny)
        for i = (0,nx)
            k =  1 + i + j*(nx+1)
            D0xu = (u[i+1,j]-u[i-1,j])*(1.0/(2.0*hx))
            D0xv = (v[i+1,j]-v[i-1,j])*(1.0/(2.0*hx))
            D0yu = (u[i,j+1]-u[i,j-1])*(1.0/(2.0*hy))
            D0yv = (v[i,j+1]-v[i,j-1])*(1.0/(2.0*hy))
            F = alpha*(D0xu+D0yv)-(D0xu^2+D0yv^2+2*D0xv*D0yu)
            prob.bP[k] = F
            if  i == 0
                prob.bP[k] = prob.bP[k]-prob.pbx[1+j,1]*hxi2
            end
            if  i == nx
                prob.bP[k] = prob.bP[k] - prob.pbx[1+j,2]*hxi2
            end
            if j == 0
                prob.bP[k] = prob.bP[k] - prob.pby[1+i,1]*hyi2
            end
            if j == ny
                prob.bP[k] = prob.bP[k] - prob.pby[1+i,2]*hyi2
            end
        end
    end
    for j = 0:ny
        for i = 0:nx
            k =  1 + i + j*(nx+1)
            prob.bPBig[k] = prob.bP[k]
        end
    end
end

function computeLE!(Leu,Lev,u,v,p,prob::INSProb)
    # This routine computes the explicit part of the right hand side
    nx = prob.nx
    ny = prob.ny
    hx = prob.hx
    hy = prob.hy
    hx2i = 0.5/hx
    hy2i = 0.5/hy
    for j = 0:ny
        for i = 0:nx
            Leu[i,j] = -(u[i,j]*(u[i+1,j]-u[i-1,j])*hx2i+
                v[i,j]*(u[i,j+1]-u[i,j-1])*hy2i+
                (p[i+1,j]-p[i-1,j])*hx2i)
            Lev[i,j] = -(u[i,j]*(v[i+1,j]-v[i-1,j])*hx2i+
                v[i,j]*(v[i,j+1]-v[i,j-1])*hy2i+
                (p[i,j+1]-p[i,j-1])*hy2i)
        end
    end
end

function computeLI!(Liu,Liv,u,v,prob::INSProb)
    # This routine computes the explicit part of the right hand side
    nx = prob.nx
    ny = prob.ny
    hx = prob.hx
    hy = prob.hy
    hxi2 = prob.nu/hx^2
    hyi2 = prob.nu/hy^2
    for j = 0:ny
        for i = 0:nx
            Liu[i,j] = (u[i+1,j]-2.0*u[i,j]+u[i-1,j])*hxi2+
                (u[i,j+1]-2.0*u[i,j]+u[i,j-1])*hyi2
            Liv[i,j] = (v[i+1,j]-2.0*v[i,j]+v[i-1,j])*hxi2+
                (v[i,j+1]-2.0*v[i,j]+v[i,j-1])*hyi2
        end
    end
end

function setupRhsideUV!(Leu,Lev,Leuold,Levold,Liu,Liv,u,v,prob::INSProb)

    prob.bU .= 0.0
    prob.bV .= 0.0
    nx = prob.nx
    ny = prob.ny
    hx = prob.hx
    hy = prob.hy

    hxi2 = 1.0/hx^2
    hyi2 = 1.0/hy^2
    # loop over side
    for j = 2:ny-1
        for i = (1,nx-1)
            l = i + (j-1)*(nx-1)
            if i == 1
                prob.bU[l] = prob.bU[l] - prob.gux[1+j,1]*hxi2
                prob.bV[l] = prob.bV[l] - prob.gvx[1+j,1]*hxi2
            end
            if i == nx-1
                prob.bU[l] = prob.bU[l] - prob.gux[1+j,2]*hxi2
                prob.bV[l] = prob.bV[l] - prob.gvx[1+j,2]*hxi2
            end
        end
    end

    # loop over side
    for j = (1,ny-1)
        for i = 1:nx-1
            l = i + (j-1)*(nx-1)
            if  j == 1
                prob.bU[l] = prob.bU[l] - prob.guy[1+i,1]*hyi2
                prob.bV[l] = prob.bV[l] - prob.gvy[1+i,1]*hyi2
            end
            if j == ny-1
                prob.bU[l] = prob.bU[l] - prob.guy[1+i,2]*hyi2
                prob.bV[l] = prob.bV[l] - prob.gvy[1+i,2]*hyi2
            end
        end
    end
    # loop over corners
    for j = (1,ny-1)
        for i = (1,nx-1)
            l = i + (j-1)*(nx-1)
            if i == 1
                prob.bU[l] = prob.bU[l] - prob.gux[1+j,1]*hxi2
                prob.bV[l] = prob.bV[l] - prob.gvx[1+j,1]*hxi2
            end
            if i == nx-1
                prob.bU[l] = prob.bU[l] - prob.gux[1+j,2]*hxi2
                prob.bV[l] = prob.bV[l] - prob.gvx[1+j,2]*hxi2
            end
            if j == 1
                prob.bU[l] = prob.bU[l] - prob.guy[1+i,1]*hyi2
                prob.bV[l] = prob.bV[l] - prob.gvy[1+i,1]*hyi2
            end
            if j == ny-1
                prob.bU[l] = prob.bU[l] - prob.guy[1+i,2]*hyi2
                prob.bV[l] = prob.bV[l] - prob.gvy[1+i,2]*hyi2
            end
        end
    end

    prob.bU .*= -0.5*prob.nu
    prob.bV .*= -0.5*prob.nu


    for j = 1:ny-1
        for i = 1:nx-1
            l = i + (j-1)*(nx-1)
            prob.bU[l] = prob.bU[l] + u[i,j]*(1.0/prob.dt)+
                1.5*Leu[i,j]-0.5*Leuold[i,j]+0.5*Liu[i,j]
            prob.bV[l] = prob.bV[l] + v[i,j]*(1.0/prob.dt)+
                1.5*Lev[i,j]-0.5*Levold[i,j]+0.5*Liv[i,j]
        end
    end

end

function taylor_vortex(X,Y,Lx,Ly,r0,gamma)
    
    R = sqrt((X-Lx)^2+(Y-Ly)^2)
    R = gamma*exp(-(R/r0)^2)
    u = -(Y-Ly)*R
    v =  (X-Lx)*R
    return u,v
end 

end
