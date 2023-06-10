using Pkg;
Pkg.add("Plots")
Pkg.add("CUDA")
Pkg.add("ForwardDiff")
Pkg.add("DifferentialEquations")
using CUDA
using ForwardDiff
CUDA.versioninfo()
using Plots
using LinearAlgebra
using DifferentialEquations
using Random
bridge_half_width=2
bridge_length=325
ra = 0.31
la = 0.31
A1=1.7
A2 = 1.7
B1 = 0.28
B2=0.28
tau=0.5
Ai=5
Bi =0.1
M=113e3
bridge_damp=0.043
Omega=2*pi*1.03

function social_force_term(diffzx,diffzy)
    distz=sqrt.(diffzx^2+diffzy^2)
    d=max.(distz,1e-8)
    nx=diffzx/d
    ny=diffzy/d
    cosphi=-ny
    F1 = 1.7*exp((2*0.31 - d) / 0.28) * (0.31 + (1 - 0.31) * ((1 + cosphi) / 2)) + 1.7 * exp((2*0.31 - d) / 0.28)
    F1*nx,F1*ny
end
function wall_avoidance_term(z)
    bound=2
    leftwall=min.(z+bound,-1e-7)
    rightwall=max.(z-bound,1e-7)    
    left_repel=5*exp.((0.31-abs.(leftwall))./0.1).*leftwall
    right_repel=5*exp.((0.31-abs.(rightwall))./0.1).*rightwall
    left_repel.+right_repel
end
function social_force_frontiers(F::CuDeviceVector{Float64},N::Int64,Z::CuDeviceMatrix{Float64},Zdot::CuDeviceMatrix{Float64},Vd::CuDeviceVector{Float64})
   
    Zshared = CUDA.@cuDynamicSharedMem(Float64, N*2)
    i=threadIdx().x
    z1::Float64=convert(Float64,0)
    z2::Float64=convert(Float64,0)
    if i<=N
        z1=Z[i,1]
        z2=Z[i,2]
        Zshared[(i-1)*2+1]=z1
        Zshared[(i-1)*2+2]=z2
    end
        sync_threads()
    if i<=N
        Fix::Float64=0
        Fiy::Float64=0
        for j in range(1,N)
            zj1=Zshared[(j-1)*2+1]
            zj2=Zshared[(j-1)*2+2]
            fx,fy=social_force_term(z1-zj1,z2-zj2)
            Fix+=(i==j) ? 0 : fx 
            Fiy+=(i==j) ? 0 : fy 
        end
        Fix+=wall_avoidance_term(z1)
        F[i]=0.5*(-Zdot[i,1]).+Fix    
        F[i+N]=0.5*(Vd[i]-Zdot[i,2]).+Fiy    
    end
    return nothing
end

function test_force(x,y)
  norm(social_force_term(
                x, y
                 )
    )
end


g=9.81
function H_1(y,p,L)
    g./L.*(p-y)
end
function bridge_ode(deriv::CuDeviceVector{Float64},
                    N::Int64,
                    L::CuDeviceVector{Float64},
                    p::CuDeviceVector{Float64},
                    rsum::Float64,
                    msum::Float64,
                    m::CuDeviceVector{Float64},
                    state::CuDeviceVector{Float64},
                    M::Float64,
                    Omega::Float64,
                    bridge_damp::Float64)
    i=convert(Int32,threadIdx().x)
    foot_force=convert(Float64,0)
    g_=convert(Float64,9.81)
    deriv_shared = CUDA.@cuDynamicSharedMem(eltype(deriv), 1)
    if i==1
        deriv_shared[1]=0
    end
    sync_threads()
    if i<=N
        r=m[i]/(M+msum)
        y0dot=state[2+5*N+i]
        zdot=state[2+4*N+i]
        vnorm=sqrt(zdot^2+y0dot^2)
        foot_force=g_/L[i]*(p[i]-state[i+2])*(carroll(vnorm)>=0.3)
        deriv_term=1/(1-rsum)*(foot_force*r)
        CUDA.atomic_add!(CUDA.pointer(deriv_shared),deriv_term)
    end
    sync_threads()
    if i==1
            xdot=state[2]
            x=state[1]
            deriv[i]=xdot
            xdotdot=1/(1-rsum)*(-bridge_damp*Omega*xdot-Omega*Omega*x)
            deriv_shared[1]+=xdotdot
            deriv[i+1]=deriv_shared[1]
    end
    sync_threads()
    if i<=N       
        deriv[i+2+3*N]=-foot_force-deriv_shared[1] 
    end
    return nothing
end

function bridge_pedestrian_ode_model2(deriv,state,param,t)
    N,L,m,p,vd,msum,rsum=param
    y0=state[3+2*N:2+3*N]
    y1=state[3:2+N]
    y=min.(max.(y1+y0,-bridge_half_width),bridge_half_width)
    z=state[3+N:2+2*N] .%bridge_length
    ydot=state[3+3*N:2+4*N]
    zdot=state[3+4*N:2+5*N]
    y0dot=state[3+5*N:2+6*N]
    M_=convert(Float64,M)
    Omega_=convert(Float64,Omega)
    bridge_damp_=convert(Float64,bridge_damp)
    @sync @cuda blocks=1 threads=N shmem=sizeof(Float64) bridge_ode(deriv,N,L,p,rsum,msum,m,state,M_,Omega_,bridge_damp_)
    Z=CuArray{Float64}(CUDA.zeros(N,2))
    Zdot=CuArray{Float64}(CUDA.zeros(N,2))
    Z[:,1].=y
    Z[:,2].=z
    Zdot[:,1].=y0dot
    Zdot[:,2].=zdot
    crowd=CuArray{Float64}(CUDA.zeros(N*2))
    @sync  @cuda blocks=1 threads=N shmem=2*N*sizeof(Float64) social_force_frontiers(crowd,N,Z,Zdot,vd)
    vnorm=sqrt.(y0dot.^2+zdot.^2)
    deriv[3:N+2].=ydot.*(carroll(vnorm).>=0.3)
    deriv[3+N:2+2*N].=zdot
    deriv[3+2*N:2+3*N].=y0dot
    deriv[3+4*N:2+5*N].=crowd[N+1:2*N]
    deriv[3+5*N:2+6*N].=crowd[1:N]
end

function carroll(v)
    0.5*(0.35*v.^3 - 1.59*v.^2 + 2.93*v)
end
function deriv_carroll(v)
    0.5*(0.35*3*v.^2-1.59*2*v .+2.93)
end
function inverse_carroll(v0,fp)
    v=v0
    for it in range(1,100)
        v.-=0.99*(carroll(v)-fp)./deriv_carroll(v)
    end
    v
end

function generate_crowd(N,mean_freq,std_freq)
    L=randn(N)*0.092.+1.17
    bmin=0.0157.+0.002*randn(N)
    bmin.*=1 .-2*rand(N)
    y0=rand(N)*4 .-2
    y0dot=zeros(N)
    y=rand(N)*0.1.-0.2
    z=rand(N)*325
    x=0.
    xdot=0.
    m=76.9 .+10*randn(N)
    vd=inverse_carroll(ones(N),abs.(randn(N)*std_freq .+mean_freq))
    vd=max.(0.1,vd)
    fp=carroll(vd)
    p=bmin.*(1.0.-tanh.(0.25*sqrt.(9.8./L) ./ fp))
    bmin.=bmin.*-1
    ydot= p .* sqrt.(9.81./L) .* tanh.(sqrt.(9.81./L) .* 0.5./fp)
    tnext=rand(N)*0.5 ./fp
    tprev=tnext-0.5 ./fp
    msum=convert(Float64,sum(m))
    r=m./(M+msum)
    rsum=convert(Float64,sum(r))
    cat([x,xdot],y,z,y0,ydot,vd,y0dot,dims=1), (L,m,p,tnext,tprev,bmin,vd,msum,rsum)
end


        

function crowd_loop(Tmax,N,crowd_state,crowd_param, tracefile)
    t=0
    t_trace_last=-0.1
    L,m,p,tnext,tprev,bmin,vd,msum,rsum=crowd_param
    L_d=CuArray{Float64}(L)
    vd_d=CuArray{Float64}(vd)
    m_d=CuArray{Float64}(m)
    while t<Tmax
        #solve the ODE until the next pedestrian footfall
        L,m,p,tnext,tprev,bmin,vd,msum,rsum=crowd_param
        p_d=CuArray{Float64}(p)
        crowd_param_gpu=N,L_d,m_d,p_d,vd_d,msum,rsum
        t1=minimum(tnext)
        if t1-t>1e-10
          prob=ODEProblem(bridge_pedestrian_ode_model2,CuArray{Float64}(crowd_state), (0,t1-t), crowd_param_gpu)
          sol=solve(prob, AutoTsit5(Rosenbrock23()),
                 abstol=1e-13,reltol=1e-10,save_everystep = false)
          crowd_state=Array(sol.u[size(sol.u)[1]])
        end
        #update the step timing and xCOP for the pedestrians that stepped...
        ii=(tnext .<=t1+1e-10)
        zdot=crowd_state[3+4*N:2+5*N]
        y0=crowd_state[3+2*N:2+3*N]
        y0dot=crowd_state[3+5*N:2+6*N]
        vnorm=sqrt.(zdot.^2+y0dot.^2)
        crowd_state[3+3*N:2+4*N].*=(carroll(vnorm).>=0.3)
        ydot=crowd_state[3+3*N:2+4*N]
        y1=crowd_state[3:2+N]
        bnd=bridge_half_width
        p[ii].=y1[ii]+(ydot[ii].*sqrt.(L[ii]./9.81).*(carroll(vnorm[ii]).>=0.3))+bmin[ii]
        bmin[ii].=-bmin[ii]
        tprev[ii].=tnext[ii]
        fp=max.(0.3,carroll(vnorm))
        step_width=abs.(bmin)./(1 .-tanh.(sqrt.(0.91 ./L).*0.25./fp))
        step_length=max.(1e-1,0.36*(zdot/1.151466))
        tnext[ii].=tnext[ii].+max.(0.1,0.5./fp[ii].*
        (1 .+(carroll(vnorm[ii]).>=0.3).*(step_width[ii].^2-(y1[ii]-p[ii]).^2)./(4*step_length[ii].^2)))
        t=t1
        #increase the time and, at a certain interval, write out the traces to the file
        if t_trace_last+0.01<=t
            z=crowd_state[3+N:2+2*N]
            x,xdot=crowd_state[1],crowd_state[2]
            t_trace_last=t
            line="$(N) $(t) $(x) $(xdot) "
            for i in range(1,N)
                line=line*" $(g/L[i]*(p[i]-y1[i])) "
            end
            for i in range(1,N)
                line=line*" $(p[i]-y1[i]) "
            end
            for i in range(1,N)
                line=line*" $(y1[i]) "
            end
            for i in range(1,N)
                line=line*" $(y0[i]) "
            end
            for i in range(1,N)
                line=line*" $(z[i]) "
            end
            for i in range(1,N)
                line=line*" $(ydot[i]) "
            end
            for i in range(1,N)
                line=line*" $(y0dot[i]) "
            end
            for i in range(1,N)
                line=line*" $(zdot[i]) "
            end 
            for i in range(1,N)
                line=line*" $(bmin[i]) "
            end
            for i in range(1,N)
                line=line*" $(tnext[i]) "
            end
            for i in range(1,N)
                line=line*" $(tprev[i]) "
            end
            for i in range(1,N)
                line=line*" $(fp[i]) "
            end


            for i in range(1,N)
                line=line*" $(vd[i]) "
            end
            write(tracefile,line*"\n")
            flush(tracefile)
        end #if
        crowd_param=(L,m,p,tnext,tprev,bmin,vd,msum,rsum)
    end   #while
end #fn
function compute_traces!(N, std_freq, sample)
    Tmax=20
    mean_freq=0.95
    CURAND.seed!(sample)
    Random.seed!(sample)
    tracefile=open("crowd-traces/$(N)_$(mean_freq)_$(std_freq)_$(sample).txt","w+")
    
                #initial conditions
    crowd_state, crowd_param=generate_crowd(N,mean_freq,std_freq)
    L,m,p,tnext,tprev,bmin,vd=crowd_param
    crowd_loop(Tmax,N,crowd_state,crowd_param,tracefile)
    nothing
end
for N in range(160,250,step=5)
    for sample in range(1,10)
      @sync begin
                for sigma in range(0,1.0,step=0.1)
                    @async begin 
                            compute_traces!(N,sigma,sample)  
                           end
                end
            end
    end
end
            
