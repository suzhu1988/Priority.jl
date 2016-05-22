using PyCall,Distributions,DataStructures
@pyimport networkx as nx
@pyimport matplotlib.pyplot as plt
function generator_graphs(L,m=2)
    g=nx.Graph()
    h=nx.barabasi_albert_graph(L,m)
    d=Normal(1,0.33)
    edges=h[:edges]()
    for edge in edges
        g[:add_edge](edge[1]+1,edge[2]+1,weight=abs(rand(d)))
    end
    return g
end

function _all_pairs_dijkstra_path(g)
    path=nx.all_pairs_dijkstra_path(g)
    return path
end

function _all_pairs_shortest_path_length(g)
    pathlength=nx.all_pairs_shortest_path_length(g)
    return pathlength
end

type Package
    size_p::Float64
    src_p::Int
    dis_p::Int
    pos_p::Int
    time_p::Int
    step_p::Int
    priority::Int
end

function package_brith(g,package,packagequeue,P,N,t)
    for i =1:P
        d=Normal(1,0.33)
        #n=nx.number_of_nPriorityes(g)
        sizep=abs(rand(d))
        srcp=sample(1:N)
        disp=sample(1:N)
        priorityp=sample(1:5)
        package[(P*(t-1)+i)]=Package(sizep,srcp,disp,srcp,0,0,priorityp)
        packagequeue[package[(P*(t-1)+i)].pos_p][P*(t-1)+i]=package[P*(t-1)+i].priority
    end
    return package,packagequeue
end

function _load(load_arr,package)
    fill!(load_arr,0)
    load=0
    for (k,v) in package
        k_pos=package[k].pos_p
        k_size=package[k].size_p
        load_arr[(k_pos)]+=k_size
        package[k].time_p+=1 #update package time
        load+=k_size
    end
    return load_arr,load
end

function effective_distance(g,k,j,path,load_arr,h)
    ekj=h*length(path[k][j])+(1-h)*load_arr[k]
    return ekj
end

function traffic_aware_routing_strategy(i,g,package,package_life,package_step,packagequeue,path,load_arr,h)
    pos=package[i].pos_p
    dis=package[i].dis_p
    if pos == dis
        package_life[i]=package[i].time_p
        package_step[i]=package[i].step_p
        delete!(package,i)
    else
        nebor_arr=nx.neighbors(g,pos)
        min_effecitve_distance=10000
        best_k=0
        for j=1:length(nebor_arr)
            ekj=effective_distance(g,nebor_arr[j],dis,path,load_arr,h)
            if min_effecitve_distance>ekj
                min_effecitve_distance=ekj
                best_k=nebor_arr[j]
            end
        end
        package[i].pos_p=best_k
        package[i].step_p+=1
        if best_k==dis
            package[i].time_p+=1
            package_life[i]=package[i].time_p
            package_step[i]=package[i].step_p
            delete!(package,i)
        else
            packagequeue[best_k][i]=package[i].priority
        end
    end
end

function initiate_queue(packagequeue,N)
    for i=1:N
        packagequeue[i]=Collections.PriorityQueue()
    end
end

function Package_move(packagequeue,move_package,N)
    for i=1:N
        if !isempty(packagequeue[i])
            push!(move_package,Collections.dequeue!(packagequeue[i]))
        end
    end
end

function update_package(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,h)
    while !isempty(move_package)
        k=pop!(move_package)
        traffic_aware_routing_strategy(k,g,package,package_life,package_step,packagequeue,path,load_arr,h)
    end
end

function initiate_graph(L)
    # initiate a graph
    g=generator_graphs(L)
    #g=remove_edges(g)
    #number of nodes
    N=nx.number_of_nodes(g)
    D=nx.diameter(g)
    path=_all_pairs_dijkstra_path(g)
    pathlength=_all_pairs_shortest_path_length(g)
    return g,N,D,path,pathlength
end

function initiate_package(L,N)
    # initiate a package
    package=Dict{Int,Package}()
    package_life=Dict{Int,Int}()
    package_step=Dict{Int,Int}()
    # initiate the moving package
    move_package=Set{Int}()
    # initiage the queue of nodes
    packagequeue=Array(Collections.PriorityQueue,N)
    initiate_queue(packagequeue,N)
    # initiate a load array
    load_arr=zeros(N)
    total_load=Float64[]
    return package,package_life,package_step,move_package,packagequeue,load_arr,total_load
end

function reach_stable_state(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,total_load,h,P,N,T)
    for t=1:T
        # generate a package
        package,packagequeue=package_brith(g,package,packagequeue,P,N,t)
        Package_move(packagequeue,move_package,N)
        # update the load array
        load_arr,load=_load(load_arr,package)
        # update the package
        update_package(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,h)
        #calculate the order parameter
        push!(total_load,load)
    end
end

function free_state_nodes(s,load_arr,m,N)
    empty!(s)
    S=0
    for i=1:N
        if load_arr[i]<=m
            push!(s,i)
            S+=1
        end
    end
    return s,S
end

function largest_cluster(g,s,load_arr,m,N)
    s,S=free_state_nodes(s,load_arr,m,N)
    G=nx.subgraph(g,s)
    SG=nx.connected_component_subgraphs(G)
    largest=length(SG[1])
    second_largest= (length(SG)>1)? length(SG[2]) : 0
    return S,largest,second_largest
end

function giant_component(g,s,load_arr,mc,N)
    s,S=free_state_nodes(s,load_arr,mc,N)
    G=nx.subgraph(g,s)
    SG=nx.connected_component_subgraphs(G)
    return SG[1]
end
function find_pc(p,sl,s,g,N,load_arr,m_min,m_max)
    empty!(p)
    empty!(sl)
    Δm1=0.2
    Δm2=0.02
    m=m_min
    while m<=m_max
        S,largest,second_largest=largest_cluster(g,s,load_arr,m,N)
        push!(p,S/N)
        push!(sl,second_largest/N)
        m += (0.3<S/N<0.7) ?  Δm2 : Δm1
    end
    max_index=findmax(sl)[2]
    pc=p[max_index]
    return pc
end

function find_mc(mm,sl,s,g,N,load_arr,m_min,m_max)
    empty!(mm)
    empty!(sl)
    Δm1=0.2
    Δm2=0.02
    m=m_min
    while m<=m_max
        S,largest,second_largest=largest_cluster(g,s,load_arr,m,N)
        push!(mm,m)
        push!(sl,second_largest/N)
        m += (0.3<S/N<0.7) ?  Δm2 : Δm1
    end
    max_index=findmax(sl)[2]
    mc=mm[max_index]
    return mc
end

function _critical_p(p,sl,s,g,N,load_arr)
    # find the minimum m and minmax m
    m_min=minimum(load_arr)
    m_max=maximum(load_arr)
    # critical threshold
    pc=find_pc(p,sl,s,g,N,load_arr,m_min,m_max)
    return pc
end

function _ρ(total_load,P,Δt,T)
    ρ=0
    T0=T-0.1*T+1
    for t=T0:T
        ρ+=(total_load[t]-total_load[t-Δt])/(Δt*P)
    end
    return ρ/(T-T0)
end

function rs(pathlength,N,D)
    R=Array(Vector{Tuple{Int,Int}},D)
    for i =1:D
        R[i]=Vector{Tuple{Int,Int}}[]
    end
    for i=1:N,j=1:N
        if pathlength[i][j] != 0
        append!(R[pathlength[i][j]],[(i,j)])
        end
    end
    return R
end

function _cor(σ2,cor_data,R,r)
    C=0
    if !isempty(R[r])
        s=0
        for i=1:length(R[r])
            xi=R[r][i][1]
            xj=R[r][i][2]
            s+=cor_data[xi]*cor_data[xj]
        end
        C=s/(length(R[r])*σ2)
    else
        C=0
    end
    return C
end

function _ζ(load_arr,pathlength,N,D)
    average_load=mean(load_arr)
    σ2=var(load_arr)
    cor_data=load_arr-average_load
    r=1
    ζ=0
    R=rs(pathlength,N,D)
    while r<=D
        C=_cor(σ2,cor_data,R,r)
        if C<=0
            ζ=r
            break
        end
        r+=1
    end
    return ζ
end

function average_bw(mm,sl,s,g,N,load_arr)
    m_min=minimum(load_arr)
    m_max=maximum(load_arr)
    mc=find_mc(mm,sl,s,g,N,load_arr,m_min,m_max)
    IIC=giant_component(g,s,load_arr,mc,N)
    bw=nx.betweenness_centrality(IIC)  #Dict
    average_C=mean(collect(values(bw)))
    return average_C
end
function reach_stable_time(P=150,h=0.3,L=100,T=10000)
    outfreachstabletime=open("/home/SuZ/Data/Priority/ScaleFree_ReachStableTime_($h)_($L)_($P).txt","w")
    g,N,D,path,pathlength=initiate_graph(L)
    package,package_life,package_step,move_package,packagequeue,load_arr,total_load=initiate_package(L,N)
    # reach stable state
    for t=1:T
        # generate a package
        package,packagequeue=package_brith(g,package,packagequeue,P,N,t)
        Package_move(packagequeue,move_package,N)
        # update the load array
        load_arr,load=_load(load_arr,package)
        # update the package
        update_package(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,h)
        #calculate the order parameter
        push!(total_load,load)
    end
    #ρ+=_ρ(total_load,P,Δt,T)
    writedlm(outfreachstabletime,total_load)
    close(outfreachstabletime)
end
function order_parameter(Pstat=10,Pend=100,ΔP=2,samples=5,h=0.3,L=100,T=10000,Δt=10)
    outforderparameter=open("/home/SuZ/Data/Priority/ScaleFree_Orderparameter_($h)_($L).txt","w")
    g,N,D,path,pathlength=initiate_graph(L)
    P=Pstat
    while P<=Pend
        ρ=0
        for i=1:samples
            #initiate
            package,package_life,package_step,move_package,packagequeue,load_arr,total_load=initiate_package(L,N)
            # reach stable state
            reach_stable_state(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,total_load,h,P,N,T)
            ρ+=_ρ(total_load,P,Δt,T)
        end
        writedlm(outforderparameter,transpose([P,ρ/samples]))
        P+=ΔP
    end
    close(outforderparameter)
end

function percolation(P=150,h=0.3,L=100,T=10000)
    #outfile
    outfpercolation=open("/home/SuZ/Data/Priority/ScaleFree_Percolation_($P)_($h)_($L).txt","w")
    #parameters array
    s=Int[]
    #initiate
    g,N,D,path,pathlength=initiate_graph(L)
    sizehint!(s,N)
    package,package_life,package_step,move_package,packagequeue,load_arr,total_load=initiate_package(L,N)
    # reach stable state
    reach_stable_state(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,total_load,h,P,N,T)
    # find the minimum m and minmax m
    m_min=minimum(load_arr)
    m_max=maximum(load_arr)
    m=m_min
    Δm=0.2
    while m<=m_max
        S,largest,second_largest=largest_cluster(g,s,load_arr,m,N)
        writedlm(outfpercolation,transpose([S/N,largest/N,second_largest/N]))
        m+=Δm
    end
    close(outfpercolation)
end

function critical_threshold(Pstat=10,Pend=100,ΔP=2,samples=5,h=0.3,L=100,T=10000)
    outfcriticalthreshold=open("/home/SuZ/Data/Priority/ScaleFree_CriticalThreshold_($h)_($L).txt","w")
    #parameters array
    p=Float64[]
    sl=Float64[]
    s=Int[]
    #initiate
    g,N,D,path,pathlength=initiate_graph(L)
    sizehint!(s,N)
    P=Pstat
    while P<=Pend
        pc=0
        for i=1:samples
            #initiate
            package,package_life,package_step,move_package,packagequeue,load_arr,total_load=initiate_package(L,N)
            # reach stable state
            reach_stable_state(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,total_load,h,P,N,T)
            pc+=_critical_p(p,sl,s,g,N,load_arr)
        end
        writedlm(outfcriticalthreshold,transpose([P,pc/samples]))
        P+=ΔP
    end
    close(outfcriticalthreshold)
end

function correlation(P=150,h=0.3,L=100,T=10000)
    #outfile
    outfcorrelaiton=open("/home/SuZ/Data/Priority/ScaleFree_Correlation_($P)_($h)_($L).txt","w")
    #initiate
    g,N,D,path,pathlength=initiate_graph(L)
    package,package_life,package_step,move_package,packagequeue,load_arr,total_load=initiate_package(L,N)
    # reach stable state
    reach_stable_state(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,total_load,h,P,N,T)
    average_load=mean(load_arr)
    σ2=var(load_arr)
    cor_data=load_arr-average_load
    R=rs(pathlength,N,D)
    r=1
    while r<=D
        C=_cor(σ2,cor_data,R,r)
        writedlm(outfcorrelaiton,transpose([r,C]))
        r+=1
    end
    close(outfcorrelaiton)
end

function correlation_length(Pstat=10,Pend=100,ΔP=2,samples=5,h=0.3,L=100,T=10000)
    outfcorrelationlength=open("/home/SuZ/Data/Priority/ScaleFree_Correlationlength_($h)_($L).txt","w")
    g,N,D,path,pathlength=initiate_graph(L)
    P=Pstat
    while P<=Pend
        ζ=0
        for i=1:samples
            #initiate
            package,package_life,package_step,move_package,packagequeue,load_arr,total_load=initiate_package(L,N)
            # reach stable state
            reach_stable_state(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,total_load,h,P,N,T)
            ζ+=_ζ(load_arr,pathlength,N,D)
        end
        writedlm(outfcorrelationlength,transpose([P,ζ/samples]))
        P+=ΔP
    end
    close(outfcorrelationlength)
end

function betweenness_distribution(P=150,h=0.3,L=100,T=10000)
    outfbetweenness=open("/home/SuZ/Data/Priority/ScaleFree_Betweenness_($P)_($h)_($L).txt","w")
    #parameters array
    mm=Float64[]
    sl=Float64[]
    s=Int[]
    #initiate
    g,N,D,path,pathlength=initiate_graph(L)
    sizehint!(s,N)
    package,package_life,package_step,move_package,packagequeue,load_arr,total_load=initiate_package(L,N)
    # reach stable state
    reach_stable_state(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,total_load,h,P,N,T)
    m_min=minimum(load_arr)
    m_max=maximum(load_arr)
    mc=find_mc(mm,sl,s,g,N,load_arr,m_min,m_max)
    IIC=giant_component(g,s,load_arr,mc,N)
    bw=nx.betweenness_centrality(IIC)
    writedlm(outfbetweenness,bw)
    close(outfbetweenness)
end

function average_betweenness(Pstat=10,Pend=100,ΔP=2,samples=5,h=0.3,L=100,T=10000)
    outfaveragebetweenness=open("/home/SuZ/Data/Priority/ScaleFree_AverageBetweenness_($h)_($L).txt","w")
    #parameters array
    mm=Float64[]
    sl=Float64[]
    s=Int[]
    #initiate
    g,N,D,path,pathlength=initiate_graph(L)
    sizehint!(s,N)
    P=Pstat
    while P<=Pend
        average_C=0
        for i=1:samples
            #initiate
            package,package_life,package_step,move_package,packagequeue,load_arr,total_load=initiate_package(L,N)
            # reach stable state
            reach_stable_state(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,total_load,h,P,N,T)
            average_C+=average_bw(mm,sl,s,g,N,load_arr)
        end
        writedlm(outfaveragebetweenness,transpose([P,average_C/samples]))
        P+=ΔP
    end
    close(outfaveragebetweenness)
end

function pathlength_and_travelingtime_distribution(P=150,h=0.3,L=100,T=10000)
    #outfile
    outfpathlength=open("/home/SuZ/Data/Priority/ScaleFree_PathLengthDistribution_($P)_($h)_($L).txt","w")
    outftravelingtime=open("/home/SuZ/Data/Priority/ScaleFree_TravelingTimeDistribution_($P)_($h)_($L).txt","w")
    #parameters array
    s=Int[]
    #initiate
    g,N,D,path,pathlength=initiate_graph(L)
    sizehint!(s,N)
    package,package_life,package_step,move_package,packagequeue,load_arr,total_load=initiate_package(L,N)
    # reach stable state
    reach_stable_state(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,total_load,h,P,N,T)
    writedlm(outfpathlength,values(package_step))
    writedlm(outftravelingtime,values(package_life))
    close(outfpathlength)
    close(outftravelingtime)
end
function average_pathlength_and_travelingtime(Pstat=10,Pend=100,ΔP=2,samples=5,h=0.3,L=100,T=10000)
    outfaveragepathlength=open("/home/SuZ/Data/Priority/ScaleFree_AveragePathlength_($h)_($L).txt","w")
    outfaveragetravelingtime=open("/home/SuZ/Data/Priority/ScaleFree_AverageTravelingtime_($h)_($L).txt","w")
    #parameters array
    mm=Float64[]
    sl=Float64[]
    s=Int[]
    #initiate
    g,N,D,path,pathlength=initiate_graph(L)
    sizehint!(s,N)
    P=Pstat
    while P<=Pend
        average_pathlength=0
        average_travelingtime=0
        for i=1:samples
            #initiate
            package,package_life,package_step,move_package,packagequeue,load_arr,total_load=initiate_package(L,N)
            # reach stable state
            reach_stable_state(g,package,package_life,package_step,packagequeue,move_package,path,load_arr,total_load,h,P,N,T)
            average_pathlength+=mean(values(package_step))
            average_travelingtime+=mean(values(package_life))
        end
        writedlm(outfaveragepathlength,transpose([P,average_pathlength/samples]))
        writedlm(outfaveragetravelingtime,transpose([P,average_travelingtime/samples]))
        P+=ΔP
    end
    close(outfaveragepathlength)
    close(outfaveragetravelingtime)
end
