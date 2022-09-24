N=10000;        %number of particles
steps=3.*10.^4;       %steps of evolution
state=zeros(1,N);   %Left and right states, 0 for L,1 for R
statesmemory=zeros(1,steps+1);    %save states every steps
statesmemory(1,1)=sum(state);
Nstep=1:1:steps+1;
for i=1:steps
    evolutionparticle=round(rand(1,1)*N+0.5); %random evolution
    state(1,evolutionparticle)=rem(1+state(1,evolutionparticle),2);
    statesmemory(1,1+i)=sum(state);
end
estimate=(N/2).*(1-(1-2/N).^(Nstep));
Lestimate=N-estimate;
Lstatesmemory=N-statesmemory;
plot(Nstep,statesmemory,Nstep,Lstatesmemory,'LineWidth',3);
hold on;
plot(Nstep,estimate,':','LineWidth',3);
plot(Nstep,Lestimate,':','LineWidth',3);
xlabel('演化步数','FontSize',15)
ylabel('分子数量','FontSize',15)
set(gca,'FontSize',15)
legend('右侧分子计算结果','左侧分子计算结果','右侧分子期望值','左侧分子期望值')
hold off;