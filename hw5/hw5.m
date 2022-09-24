L=20;
T=40;
dx=0.1;
dt=0.001;
t=0:dt:T;
x=0:dx:L;
u=zeros(T/dt+1,L/dx+1);
u(1,1:round((L/dx+1)/2))=x(1,1:round((L/dx+1)/2));
u(1,round((L/dx+1)/2):round(L/dx+1))=L-x(1,round((L/dx+1)/2):round(L/dx+1));
for i=1:1:round(T/dt)
    for j=1:1:round(L/dx+1)
        if j==1
            u(i+1,j)=0;
        elseif j==(L/dx+1)
            u(i+1,j)=0;
        else
            ut=(u(i,j+1)+u(i,j-1)-2*u(i,j))/(dx*dx);
            u(i+1,j)=dt*ut+u(i,j);
        end
    end
end
figure(1)
s=surf(x,t,u)
s.EdgeColor = 'none';
xlabel('x','FontSize',18)
ylabel('t','FontSize',18)
zlabel('u','FontSize',18)
set(gca,'FontSize',18)
figure(2)
plot(x,u(round(T/dt+1),:),'LineWidth',2);
xlabel('x','FontSize',18)
ylabel('u','FontSize',18)
set(gca,'FontSize',18)
legend('T=40的温度')
