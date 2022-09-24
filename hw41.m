x0=2;
t1=3;
dt=0.1;
xstore=zeros(1,t1/dt+1);
xstore(1,1)=x0;
t=0:dt:t1;
for i=1:1:(t1/dt)
    K1=10-3*xstore(1,i);
    K2=10-3*(xstore(1,i)+dt*K1*0.5);
    K3=10-3*(xstore(1,i)+dt*K2*0.5);
    K4=10-3*(xstore(1,i)+dt*K3);
    xstore(1,i+1)=xstore(1,i)+dt*(K1+K2*2+K3*2+K4)/6;
end
plot(t,xstore,'LineWidth',2)
xlabel('t','FontSize',15)
ylabel('x','FontSize',15)
set(gca,'FontSize',15)
hold on
testimate=0:0.01:3;
xestimate=(10-4*exp(-3*testimate))/3;
plot(testimate,xestimate,'LineWidth',2)
hold off
