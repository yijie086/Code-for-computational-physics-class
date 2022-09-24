figure(1)
N=10000000;
lambda=1;
s1=rand(1,N);
x=-(1/lambda)*log(s1);
hist(x,100);
hold on;
xexc=0:0.1:10;
yexc=2000000*exp(-xexc);
plot(xexc,yexc,'r','LineWidth',1.8);
xlabel('x','FontSize',15)
ylabel('统计频次','FontSize',15)
set(gca,'FontSize',15)
legend('统计频次','统计频次期望值')
hold off;
answer=sum(x.^(1.5))/N

figure(2)
N=10000000;
x0=0;
gamma=1;
s1=rand(1,N);
x=gamma*tan(pi*(s1-0.5))+x0;
xbins=-22:22;
hist(x,xbins,44);
hold on;
axis([-20 20 0 inf])
xexc=-10:0.1:10;
yexc=(1/pi)*(1./(xexc.^2+1))*N;
plot(xexc,yexc,'r','LineWidth',1.8);
xlabel('x','FontSize',15)
ylabel('统计频次','FontSize',15)
set(gca,'FontSize',15)
legend('统计频次','统计频次期望值')
hold off;
answer=real(sum(pi*x.^(0.5))/N)